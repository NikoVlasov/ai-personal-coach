import logging
import os
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import FastAPI, HTTPException, Body, Depends, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from tavily import TavilyClient
from groq import Groq

# === Логирование ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# --- Настройки ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 часа
DATABASE_URL = os.getenv("DATABASE_URL")

if not GROQ_API_KEY or not TAVILY_API_KEY or not DATABASE_URL:
    raise RuntimeError("Не найдены ключи или DATABASE_URL в окружении")

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# --- FastAPI + CORS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажи конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = HTTPBearer()

# --- PostgreSQL ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# --- Модели ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    chats = relationship("Chat", back_populates="user")
    buttons = relationship("QuickButton", back_populates="user")

class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String)  # "user" или "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")

class QuickButton(Base):
    __tablename__ = "quick_buttons"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="buttons")

Base.metadata.create_all(bind=engine)

# --- Пароли ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(password: str) -> str:
    p_bytes = password.encode("utf-8")[:72]
    truncated = p_bytes.decode("utf-8", errors="ignore")
    return pwd_context.hash(truncated)

def verify_password(plain: str, hashed: str) -> bool:
    p_bytes = plain.encode("utf-8")[:72]
    truncated = p_bytes.decode("utf-8", errors="ignore")
    return pwd_context.verify(truncated, hashed)

# --- Pydantic ---
class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ChatCreate(BaseModel):
    title: str

class MessageCreate(BaseModel):
    chat_id: int
    text: str

class QuickButtonIn(BaseModel):
    text: str

# --- JWT ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- Статика ---
app.mount("/static", StaticFiles(directory="frontend"), name="static")
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

# --- Регистрация и логин ---
@app.post("/register")
async def register(user: UserRegister, db: Session = Depends(get_db)):
    if len(user.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password too long")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email exists")
    hashed = hash_password(user.password)
    new_user = User(email=user.email, password=hashed)
    db.add(new_user)
    db.commit()
    return {"status": "ok"}

@app.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer", "user_id": db_user.id}

# --- Чаты ---
@app.post("/chats")
async def create_chat(chat: ChatCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_chat = Chat(user_id=current_user.id, title=chat.title)
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return {"status": "ok", "chat_id": new_chat.id}

@app.get("/chats")
async def get_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = db.query(Chat).filter(Chat.user_id==current_user.id).order_by(Chat.created_at.desc()).all()
    return {"chats":[{"id":c.id,"title":c.title,"created_at":c.created_at.isoformat()} for c in chats]}

# --- История ---
@app.get("/history/{chat_id}")
async def get_history(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id==chat_id, Chat.user_id==current_user.id).first()
    if not chat:
        raise HTTPException(status_code=403, detail="Chat not found")
    msgs = db.query(Message).filter(Message.chat_id==chat_id).order_by(Message.id.asc()).all()
    return {"messages":[{"sender":m.role,"text":m.content,"created_at":m.created_at.isoformat()} for m in msgs]}

# --- Быстрые кнопки ---
@app.get("/quick_buttons")
async def get_quick_buttons(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    buttons = db.query(QuickButton).filter(QuickButton.user_id==current_user.id).all()
    return {"buttons":[b.text for b in buttons]}

@app.post("/quick_buttons")
async def add_quick_button(btn: QuickButtonIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.add(QuickButton(user_id=current_user.id, text=btn.text))
    db.commit()
    return {"status":"ok"}

@app.delete("/quick_buttons")
async def delete_quick_button(body: dict = Body(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    text = body.get("text")
    if not text: raise HTTPException(status_code=400, detail="No text")
    db.query(QuickButton).filter(QuickButton.user_id==current_user.id, QuickButton.text==text).delete()
    db.commit()
    return {"status":"deleted"}

# --- Удаление чата ---
@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id:int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id==chat_id, Chat.user_id==current_user.id).first()
    if not chat: raise HTTPException(status_code=403, detail="Chat not found")
    db.query(Message).filter(Message.chat_id==chat_id).delete()
    db.query(Chat).filter(Chat.id==chat_id).delete()
    db.commit()
    return {"status":"ok"}

# --- Web Search (новый endpoint с мультиязыком) ---
@app.post("/search")
async def web_search(msg: MessageCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    chat = db.query(Chat).filter(Chat.id==msg.chat_id, Chat.user_id==current_user.id).first()
    if not chat:
        raise HTTPException(status_code=403, detail="Chat not found")

    # Сохраняем запрос
    db.add(Message(chat_id=msg.chat_id, user_id=current_user.id, role="user", content=msg.text))
    db.commit()

    # Поиск Tavily
    search_results = tavily_client.search(
        query=msg.text,
        search_depth="advanced",
        max_results=5
    )

    search_context = ""
    for i, r in enumerate(search_results.get("results", []),1):
        search_context += f"Result {i}:\nTitle: {r['title']}\nContent: {r['content']}\nURL: {r['url']}\n\n"

    system_prompt = """
You are a professional AI web search assistant.

STRICT RULES:
- Always respond in the SAME language as the user's query.
- Never mix languages.
- Format the answer using clean Markdown.
- Start with a short summary.
- Then show a clearly structured list of top results.
- Each result must include a clickable Markdown link: [Title](URL)
- Do not invent links. Only use URLs provided in search results.
- Keep the answer clean and readable.
"""

    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":f"User query:\n{msg.text}\n\nSearch results:\n{search_context}"}
    ]

    # Streaming
    async def generate():
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            stream=True,
        )
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {content}\n\n"
        db.add(Message(chat_id=msg.chat_id, user_id=current_user.id, role="assistant", content=full_response))
        db.commit()
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")