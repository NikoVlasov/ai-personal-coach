import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
import asyncio
from tavily import TavilyClient
from groq import Groq
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import FastAPI, HTTPException, Body, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi import Request, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

# === Логирование ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# --- Настройки ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
DATABASE_URL = os.getenv("DATABASE_URL")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY не найден")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY не найден")

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# --- FastAPI + CORS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = HTTPBearer()

# --- PostgreSQL ---
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL не найден")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
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
    sender = Column(String)  # "user" или "ai"
    text = Column(Text)
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
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    truncated = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.hash(truncated)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    truncated = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.verify(truncated, hashed_password)

# --- Pydantic модели ---
class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ChatCreate(BaseModel):
    title: str

class MessageIn(BaseModel):
    chat_id: int
    text: str

class QuickButtonIn(BaseModel):
    text: str

class MessageRequest(BaseModel):
    chat_id: int
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

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
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

# --- Регистрация / Логин ---
@app.post("/register")
async def register(user: UserRegister, db: Session = Depends(get_db)):
    if len(user.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password too long (max 72 bytes)")
    hashed_password = hash_password(user.password)
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(email=user.email, password=hashed_password)
    db.add(new_user)
    db.commit()
    logger.info(f"User registered: {user.email}")
    return {"status": "ok"}

@app.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": db_user.email})
    logger.info(f"User logged in: {db_user.email}")
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
    chats = db.query(Chat).filter(Chat.user_id == current_user.id).order_by(Chat.created_at.desc()).all()
    return {"chats": [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in chats]}

# --- AI Coach / Search (мультиязычный) ---
@app.post("/search")
async def web_search(
    msg: MessageRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):

    chat = db.query(Chat).filter(
        Chat.id == msg.chat_id,
        Chat.user_id == user.id
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Сохраняем сообщение пользователя
    user_message = Message(
        chat_id=chat.id,
        sender="user",
        text=msg.text
    )
    db.add(user_message)
    db.commit()

    try:
        # 🔎 Tavily без raw_content (стабильно)
        search_results = tavily_client.search(
            query=msg.text,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )

        results = search_results.get("results", [])

        if not results:
            ai_response = "No search results found."
        else:
            sources_text = ""

            for i, result in enumerate(results, 1):
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")

                # 🔒 Ограничиваем размер (чтобы не упереться в токены)
                content = content[:2000]

                sources_text += (
                    f"Source {i}:\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Content:\n{content}\n\n"
                )

            # 🤖 Groq
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional research assistant.\n"
                            "Write a detailed structured Markdown answer.\n"
                            "Do NOT write 'Read more'.\n"
                            "Always show full URLs."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User query:\n{msg.text}\n\n"
                            f"Sources:\n{sources_text}"
                        )
                    }
                ],
                temperature=0.3
            )

            ai_response = completion.choices[0].message.content.strip()

        # Сохраняем ответ
        ai_message = Message(
            chat_id=chat.id,
            sender="ai",
            text=ai_response
        )
        db.add(ai_message)
        db.commit()

        # Стримим
        async def stream():
            yield f"data: {ai_response}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    except Exception as e:
        print("SEARCH ERROR:", str(e))
        raise HTTPException(status_code=500, detail="Search failed")

# --- История ---
@app.get("/history/{chat_id}")
async def get_history(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=403, detail="Chat not found or access denied")
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.id.asc()).all()
    return {"messages": [{"sender": m.sender, "text": m.text, "created_at": m.created_at.isoformat()} for m in messages]}

# --- Быстрые кнопки ---
@app.get("/quick_buttons")
async def get_quick_buttons(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    buttons = db.query(QuickButton).filter(QuickButton.user_id == current_user.id).all()
    return {"buttons": [b.text for b in buttons]}

@app.post("/quick_buttons")
async def add_quick_button(btn: QuickButtonIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.add(QuickButton(user_id=current_user.id, text=btn.text))
    db.commit()
    return {"status": "ok"}

@app.delete("/quick_buttons")
async def delete_quick_button(body: dict = Body(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    text = body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="No button text provided")
    db.query(QuickButton).filter(QuickButton.user_id == current_user.id, QuickButton.text == text).delete()
    db.commit()
    return {"status": "deleted"}

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=403, detail="Chat not found or access denied")
    db.query(Message).filter(Message.chat_id == chat_id).delete()
    db.query(Chat).filter(Chat.id == chat_id).delete()
    db.commit()
    return {"status": "ok"}