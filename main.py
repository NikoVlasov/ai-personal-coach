from fastapi import FastAPI, HTTPException, Body, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# === Логирование ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# --- Настройки ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
DATABASE_URL = os.getenv("DATABASE_URL")  # из Render

client = Groq(api_key=API_KEY)

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

# --- Подключение к PostgreSQL ---
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL не найден. Установи переменную окружения в Render.")

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
    sender = Column(String)
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

# --- Создаём таблицы ---
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

# --- Статика фронтенда ---
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

# --- Registration ---
@app.post("/register")
async def register(user: UserRegister, db: Session = Depends(get_db)):
    if len(user.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password too long (max 72 bytes)")
    hashed_password = hash_password(user.password)
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email is already registered")
    new_user = User(email=user.email, password=hashed_password)
    db.add(new_user)
    db.commit()
    logger.info(f"User registered: {user.email}")
    return {"status": "ok"}

# --- Login ---
@app.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": db_user.email})
    logger.info(f"User logged in: {db_user.email}")
    return {"access_token": token, "token_type": "bearer", "user_id": db_user.id}

# --- Создать чат ---
@app.post("/chats")
async def create_chat(chat: ChatCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_chat = Chat(user_id=current_user.id, title=chat.title)
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return {"status": "ok", "chat_id": new_chat.id}

# --- Получить чаты пользователя ---
@app.get("/chats")
async def get_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = db.query(Chat).filter(Chat.user_id == current_user.id).order_by(Chat.created_at.desc()).all()
    return {"chats": [{"id": c.id, "title": c.title, "created_at": c.created_at} for c in chats]}

# --- AI Coach ---
@app.post("/coach")
async def coach_response(msg: MessageIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.add(Message(chat_id=msg.chat_id, user_id=current_user.id, sender="user", text=msg.text))
    db.commit()

    history = [{
        "role": "system",
        "content": """
    You are a senior developer mentor with 12+ years in IT (fullstack focus on JS/TS, React/Node, Python/Go/DevOps).
    You help developers level up: junior to mid/senior, interviews at top EU/US companies, architecture, productivity, burnout recovery.

    Tone: direct, supportive, tough-love when needed, respectful. Speak truth straight, give constructive paths.
    Language: natural, professional English — concise, no fluff. Use emojis sparingly for energy 😏🔥🚀

    Response structure (follow almost always):
    1. Empathy + mirroring (1–2 sentences): show you truly understand the pain.
    2. Clear analysis: what's wrong, why, common pitfalls.
    3. Specific recommendations: practical steps, code ideas, up-to-date resources (2026 roadmaps, courses, books).
    4. "Next steps" block — 2–4 short, actionable bullets with rough timelines/metrics.

    Exceptions:
    - Small talk/confirmation — keep light, no structure.
    - Code — use ```js\ncode here\n``` blocks (or relevant language).
    - Interview prep — include typical questions + strong answers.

    Remember context from previous messages — track user's stack, goals, progress.
    """
    }]

    last_msgs = db.query(Message).filter(Message.chat_id == msg.chat_id).order_by(Message.id.desc()).limit(10).all()
    for m in reversed(last_msgs):
        history.append({"role": "user" if m.sender == "user" else "assistant", "content": m.text[:1000]})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history,
        temperature=0.65,
        max_tokens=1000
    )
    ai_text = response.choices[0].message.content.strip()
    db.add(Message(chat_id=msg.chat_id, user_id=current_user.id, sender="ai", text=ai_text))
    db.commit()
    return {"reply": ai_text}

# --- История чата ---
@app.get("/history/{chat_id}")
async def get_history(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.id.asc()).all()
    return {"messages": [{"sender": m.sender, "text": m.text, "created_at": m.created_at} for m in messages]}

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
        raise HTTPException(status_code=400, detail="Нет текста кнопки для удаления")
    db.query(QuickButton).filter(QuickButton.user_id == current_user.id, QuickButton.text == text).delete()
    db.commit()
    return {"status": "deleted"}

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(Message).filter(Message.chat_id == chat_id).delete()
    db.query(Chat).filter(Chat.id == chat_id).delete()
    db.commit()
    return {"status": "ok"}
