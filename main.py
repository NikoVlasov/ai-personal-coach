import asyncio
import logging
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# --- API CLIENTS ---
from tavily import TavilyClient
from groq import Groq


# =========================
# CONFIG
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY missing")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# =========================
# INIT CLIENTS
# =========================

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# =========================
# FASTAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = HTTPBearer()

# =========================
# DATABASE
# =========================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    chats = relationship("Chat", back_populates="user")


class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    sender = Column(String)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")


Base.metadata.create_all(bind=engine)

# =========================
# SECURITY
# =========================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401)
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=401)
        return user
    except JWTError:
        raise HTTPException(status_code=401)


# =========================
# SCHEMAS
# =========================

class UserRegister(BaseModel):
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class ChatCreate(BaseModel):
    title: str


class MessageRequest(BaseModel):
    chat_id: int
    text: str


# =========================
# STATIC
# =========================

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


# =========================
# AUTH
# =========================

@app.post("/register")
async def register(data: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email exists")

    user = User(email=data.email, password=hash_password(data.password))
    db.add(user)
    db.commit()

    return {"status": "ok"}


@app.post("/login")
async def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()

    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.email})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id
    }


# =========================
# CHATS
# =========================

@app.post("/chats")
async def create_chat(data: ChatCreate,
                      user=Depends(get_current_user),
                      db: Session = Depends(get_db)):
    chat = Chat(user_id=user.id, title=data.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"chat_id": chat.id}


@app.get("/chats")
async def get_chats(user=Depends(get_current_user),
                    db: Session = Depends(get_db)):
    chats = db.query(Chat).filter(Chat.user_id == user.id).order_by(Chat.id.desc()).all()
    return {
        "chats": [{"id": c.id, "title": c.title} for c in chats]
    }


@app.get("/history/{chat_id}")
async def history(chat_id: int,
                  user=Depends(get_current_user),
                  db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=403)

    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.id).all()

    return {
        "messages": [{"sender": m.sender, "text": m.text} for m in messages]
    }


# =========================
# AI COACH
# =========================

@app.post("/coach")
async def coach(msg: MessageRequest,
                user=Depends(get_current_user),
                db: Session = Depends(get_db)):

    chat = db.query(Chat).filter(Chat.id == msg.chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404)

    db.add(Message(chat_id=chat.id, sender="user", text=msg.text))
    db.commit()

    try:
        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": """
                    You are a professional AI assistant.

                    IMPORTANT LINK RULES:
                    1. Only provide official, existing websites.
                    2. Provide ONLY main homepage URLs (e.g. https://www.sixt.com).
                    3. Do NOT generate deep links, tracking links, or long URLs.
                    4. If unsure that a page exists, provide only the company’s main website.
                    5. Never invent URLs.
                    6. Do not include broken or speculative links.
                    """},

                    {"role": "user", "content": msg.text}
                ],
                temperature=0.3
            )
        )

        ai_text = completion.choices[0].message.content.strip()

        db.add(Message(chat_id=chat.id, sender="ai", text=ai_text))
        db.commit()

        return PlainTextResponse(ai_text)

    except Exception as e:
        logger.error(f"COACH ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# SEARCH
# =========================

@app.post("/search")
async def search(msg: MessageRequest,
                 user=Depends(get_current_user),
                 db: Session = Depends(get_db)):

    chat = db.query(Chat).filter(Chat.id == msg.chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404)

    db.add(Message(chat_id=chat.id, sender="user", text=msg.text))
    db.commit()

    try:
        search_results = await asyncio.to_thread(
            lambda: tavily_client.search(
                query=msg.text,
                search_depth="advanced",
                max_results=5
            )
        )

        sources = ""
        for i, r in enumerate(search_results.get("results", []), 1):
            sources += f"{i}. {r.get('title')}\n{r.get('url')}\n{r.get('content')}\n\n"

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": """
                    You are a professional AI assistant.

                    IMPORTANT LINK RULES:
                    1. Only provide official, existing websites.
                    2. Provide ONLY main homepage URLs (e.g. https://www.sixt.com).
                    3. Do NOT generate deep links, tracking links, or long URLs.
                    4. If unsure that a page exists, provide only the company’s main website.
                    5. Never invent URLs.
                    6. Do not include broken or speculative links.
                    """}
                ],
                temperature=0.3
            )
        )

        ai_text = completion.choices[0].message.content.strip()

        db.add(Message(chat_id=chat.id, sender="ai", text=ai_text))
        db.commit()

        return PlainTextResponse(ai_text)

    except Exception as e:
        logger.error(f"SEARCH ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int,
                      user=Depends(get_current_user),
                      db: Session = Depends(get_db)):

    chat = db.query(Chat).filter(
        Chat.id == chat_id,
        Chat.user_id == user.id
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Удаляем сообщения
    db.query(Message).filter(Message.chat_id == chat_id).delete()

    # Удаляем чат
    db.delete(chat)
    db.commit()

    return {"status": "deleted"}