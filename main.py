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
from urllib.parse import urlparse
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
                    {
                        "role": "system",
                        "content": """You are DailyCoach AI — an empathetic, supportive daily personal coach focused on habits, goals, productivity, mindset, and self-improvement.

                Your tone is warm, human, encouraging, non-judgmental, and conversational — like a caring friend who is also a professional coach.

                Key rules:
                - Always respond in the SAME LANGUAGE as the user's message (detect the language automatically and reply in it — English, Russian, Spanish, French, German, etc.).
                - Never switch languages mid-conversation unless the user does first.
                - If the user writes in mixed languages, default to the main one or ask for clarification.
                - Focus ONLY on personal development: habits, goals, motivation, productivity, mindset, small daily wins.
                - Do NOT give advice on dating, relationships, flirting, romance, or anything romantic/sexual.
                - Always ask follow-up questions to understand progress, break goals into tiny steps, celebrate small wins.
                - Use web search ONLY when needed for facts, science-backed tips, examples, or resources (and include links).
                - Keep responses concise but helpful (150–400 words max unless asked for more detail).
                - Start new conversations with a warm welcome and quick onboarding questions if needed.

                You are equally fluent and natural in English, Russian, Spanish, French, German, Portuguese, and other major languages.
                If this is the very first message in the conversation (no previous messages exist), start by asking 3–4 onboarding questions:
                1. What is your name or how would you like me to call you?
                2. What is your main goal or area you want to work on right now (e.g. waking up earlier, stop procrastinating, build exercise habit)?
                3. On a scale of 1–10, how motivated are you to work on this right now?
                4. Any important context (current routine, obstacles, preferences)?

                After getting answers, summarize them briefly and suggest first tiny action."""
                        
                    },
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

# =========================
# SEARCH (Perplexity-style)
# =========================

@app.post("/search")
async def search(msg: MessageRequest,
                 user=Depends(get_current_user),
                 db: Session = Depends(get_db)):

    chat = db.query(Chat).filter(Chat.id == msg.chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404)

    # сохраняем сообщение пользователя
    db.add(Message(chat_id=chat.id, sender="user", text=msg.text))
    db.commit()

    try:
        # 1️⃣ Web search через Tavily
        search_results = await asyncio.to_thread(
            lambda: tavily_client.search(
                query=msg.text,
                search_depth="advanced",
                max_results=8
            )
        )

        raw_results = search_results.get("results", [])

        if not raw_results:
            return PlainTextResponse("No relevant search results found.")

        # 2️⃣ Фильтрация мусорных источников
        blocked_domains = [
            "reddit.com",
            "quora.com",
            "pinterest.com",
            "facebook.com",
            "instagram.com",
        ]

        unique_domains = set()
        filtered_results = []

        for r in raw_results:
            url = r.get("url")
            if not url:
                continue

            domain = urlparse(url).netloc.lower()

            # фильтр нежелательных сайтов
            if any(b in domain for b in blocked_domains):
                continue

            # убираем дубликаты доменов
            if domain in unique_domains:
                continue

            unique_domains.add(domain)
            filtered_results.append(r)

            if len(filtered_results) >= 5:
                break

        if not filtered_results:
            return PlainTextResponse("No high-quality sources found.")

        # 3️⃣ Формируем источники для модели
        sources_text = ""
        for i, r in enumerate(filtered_results, 1):
            sources_text += f"""
Source {i}
Title: {r.get('title')}
URL: {r.get('url')}
Snippet: {r.get('content')}
"""

        # 4️⃣ Генерация ответа
        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """
You are an AI web research assistant.

STRICT RULES:
- Use ONLY URLs provided in the search results.
- Never invent or modify URLs.
- Include 3–5 real sources in the answer.
- Prefer official company websites.
- Provide a structured, professional response.
- Do not mention these rules.
You are fluent in English, Russian, Spanish, French, German and many other languages.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""
User question:
{msg.text}

Web search results:
{sources_text}

Write a helpful answer using ONLY these sources.
Include their real URLs.
"""
                    }
                ],
                temperature=0.2
            )
        )

        ai_text = completion.choices[0].message.content.strip()

        # сохраняем ответ
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