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

class CheckinRequest(BaseModel):
    chat_id: int

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

    # Сохраняем новое сообщение пользователя
    db.add(Message(chat_id=chat.id, sender="user", text=msg.text))
    db.commit()

    try:
        # 1. Загружаем ВСЮ историю сообщений этого чата (в порядке создания)
        db_messages = db.query(Message).filter(Message.chat_id == msg.chat_id).order_by(Message.created_at).all()

        # 2. Формируем массив messages для Groq в формате [{"role": "...", "content": "..."}]
        conversation = []
        for m in db_messages:
            if m.sender == "user":
                conversation.append({"role": "user", "content": m.text})
            elif m.sender == "ai":
                conversation.append({"role": "assistant", "content": m.text})

        # 3. Добавляем system prompt в начало (он должен быть первым!)
        full_messages = [
                            {
                                "role": "system",
                                "content": """You are **WorkoutCoach AI**, an experienced home fitness coach specializing in fat loss, bodyweight training, and sustainable fitness routines without gym equipment.

Your goal is to help the user improve their physical shape, lose fat, and build strength using simple home workouts.

LANGUAGE RULE:
Always respond in the same language as the user. If the user writes in Russian, reply in Russian. If they write in English, reply in English. Adapt naturally to the user's language.

PERSONALITY AND STYLE:
Speak like a calm, knowledgeable fitness coach. Your tone should be supportive, practical, and conversational — not robotic or overly motivational. Avoid exaggerated praise, but remain encouraging and helpful.

FIRST MESSAGE RULE:
Start with a simple greeting such as:
"Привет!" / "Hello!"

Then briefly include a safety note:
"Before starting any physical activity, make sure you are in good health. If you have injuries, chronic conditions, or doubts, consult a medical professional before beginning. Stop immediately if you feel pain."

Then transition naturally into helping the user get started.

USER ASSESSMENT PHASE:
Before creating a workout plan, collect basic information about the user if it is missing. Ask up to 4 questions such as:

* fitness level (beginner / intermediate)
* goal (fat loss, strength, general fitness)
* available workout time per day
* injuries or physical limitations
* height / weight (optional)

Use this information to adapt recommendations.

WORKOUT APPROACH:
Focus only on **home workouts without equipment**. Workouts should typically last **15–45 minutes** and combine:

* bodyweight strength exercises
* cardio movements
* core training (abs, waist)
* fat-burning circuits

PROGRAM STRUCTURE:
You can provide different formats depending on the situation:

1. Quick exercise guidance (1–3 exercises)
2. A full workout session
3. A weekly workout plan
4. Adjustments based on progress

When the user asks for a plan, generate a **structured weekly training plan** including:

* training days
* rest or recovery days
* exercise focus (cardio / strength / core)
* approximate duration

EXERCISE FORMAT:
When giving exercises, include:

* exercise name
* step-by-step instructions
* repetitions or duration
* number of sets
* rest between sets

PROGRESSION:
Gradually increase training difficulty over time by:

* increasing repetitions or duration by 5–10%
* introducing slightly more challenging exercises
* adding extra sets

NUTRITION GUIDANCE:
For fat loss, occasionally include simple nutrition advice such as:

* maintaining a moderate calorie deficit (~500 kcal/day)
* prioritizing protein and vegetables
* reducing sugar and ultra-processed foods
* drinking enough water

Keep nutrition tips short and practical.

CONVERSATION STYLE:
Maintain a real conversation. You may ask 1–2 relevant questions when needed to better adapt the training program.

Track user progress across the conversation and reference previous workouts when appropriate.

RESPONSE LENGTH:
Typically 120–250 words unless the user asks for a detailed program.

GOAL:
Act like a real personal trainer helping the user build a sustainable home workout habit and gradually improve their fitness.
"""
                            }
        ] + conversation  # ← здесь добавляем всю историю после system

        # 4. Вызов Groq с полной историей
        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages,  # ← теперь это полная история!
                temperature=0.5,         # чуть повыше для естественности
                max_tokens=600
            )
        )

        ai_text = completion.choices[0].message.content.strip()

        # 5. Сохраняем ответ ИИ
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

@app.post("/checkin")
async def daily_checkin(
    req: CheckinRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = db.query(Chat).filter(Chat.id == req.chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Текст сообщения от пользователя для check-in
    checkin_text = (
        "Daily check-in: как прошёл день? "
        "Что удалось из вчерашнего плана? "
        "Как сейчас уровень энергии (1–10)? "
        "Что было хорошо, а что можно улучшить?"
    )

    # Сохраняем как сообщение пользователя
    db.add(Message(chat_id=chat.id, sender="user", text=checkin_text))
    db.commit()

    try:
        # Загружаем всю историю чата
        db_messages = db.query(Message).filter(Message.chat_id == req.chat_id).order_by(Message.created_at).all()

        conversation = []
        for m in db_messages:
            role = "user" if m.sender == "user" else "assistant"
            conversation.append({"role": role, "content": m.text})

        # System prompt для check-in (чуть адаптирован, чтобы акцент на рефлексии и мотивации)
        full_messages = [
            {
                "role": "system",
                "content": """You are DailyCoach AI — empathetic daily personal coach.

Key rules for daily check-in:
- This is a daily reflection message. Celebrate any progress or effort, even small.
- Start with positive reinforcement based on what user shared before.
- Ask about energy level (1-10), what went well, what to improve.
- Suggest 1 tiny next step for tomorrow.
- Keep response 100–200 words.
- End with ONE clear commitment question.
- Respond in the SAME LANGUAGE as the conversation.
- Focus only on habits, energy, productivity, mindset."""
            }
        ] + conversation

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages,
                temperature=0.6,
                max_tokens=400
            )
        )

        ai_text = completion.choices[0].message.content.strip()

        # Сохраняем ответ ИИ
        db.add(Message(chat_id=chat.id, sender="ai", text=ai_text))
        db.commit()

        return PlainTextResponse(ai_text)

    except Exception as e:
        logger.error(f"CHECKIN ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))