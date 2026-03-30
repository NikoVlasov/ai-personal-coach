import asyncio
import logging
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime, Float
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from urllib.parse import urlparse
from typing import Optional

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
# DATABASE MODELS
# =========================

# СТАЛО:
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,        # проверяет соединение перед использованием
    pool_recycle=300,          # пересоздаёт соединение каждые 5 минут
    connect_args={"sslmode": "require"}
)
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


class FitnessProfile(Base):
    __tablename__ = "fitness_profiles"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    goal = Column(String)        # fat_loss / strength / general
    level = Column(String)       # beginner / intermediate / advanced
    height = Column(Integer)     # cm
    weight = Column(Float)       # kg
    age = Column(Integer, nullable=True)
    limitations = Column(Text, nullable=True)   # injuries / limitations
    days_per_week = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DailyCheckin(Base):
    __tablename__ = "daily_checkins"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    energy = Column(Integer)
    soreness = Column(Integer)
    mood = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class WorkoutLog(Base):
    __tablename__ = "workout_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise = Column(String)
    sets = Column(Integer, nullable=True)
    reps = Column(Integer, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


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


class CheckinRequest(BaseModel):
    chat_id: int
    energy: int      # 1–10
    soreness: int    # 1–10
    mood: int        # 1–10


class FitnessProfileRequest(BaseModel):
    goal: str                        # fat_loss / strength / general
    level: str                       # beginner / intermediate / advanced
    height: int                      # cm
    weight: float                    # kg
    age: Optional[int] = None
    limitations: Optional[str] = None
    days_per_week: Optional[int] = None


class WorkoutLogRequest(BaseModel):
    exercise: str
    sets: Optional[int] = None
    reps: Optional[int] = None
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None


# =========================
# HELPERS
# =========================

def build_profile_context(profile: FitnessProfile) -> str:
    """Converts a FitnessProfile into a readable string for the AI system prompt."""
    if not profile:
        return ""

    goal_map = {
        "fat_loss": "fat loss and body recomposition",
        "strength": "building strength and muscle",
        "general": "general fitness and healthy lifestyle"
    }
    level_map = {
        "beginner": "beginner (little to no training experience)",
        "intermediate": "intermediate (1-2 years of training)",
        "advanced": "advanced (3+ years of training)"
    }

    lines = [
        "\n\n--- USER FITNESS PROFILE ---",
        f"Goal: {goal_map.get(profile.goal, profile.goal)}",
        f"Level: {level_map.get(profile.level, profile.level)}",
        f"Height: {profile.height} cm",
        f"Weight: {profile.weight} kg",
    ]
    if profile.age:
        lines.append(f"Age: {profile.age}")
    if profile.days_per_week:
        lines.append(f"Available training days per week: {profile.days_per_week}")
    if profile.limitations:
        lines.append(f"Physical limitations / injuries: {profile.limitations}")
    lines.append("--- END OF PROFILE ---\n")
    lines.append("Always personalise your advice based on this profile. Reference it naturally without reading it aloud.")

    return "\n".join(lines)


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
    return {"chats": [{"id": c.id, "title": c.title} for c in chats]}


@app.get("/history/{chat_id}")
async def history(chat_id: int,
                  user=Depends(get_current_user),
                  db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=403)
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.id).all()
    return {"messages": [{"sender": m.sender, "text": m.text} for m in messages]}


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int,
                      user=Depends(get_current_user),
                      db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.query(Message).filter(Message.chat_id == chat_id).delete()
    db.delete(chat)
    db.commit()
    return {"status": "deleted"}


# =========================
# FITNESS PROFILE
# =========================

@app.post("/profile")
async def save_profile(data: FitnessProfileRequest,
                       user=Depends(get_current_user),
                       db: Session = Depends(get_db)):
    profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
    if profile:
        # Update existing
        profile.goal = data.goal
        profile.level = data.level
        profile.height = data.height
        profile.weight = data.weight
        profile.age = data.age
        profile.limitations = data.limitations
        profile.days_per_week = data.days_per_week
        profile.updated_at = datetime.utcnow()
    else:
        # Create new
        profile = FitnessProfile(
            user_id=user.id,
            goal=data.goal,
            level=data.level,
            height=data.height,
            weight=data.weight,
            age=data.age,
            limitations=data.limitations,
            days_per_week=data.days_per_week
        )
        db.add(profile)
    db.commit()
    return {"status": "ok"}


@app.get("/profile")
async def get_profile(user=Depends(get_current_user),
                      db: Session = Depends(get_db)):
    profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
    if not profile:
        return {"profile": None}
    return {
        "profile": {
            "goal": profile.goal,
            "level": profile.level,
            "height": profile.height,
            "weight": profile.weight,
            "age": profile.age,
            "limitations": profile.limitations,
            "days_per_week": profile.days_per_week
        }
    }


# =========================
# WORKOUT LOG
# =========================

@app.post("/log")
async def log_workout(data: WorkoutLogRequest,
                      user=Depends(get_current_user),
                      db: Session = Depends(get_db)):
    entry = WorkoutLog(
        user_id=user.id,
        exercise=data.exercise,
        sets=data.sets,
        reps=data.reps,
        duration_minutes=data.duration_minutes,
        notes=data.notes
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "ok", "id": entry.id}


@app.get("/log")
async def get_logs(user=Depends(get_current_user),
                   db: Session = Depends(get_db)):
    logs = db.query(WorkoutLog).filter(
        WorkoutLog.user_id == user.id
    ).order_by(WorkoutLog.created_at.desc()).limit(50).all()

    return {
        "logs": [
            {
                "id": l.id,
                "exercise": l.exercise,
                "sets": l.sets,
                "reps": l.reps,
                "duration_minutes": l.duration_minutes,
                "notes": l.notes,
                "date": l.created_at.strftime("%Y-%m-%d %H:%M")
            }
            for l in logs
        ]
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
        # Load last 20 messages
        db_messages = db.query(Message).filter(
            Message.chat_id == msg.chat_id
        ).order_by(Message.created_at.desc()).limit(20).all()
        db_messages.reverse()

        conversation = []
        for m in db_messages:
            role = "user" if m.sender == "user" else "assistant"
            conversation.append({"role": role, "content": m.text})

        # Load user fitness profile for personalisation
        profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
        profile_context = build_profile_context(profile)

        system_prompt = """You are **WorkoutCoach AI**, an experienced home fitness coach specialising in fat loss, bodyweight training, and sustainable fitness routines without gym equipment.

Your goal is to help the user improve their physical shape, lose fat, and build strength using simple home workouts.

LANGUAGE RULE:
Always respond in the same language as the user. If the user writes in Russian, reply in Russian. If they write in English, reply in English.

PERSONALITY AND STYLE:
Speak like a calm, knowledgeable fitness coach. Supportive, practical, conversational — not robotic. Avoid exaggerated praise.

FIRST MESSAGE RULE:
Start with a simple greeting. Include a brief safety note about consulting a medical professional if they have injuries or chronic conditions.

USER ASSESSMENT PHASE:
If the user has no saved profile, ask up to 4 questions: fitness level, goal, available time, injuries. Use this to adapt recommendations.

WORKOUT APPROACH:
Home workouts only, no equipment. 15–45 minutes. Combine bodyweight strength, cardio, core, and fat-burning circuits.

EXERCISE FORMAT:
Include: exercise name, step-by-step instructions, reps or duration, sets, rest between sets.

NUTRITION GUIDANCE:
Occasionally include simple advice: moderate calorie deficit (~500 kcal/day), prioritise protein and vegetables, reduce sugar, drink water.

RESPONSE FORMAT:
Use markdown formatting — headers (##), bullet points (-), and **bold** for exercise names. This improves readability.

RESPONSE LENGTH:
120–250 words unless the user asks for a detailed program.""" + profile_context

        full_messages = [{"role": "system", "content": system_prompt}] + conversation

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages,
                temperature=0.5,
                max_tokens=600
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
                max_results=8
            )
        )

        raw_results = search_results.get("results", [])
        if not raw_results:
            return PlainTextResponse("No relevant search results found.")

        blocked_domains = ["reddit.com", "quora.com", "pinterest.com", "facebook.com", "instagram.com"]
        unique_domains = set()
        filtered_results = []

        for r in raw_results:
            url = r.get("url")
            if not url:
                continue
            domain = urlparse(url).netloc.lower()
            if any(b in domain for b in blocked_domains):
                continue
            if domain in unique_domains:
                continue
            unique_domains.add(domain)
            filtered_results.append(r)
            if len(filtered_results) >= 5:
                break

        if not filtered_results:
            return PlainTextResponse("No high-quality sources found.")

        sources_text = ""
        for i, r in enumerate(filtered_results, 1):
            sources_text += f"\nSource {i}\nTitle: {r.get('title')}\nURL: {r.get('url')}\nSnippet: {r.get('content')}\n"

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI web research assistant.
STRICT RULES:
- Use ONLY URLs provided in the search results.
- Never invent or modify URLs.
- Include 3–5 real sources in the answer.
- Provide a structured, professional response.
- Use markdown formatting for readability.
- Respond in the same language as the user's question."""
                    },
                    {
                        "role": "user",
                        "content": f"User question:\n{msg.text}\n\nWeb search results:\n{sources_text}\n\nWrite a helpful answer using ONLY these sources. Include their real URLs."
                    }
                ],
                temperature=0.2
            )
        )

        ai_text = completion.choices[0].message.content.strip()

        db.add(Message(chat_id=chat.id, sender="ai", text=ai_text))
        db.commit()

        return PlainTextResponse(ai_text)

    except Exception as e:
        logger.error(f"SEARCH ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# DAILY CHECK-IN
# =========================

@app.post("/checkin")
async def daily_checkin(
    req: CheckinRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = db.query(Chat).filter(Chat.id == req.chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save checkin scores to DB
    checkin = DailyCheckin(
        user_id=user.id,
        energy=req.energy,
        soreness=req.soreness,
        mood=req.mood
    )
    db.add(checkin)
    db.commit()

    # Build a natural user message from the scores
    def score_label(score: int) -> str:
        if score <= 3:
            return "low"
        elif score <= 6:
            return "moderate"
        else:
            return "high"

    checkin_text = (
        f"Daily check-in: "
        f"Energy {req.energy}/10 ({score_label(req.energy)}), "
        f"Muscle soreness {req.soreness}/10 ({score_label(req.soreness)}), "
        f"Mood {req.mood}/10 ({score_label(req.mood)}). "
        f"Based on how I feel today, what should my workout look like?"
    )

    db.add(Message(chat_id=chat.id, sender="user", text=checkin_text))
    db.commit()

    try:
        # Load full chat history
        db_messages = db.query(Message).filter(
            Message.chat_id == req.chat_id
        ).order_by(Message.created_at).all()

        conversation = []
        for m in db_messages:
            role = "user" if m.sender == "user" else "assistant"
            conversation.append({"role": role, "content": m.text})

        # Load profile for personalisation
        profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
        profile_context = build_profile_context(profile)

        system_prompt = """You are DailyCoach AI — an empathetic daily fitness coach.

DAILY CHECK-IN RULES:
- The user has shared their energy, soreness, and mood scores for today.
- Adapt today's workout recommendation to these scores:
  * Low energy (1-4): suggest a light recovery session or stretching
  * Moderate energy (5-7): suggest a normal workout with moderate intensity
  * High energy (8-10): suggest a challenging full session
  * High soreness (7-10): recommend active recovery, avoid the sore muscle groups
  * Low mood (1-4): be extra encouraging, suggest something fun and short
- Start with a brief empathetic acknowledgement of how they feel.
- Recommend one specific workout or recovery activity for today.
- Keep it to 150–220 words.
- Use markdown formatting (bullet points, bold exercise names).
- End with ONE motivating sentence.
- Respond in the SAME LANGUAGE as the conversation.""" + profile_context

        full_messages = [{"role": "system", "content": system_prompt}] + conversation

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages,
                temperature=0.6,
                max_tokens=400
            )
        )

        ai_text = completion.choices[0].message.content.strip()

        db.add(Message(chat_id=chat.id, sender="ai", text=ai_text))
        db.commit()

        return PlainTextResponse(ai_text)

    except Exception as e:
        logger.error(f"CHECKIN ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))






