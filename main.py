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
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from urllib.parse import urlparse
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
    fitness_profile = relationship("FitnessProfile", uselist=False, back_populates="user")
    checkins = relationship("DailyCheckin", back_populates="user")

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
    user_id = Column(Integer, ForeignKey("users.id"))
    goal = Column(String)          # fat loss / strength / general fitness
    level = Column(String)         # beginner / intermediate
    height = Column(Integer)
    weight = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="fitness_profile")

class DailyCheckin(Base):
    __tablename__ = "daily_checkins"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    energy = Column(Integer)
    soreness = Column(Integer)
    mood = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="checkins")

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

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):
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

class FitnessProfileUpdate(BaseModel):
    goal: str
    level: str
    height: int
    weight: int

class CheckinRequest(BaseModel):
    chat_id: int
    energy: int
    soreness: int
    mood: int

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
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}

# =========================
# FITNESS PROFILE
# =========================
@app.post("/profile")
async def update_profile(profile: FitnessProfileUpdate,
                         user: User = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    existing = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
    if existing:
        existing.goal = profile.goal
        existing.level = profile.level
        existing.height = profile.height
        existing.weight = profile.weight
    else:
        fp = FitnessProfile(user_id=user.id,
                            goal=profile.goal,
                            level=profile.level,
                            height=profile.height,
                            weight=profile.weight)
        db.add(fp)
    db.commit()
    return {"status": "ok"}

# =========================
# CHATS
# =========================
@app.post("/chats")
async def create_chat(data: ChatCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    chat = Chat(user_id=user.id, title=data.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"chat_id": chat.id}

@app.get("/chats")
async def get_chats(user=Depends(get_current_user), db: Session = Depends(get_db)):
    chats = db.query(Chat).filter(Chat.user_id == user.id).order_by(Chat.id.desc()).all()
    return {"chats": [{"id": c.id, "title": c.title} for c in chats]}

@app.get("/history/{chat_id}")
async def history(chat_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=403)
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.id).all()
    return {"messages": [{"sender": m.sender, "text": m.text} for m in messages]}

# =========================
# AI COACH (с прогрессом и недельным планом)
# =========================
@app.post("/coach")
async def coach(msg: MessageRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == msg.chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404)
    db.add(Message(chat_id=chat.id, sender="user", text=msg.text))
    db.commit()

    try:
        # Получаем последние 20 сообщений для контекста
        db_messages = db.query(Message).filter(Message.chat_id == msg.chat_id)\
            .order_by(Message.created_at.desc()).limit(20).all()
        db_messages.reverse()

        conversation = [{"role": "user" if m.sender=="user" else "assistant", "content": m.text} for m in db_messages]

        # Добавляем system prompt
        system_prompt = {
            "role": "system",
            "content": f"""
You are WorkoutCoach AI — calm and knowledgeable home fitness coach.
Use user's profile and previous check-ins to generate personalized weekly plans.
Maintain language of user. Keep tone professional, practical, and supportive.
Provide exercises (name, instructions, sets, reps, rest) and simple nutrition advice.
Track user's progress across chats.
"""
        }

        full_messages = [system_prompt] + conversation

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages,
                temperature=0.5,
                max_tokens=800
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
# DAILY CHECKIN
# =========================
@app.post("/checkin")
async def daily_checkin(req: CheckinRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == req.chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404)

    db.add(DailyCheckin(user_id=user.id, energy=req.energy, soreness=req.soreness, mood=req.mood))
    db.commit()

    checkin_text = (
        f"Daily check-in data: Energy={req.energy}, Soreness={req.soreness}, Mood={req.mood}"
    )
    db.add(Message(chat_id=chat.id, sender="user", text=checkin_text))
    db.commit()

    # Подготовка AI с учётом новых данных
    try:
        db_messages = db.query(Message).filter(Message.chat_id == req.chat_id).order_by(Message.created_at).all()
        conversation = [{"role": "user" if m.sender=="user" else "assistant", "content": m.text} for m in db_messages]

        system_prompt = {
            "role": "system",
            "content": """
You are DailyCoach AI — empathetic daily personal coach.
Use user's recent check-in to suggest next steps and track energy/progress.
Respond in user's language. Keep it short (100–200 words) and actionable.
"""
        }

        full_messages = [system_prompt] + conversation

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