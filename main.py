import asyncio
import logging
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
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

if not GROQ_API_KEY or not TAVILY_API_KEY or not DATABASE_URL:
    raise RuntimeError("Missing env variables")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# =========================
# CLIENTS
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
    feedbacks = relationship("TrialFeedback", back_populates="user")

class FitnessProfile(Base):
    __tablename__ = "fitness_profiles"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    goal = Column(String)
    level = Column(String)
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

class TrialFeedback(Base):
    __tablename__ = "trial_feedbacks"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    difficulty = Column(Integer)
    soreness_area = Column(String)
    energy_mood = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="feedbacks")

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
    from jose import jwt
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):
    from jose import jwt, JWTError
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
from pydantic import BaseModel

class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class FitnessProfileUpdate(BaseModel):
    goal: str
    level: str
    height: int
    weight: int

class FeedbackRequest(BaseModel):
    difficulty: int
    soreness_area: str
    energy_mood: str

class MessageRequest(BaseModel):
    chat_id: int
    text: str

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
# CHAT
# =========================
@app.post("/chats")
async def create_chat(user=Depends(get_current_user), db: Session = Depends(get_db)):
    chat = Chat(user_id=user.id, title="Home Fitness")
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"chat_id": chat.id, "title": chat.title}

@app.get("/chats")
async def get_chats(user=Depends(get_current_user), db: Session = Depends(get_db)):
    chats = db.query(Chat).filter(Chat.user_id == user.id).all()
    return {"chats": [{"id": c.id, "title": c.title} for c in chats]}

@app.get("/history/{chat_id}")
async def history(chat_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=403)
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.id).all()
    return {"messages": [{"sender": m.sender, "text": m.text} for m in messages]}

# =========================
# TRIAL TRAINING & FEEDBACK
# =========================
@app.post("/start-trial")
async def start_trial(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.user_id == user.id).first()
    if not chat:
        chat = Chat(user_id=user.id, title="Home Fitness")
        db.add(chat)
        db.commit()
        db.refresh(chat)

    # Сообщение ИИ с пробной тренировкой
    trial_text = (
        "Привет! 💪 Давай начнем с короткой 5-минутной пробной тренировки:\n"
        "- 10 приседаний\n- 20 секунд планка\n- 10 прыжков\n\n"
        "Выполнил? После тренировки расскажи, как прошло."
    )
    db.add(Message(chat_id=chat.id, sender="ai", text=trial_text))
    db.commit()
    return {"chat_id": chat.id, "text": trial_text}

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest,
                          user: User = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    db.add(TrialFeedback(
        user_id=user.id,
        difficulty=feedback.difficulty,
        soreness_area=feedback.soreness_area,
        energy_mood=feedback.energy_mood
    ))
    db.commit()

    # Генерация персонального плана с учетом профиля и обратной связи
    fp = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
    if not fp:
        raise HTTPException(status_code=400, detail="Profile not set")

    try:
        conversation = [
            {"role": "system",
             "content": f"""
You are HomeFitness AI — friendly and practical coach.
Use user's profile:
- Goal: {fp.goal}
- Level: {fp.level}
- Height: {fp.height} cm
- Weight: {fp.weight} kg

Use the feedback from trial training:
- Difficulty: {feedback.difficulty}
- Soreness: {feedback.soreness_area}
- Energy/Mood: {feedback.energy_mood}

Generate a **weekly fitness plan**, short instructions, minimal text, actionable steps for home workouts. Keep it supportive and motivating.
"""}]

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=conversation,
                temperature=0.6,
                max_tokens=700
            )
        )
        plan_text = completion.choices[0].message.content.strip()
        chat = db.query(Chat).filter(Chat.user_id == user.id).first()
        db.add(Message(chat_id=chat.id, sender="ai", text=plan_text))
        db.commit()
        return PlainTextResponse(plan_text)
    except Exception as e:
        logger.error(f"FEEDBACK ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# DAILY CHECK-IN
# =========================
@app.post("/checkin")
async def daily_checkin(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404)

    # Можно заполнять дефолтными значениями или фронтом
    db.add(DailyCheckin(user_id=user.id, energy=5, soreness=2, mood=5))
    db.commit()

    # Сообщение ИИ после check-in
    ai_text = "Отлично! Сегодняшний день отмечен. Продолжай в том же духе 💪"
    db.add(Message(chat_id=chat.id, sender="ai", text=ai_text))
    db.commit()
    return PlainTextResponse(ai_text)