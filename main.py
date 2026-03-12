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
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
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
    fitness_profile = relationship("FitnessProfile", uselist=False, back_populates="user")
    messages = relationship("Message", back_populates="user")
    checkins = relationship("DailyCheckin", back_populates="user")

class FitnessProfile(Base):
    __tablename__ = "fitness_profiles"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    goal = Column(String)      # fat loss / strength / general fitness
    level = Column(String)     # beginner / intermediate
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

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    sender = Column(String)  # "user" or "ai"
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="messages")

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

class FitnessProfileUpdate(BaseModel):
    goal: str
    level: str
    height: int
    weight: int

class CheckinRequest(BaseModel):
    energy: int
    soreness: int
    mood: int

class MessageRequest(BaseModel):
    text: str

# =========================
# STATIC FILES
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
# AI COACH (с пробной тренировкой и сопровождением)
# =========================
@app.post("/coach")
async def coach(msg: MessageRequest,
                user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):

    db.add(Message(user_id=user.id, sender="user", text=msg.text))
    db.commit()

    try:
        db_messages = db.query(Message).filter(Message.user_id == user.id)\
            .order_by(Message.created_at.desc()).limit(50).all()
        db_messages.reverse()
        conversation = [{"role": "user" if m.sender=="user" else "assistant", "content": m.text} for m in db_messages]

        profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
        profile_text = f"User profile: goal={profile.goal}, level={profile.level}, height={profile.height}, weight={profile.weight}."

        system_prompt = {
            "role": "system",
            "content": f"""
You are HomeFitnessCoach AI — calm, knowledgeable, supportive.
Do not ask about user's goal, level, height, or weight — they are already provided.
Start by suggesting a short trial workout.
After the trial workout, ask minimal follow-up questions (energy, soreness, mood).
Use this info to generate personalized weekly plan.
Keep language concise, actionable, motivational.
{profile_text}
"""
        }

        full_messages = [system_prompt] + conversation

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages,
                temperature=0.6,
                max_tokens=800
            )
        )
        ai_text = completion.choices[0].message.content.strip()
        db.add(Message(user_id=user.id, sender="ai", text=ai_text))
        db.commit()
        return PlainTextResponse(ai_text)

    except Exception as e:
        logger.error(f"COACH ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# DAILY CHECK-IN
# =========================
@app.post("/checkin")
async def daily_checkin(req: CheckinRequest,
                        user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):

    db.add(DailyCheckin(user_id=user.id, energy=req.energy, soreness=req.soreness, mood=req.mood))
    db.commit()

    checkin_text = f"Daily check-in: Energy={req.energy}, Soreness={req.soreness}, Mood={req.mood}"
    db.add(Message(user_id=user.id, sender="user", text=checkin_text))
    db.commit()

    try:
        db_messages = db.query(Message).filter(Message.user_id == user.id).order_by(Message.created_at).all()
        conversation = [{"role": "user" if m.sender=="user" else "assistant", "content": m.text} for m in db_messages]

        profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()
        profile_text = f"User profile: goal={profile.goal}, level={profile.level}, height={profile.height}, weight={profile.weight}."

        system_prompt = {
            "role": "system",
            "content": f"""
You are DailyCoach AI — empathetic home fitness coach.
Use recent check-in data and user's profile to provide encouragement, motivation, and actionable tips for today.
Keep response short (100–200 words), friendly, actionable.
{profile_text}
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
        db.add(Message(user_id=user.id, sender="ai", text=ai_text))
        db.commit()
        return PlainTextResponse(ai_text)

    except Exception as e:
        logger.error(f"CHECKIN ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_trial")
async def start_trial(msg: MessageRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        # Добавляем сообщение пользователя в чат
        db.add(Message(user_id=user.id, sender="user", text="Start trial"))
        db.commit()

        system_prompt = {
            "role": "system",
            "content": """
You are HomeFitnessCoach AI — calm, knowledgeable, supportive.
Provide a short 5-10 min trial workout, then ask 2-3 simple questions about how user felt.
Keep tone friendly and actionable.
Use user's profile if available, do not repeat already known info (goal, height, weight).
"""
        }

        full_messages = [system_prompt]

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages,
                temperature=0.6,
                max_tokens=500
            )
        )
        ai_text = completion.choices[0].message.content.strip()
        db.add(Message(user_id=user.id, sender="ai", text=ai_text))
        db.commit()
        return PlainTextResponse(ai_text)

    except Exception as e:
        logger.error(f"START TRIAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))