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
from groq import Groq


# =========================
# CONFIG
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# =========================
# INIT CLIENTS
# =========================

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


class DailyCheckin(Base):
    __tablename__ = "daily_checkins"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    energy = Column(Integer)
    soreness = Column(Integer)
    mood = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)

class CheckinRequest(BaseModel):
    energy: int
    soreness: int
    mood: int



class UserStats(Base):
    __tablename__ = "user_stats"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)

    last_workout_date = Column(DateTime, nullable=True)

    total_workouts = Column(Integer, default=0)

class WorkoutLog(Base):
    __tablename__ = "workout_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    duration = Column(Integer)  # минуты
    completed = Column(Integer, default=1)

    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

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

@app.post("/checkin")
async def checkin(
    data: CheckinRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    checkin = DailyCheckin(
        user_id=user.id,
        energy=data.energy,
        soreness=data.soreness,
        mood=data.mood
    )

    db.add(checkin)
    db.commit()

    return {"status": "saved"}

@app.post("/workout")
async def generate_workout(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 1. Берём последний checkin
    checkin = db.query(DailyCheckin)\
        .filter(DailyCheckin.user_id == user.id)\
        .order_by(DailyCheckin.created_at.desc())\
        .first()

    if not checkin:
        raise HTTPException(status_code=400, detail="No checkin")

    # 2. Простая логика (пока без AI)
    if checkin.energy <= 3:
        workout = [
            {"name": "Walking in place", "duration": "5 min"},
            {"name": "Light squats", "reps": "10 x 2"},
            {"name": "Stretching", "duration": "5 min"}
        ]
    elif checkin.energy >= 7:
        workout = [
            {"name": "Jump squats", "reps": "15 x 3"},
            {"name": "Push-ups", "reps": "12 x 3"},
            {"name": "Plank", "duration": "45 sec x 3"}
        ]
    else:
        workout = [
            {"name": "Squats", "reps": "12 x 3"},
            {"name": "Push-ups", "reps": "10 x 3"},
            {"name": "Plank", "duration": "30 sec x 3"}
        ]

    return {
        "workout": workout
    }

@app.post("/complete-workout")
async def complete_workout(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    stats = db.query(UserStats).filter(UserStats.user_id == user.id).first()

    if not stats:
        stats = UserStats(user_id=user.id)
        db.add(stats)

    today = datetime.utcnow().date()

    # Проверка streak
    if stats.last_workout_date:
        last_date = stats.last_workout_date.date()

        if last_date == today - timedelta(days=1):
            stats.current_streak += 1
        elif last_date == today:
            return {"status": "already_done"}
        else:
            stats.current_streak = 1
    else:
        stats.current_streak = 1

    stats.last_workout_date = datetime.utcnow()
    stats.total_workouts += 1

    if stats.current_streak > stats.longest_streak:
        stats.longest_streak = stats.current_streak

    # логируем тренировку
    log = WorkoutLog(user_id=user.id, duration=20)
    db.add(log)

    db.commit()

    return {
        "status": "ok",
        "streak": stats.current_streak
    }

@app.get("/status")
async def get_status(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    stats = db.query(UserStats).filter(UserStats.user_id == user.id).first()

    if not stats:
        return {
            "streak": 0,
            "total": 0
        }

    return {
        "streak": stats.current_streak,
        "total": stats.total_workouts
    }









