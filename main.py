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

if not GROQ_API_KEY or not DATABASE_URL:
    raise RuntimeError("Missing env variables")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

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

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    sender = Column(String)
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
# PROFILE
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
        fp = FitnessProfile(
            user_id=user.id,
            goal=profile.goal,
            level=profile.level,
            height=profile.height,
            weight=profile.weight
        )
        db.add(fp)
    db.commit()
    return {"status": "ok"}

# =========================
# COACH (ГЛАВНЫЙ)
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

        conversation = [
            {"role": "user" if m.sender == "user" else "assistant", "content": m.text}
            for m in db_messages
        ]

        profile = db.query(FitnessProfile).filter(FitnessProfile.user_id == user.id).first()

        profile_text = ""
        if profile:
            profile_text = f"User profile: goal={profile.goal}, level={profile.level}, height={profile.height}, weight={profile.weight}."

        if "start trial" in msg.text.lower():
            instruction = "Give a short 5-10 min trial workout and ask 2-3 short questions."
        else:
            instruction = "Continue coaching, track progress, give next steps."

        system_prompt = {
            "role": "system",
            "content": f"""
        You are a personal home fitness coach.

        FLOW:
        1. First → give short trial workout (5-10 min)
        2. Then → ask 2-3 questions (difficulty, fatigue, feeling)
        3. Then → build personalized plan
        4. Then → guide user daily

        Rules:
        - Be proactive
        - Guide step-by-step
        - Do not overwhelm
        - Keep answers structured and clear

        {profile_text}
        """
        }

        full_messages = [system_prompt] + conversation

        completion = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages,
                temperature=0.6,
                max_tokens=700
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
# CHECK-IN
# =========================
@app.post("/checkin")
async def daily_checkin(req: CheckinRequest,
                        user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):

    db.add(DailyCheckin(user_id=user.id, energy=req.energy, soreness=req.soreness, mood=req.mood))
    db.commit()

    db.add(Message(user_id=user.id, sender="user",
                   text=f"Check-in: energy={req.energy}, soreness={req.soreness}, mood={req.mood}"))
    db.commit()

    return PlainTextResponse("Check-in saved")

@app.get("/messages")
async def get_messages(user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    messages = db.query(Message)\
        .filter(Message.user_id == user.id)\
        .order_by(Message.created_at)\
        .all()

    return [
        {"sender": m.sender, "text": m.text}
        for m in messages
    ]