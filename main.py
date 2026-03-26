import asyncio
import os
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles

from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

from dotenv import load_dotenv
from groq import Groq

# =========================
# CONFIG
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")

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
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    password = Column(String)
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

pwd_context = CryptContext(schemes=["bcrypt"])


def hash_password(p):
    return pwd_context.hash(p)


def verify_password(p, h):
    return pwd_context.verify(p, h)


def create_token(data: dict):
    data["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user = db.query(User).filter(User.email == payload["sub"]).first()
        if not user:
            raise HTTPException(401)
        return user
    except JWTError:
        raise HTTPException(401)

# =========================
# SCHEMAS
# =========================

class UserIn(BaseModel):
    email: str
    password: str


class ChatCreate(BaseModel):
    title: str


class MessageIn(BaseModel):
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
async def register(data: UserIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "exists")

    user = User(email=data.email, password=hash_password(data.password))
    db.add(user)
    db.commit()
    return {"ok": True}


@app.post("/login")
async def login(data: UserIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()

    if not user or not verify_password(data.password, user.password):
        raise HTTPException(401)

    token = create_token({"sub": user.email})
    return {"access_token": token}

# =========================
# CHATS
# =========================

@app.post("/chats")
async def create_chat(data: ChatCreate, user=Depends(get_user), db: Session = Depends(get_db)):
    chat = Chat(user_id=user.id, title=data.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"chat_id": chat.id}


@app.get("/chats")
async def get_chats(user=Depends(get_user), db: Session = Depends(get_db)):
    chats = db.query(Chat).filter(Chat.user_id == user.id).all()
    return {"chats": [{"id": c.id, "title": c.title} for c in chats]}


@app.get("/history/{chat_id}")
async def history(chat_id: int, user=Depends(get_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(403)

    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.id).all()

    return {
        "messages": [{"sender": m.sender, "text": m.text} for m in messages]
    }

# =========================
# AI
# =========================

@app.post("/coach")
async def coach(msg: MessageIn, user=Depends(get_user), db: Session = Depends(get_db)):

    chat = db.query(Chat).filter(Chat.id == msg.chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(404)

    # save user msg
    db.add(Message(chat_id=chat.id, sender="user", text=msg.text))
    db.commit()

    # get history
    db_messages = db.query(Message)\
        .filter(Message.chat_id == chat.id)\
        .order_by(Message.id.desc())\
        .limit(20)\
        .all()

    db_messages.reverse()

    messages = [
        {"role": "system", "content": "You are a helpful AI coach."}
    ]

    for m in db_messages:
        role = "user" if m.sender == "user" else "assistant"
        messages.append({"role": role, "content": m.text})

    completion = await asyncio.to_thread(
        lambda: groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
    )

    ai_text = completion.choices[0].message.content

    db.add(Message(chat_id=chat.id, sender="ai", text=ai_text))
    db.commit()

    return PlainTextResponse(ai_text)