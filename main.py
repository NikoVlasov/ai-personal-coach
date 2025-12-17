from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# === Логирование ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# --- Настройки ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
DATABASE_URL = os.getenv("DATABASE_URL")

client = Groq(api_key=API_KEY)

# --- FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
oauth2_scheme = HTTPBearer()

# --- БД ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# =======================
#        MODELS
# =======================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    chats = relationship("Chat", back_populates="user")
    buttons = relationship("QuickButton", back_populates="user")
    memories = relationship("UserMemory", back_populates="user")


class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    sender = Column(String)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", back_populates="messages")


class QuickButton(Base):
    __tablename__ = "quick_buttons"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="buttons")


# 🔥 НОВОЕ: User Memory
class UserMemory(Base):
    __tablename__ = "user_memory"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    key = Column(String, index=True)
    value = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="memories")


Base.metadata.create_all(bind=engine)

# =======================
#     AUTH HELPERS
# =======================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")[:72]
    return pwd_context.hash(password_bytes.decode("utf-8", errors="ignore"))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode("utf-8")[:72]
    return pwd_context.verify(password_bytes.decode("utf-8", errors="ignore"), hashed_password)

# =======================
#     SCHEMAS
# =======================

class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ChatCreate(BaseModel):
    title: str

class MessageIn(BaseModel):
    chat_id: int
    text: str

class QuickButtonIn(BaseModel):
    text: str

# 🔥 Memory schema
class MemoryIn(BaseModel):
    key: str
    value: str

# =======================
#        JWT
# =======================

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
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
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# =======================
#      FRONTEND
# =======================

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

# =======================
#        AUTH
# =======================

@app.post("/register")
async def register(user: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    db.add(User(email=user.email, password=hash_password(user.password)))
    db.commit()
    return {"status": "ok"}

@app.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer", "user_id": db_user.id}

# =======================
#        CHATS
# =======================

@app.post("/chats")
async def create_chat(chat: ChatCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_chat = Chat(user_id=current_user.id, title=chat.title)
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return {"chat_id": new_chat.id}

@app.get("/chats")
async def get_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = db.query(Chat).filter(Chat.user_id == current_user.id).order_by(Chat.created_at.desc()).all()
    return {"chats": [{"id": c.id, "title": c.title} for c in chats]}

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(Message).filter(Message.chat_id == chat_id).delete()
    db.query(Chat).filter(Chat.id == chat_id).delete()
    db.commit()
    return {"status": "ok"}

# =======================
#     USER MEMORY 🔥
# =======================

@app.get("/memory")
async def get_memory(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    memories = db.query(UserMemory).filter(UserMemory.user_id == current_user.id).all()
    return {"memory": [{"key": m.key, "value": m.value} for m in memories]}

@app.post("/memory")
async def add_memory(
    memory: MemoryIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db.add(UserMemory(
        user_id=current_user.id,
        key=memory.key,
        value=memory.value
    ))
    db.commit()
    return {"status": "saved"}

# =======================
#        AI COACH
# =======================

@app.post("/coach")
async def coach_response(msg: MessageIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.add(Message(chat_id=msg.chat_id, user_id=current_user.id, sender="user", text=msg.text))
    db.commit()

    memories = db.query(UserMemory).filter(UserMemory.user_id == current_user.id).all()
    memory_block = ""
    if memories:
        memory_block = "User profile:\n"
        for m in memories:
            memory_block += f"- {m.key}: {m.value}\n"

    system_prompt = f"""
You are a personal AI mentor.
Use the following information about the user if relevant.
{memory_block}
"""

    history = [{"role": "system", "content": system_prompt}]
    last_msgs = db.query(Message).filter(Message.chat_id == msg.chat_id).order_by(Message.id.desc()).limit(10).all()
    for m in reversed(last_msgs):
        history.append({"role": "user" if m.sender == "user" else "assistant", "content": m.text[:1000]})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=history,
        temperature=0.7,
        max_tokens=500
    )

    ai_text = response.choices[0].message.content.strip()
    db.add(Message(chat_id=msg.chat_id, user_id=current_user.id, sender="ai", text=ai_text))
    db.commit()

    suggest_memory = "I am" in msg.text or "I'm" in msg.text

    return {
        "reply": ai_text,
        "suggest_memory": suggest_memory
    }

# =======================
#     CHAT HISTORY
# =======================

@app.get("/history/{chat_id}")
async def get_history(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.id.asc()).all()
    return {"messages": [{"sender": m.sender, "text": m.text} for m in messages]}

# =======================
#    QUICK BUTTONS
# =======================

@app.get("/quick_buttons")
async def get_quick_buttons(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    buttons = db.query(QuickButton).filter(QuickButton.user_id == current_user.id).all()
    return {"buttons": [b.text for b in buttons]}

@app.post("/quick_buttons")
async def add_quick_button(btn: QuickButtonIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.add(QuickButton(user_id=current_user.id, text=btn.text))
    db.commit()
    return {"status": "ok"}

@app.delete("/quick_buttons")
async def delete_quick_button(
    body: dict = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    text = body.get("text")
    db.query(QuickButton).filter(
        QuickButton.user_id == current_user.id,
        QuickButton.text == text
    ).delete()
    db.commit()
    return {"status": "deleted"}
