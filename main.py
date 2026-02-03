from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import asyncio

# === Логирование ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# --- Настройки ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 часа
DATABASE_URL = os.getenv("DATABASE_URL")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY не найден")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL не найден")

client = Groq(api_key=API_KEY)

# --- Асинхронный движок и сессия ---
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# --- FastAPI + CORS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = HTTPBearer()

# --- Модели БД (все в одном файле) ---
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    chats = relationship("Chat", back_populates="user")
    buttons = relationship("QuickButton", back_populates="user")


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


# Создаём таблицы
Base.metadata.create_all(bind=engine)

# --- Пароли ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    truncated = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.hash(truncated)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    truncated = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.verify(truncated, hashed_password)


# --- Pydantic модели ---
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


# --- JWT ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# --- Получение текущего пользователя ---
async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
        db: AsyncSession = Depends(async_session)
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        result = await db.execute(
            text("SELECT id, email FROM users WHERE email = :email"),
            {"email": email}
        )
        user = result.first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return User(id=user.id, email=user.email)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# --- Статика фронтенда ---
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


# --- Registration ---
@app.post("/register")
async def register(user: UserRegister, db: AsyncSession = Depends(async_session)):
    hashed_password = hash_password(user.password)
    result = await db.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": user.email}
    )
    if result.scalar():
        raise HTTPException(status_code=400, detail="Email already registered")

    await db.execute(
        text("INSERT INTO users (email, password) VALUES (:email, :password)"),
        {"email": user.email, "password": hashed_password}
    )
    await db.commit()
    logger.info(f"User registered: {user.email}")
    return {"status": "ok"}


# --- Login ---
@app.post("/login")
async def login(user: UserLogin, db: AsyncSession = Depends(async_session)):
    result = await db.execute(
        text("SELECT id, email, password FROM users WHERE email = :email"),
        {"email": user.email}
    )
    db_user = result.first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": db_user.email})
    logger.info(f"User logged in: {db_user.email}")
    return {"access_token": token, "token_type": "bearer", "user_id": db_user.id}


# --- Создать чат ---
@app.post("/chats")
async def create_chat(chat: ChatCreate, current_user: User = Depends(get_current_user),
                      db: AsyncSession = Depends(async_session)):
    result = await db.execute(
        text("INSERT INTO chats (user_id, title) VALUES (:user_id, :title) RETURNING id"),
        {"user_id": current_user.id, "title": chat.title}
    )
    chat_id = result.scalar()
    await db.commit()
    return {"status": "ok", "chat_id": chat_id}


# --- Получить чаты пользователя ---
@app.get("/chats")
async def get_chats(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(async_session)):
    result = await db.execute(
        text("SELECT id, title, created_at FROM chats WHERE user_id = :user_id ORDER BY created_at DESC"),
        {"user_id": current_user.id}
    )
    chats = result.fetchall()
    return {"chats": [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in chats]}


# --- УДАЛЕНИЕ ВСЕХ СООБЩЕНИЙ В ЧАТЕ ---
@app.delete("/chats/{chat_id}/messages")
async def delete_chat_messages(
        chat_id: int,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(async_session)
):
    try:
        result = await db.execute(
            text("SELECT id FROM chats WHERE id = :chat_id AND user_id = :user_id"),
            {"chat_id": chat_id, "user_id": current_user.id}
        )
        if not result.scalar():
            raise HTTPException(status_code=403, detail="Chat not found or access denied")

        await db.execute(
            text("DELETE FROM messages WHERE chat_id = :chat_id"),
            {"chat_id": chat_id}
        )
        await db.commit()

        return {"status": "messages deleted"}

    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting messages in chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# --- AI Coach (стриминг) ---
@app.post("/coach")
async def coach_response(msg: MessageIn, current_user: User = Depends(get_current_user),
                         db: AsyncSession = Depends(async_session)):
    result = await db.execute(
        text("SELECT id FROM chats WHERE id = :chat_id AND user_id = :user_id"),
        {"chat_id": msg.chat_id, "user_id": current_user.id}
    )
    if not result.scalar():
        raise HTTPException(status_code=403, detail="Chat not found or access denied")

    await db.execute(
        text("INSERT INTO messages (chat_id, user_id, sender, text) VALUES (:chat_id, :user_id, 'user', :text)"),
        {"chat_id": msg.chat_id, "user_id": current_user.id, "text": msg.text}
    )
    await db.commit()

    history = [{
        "role": "system",
        "content": """
You are an experienced senior developer and mentor with 12+ years in IT (Fullstack, focus on JS/TS, React/Node, sometimes Python/Go/DevOps).
You help developers grow: from junior to mid/senior, interviews in Europe, architecture, productivity, burnout.

Tone: direct, honest, supportive, sometimes tough-motivating (like a big brother). Tell the truth, but always constructively.
Language: natural English, no fluff. Emojis moderately 😏🔥🚀

Structure most responses:
1. Empathy + mirror (1–2 sentences)
2. Clear analysis
3. Specific recommendations (code, 2026 resources)
4. "Next steps" — 2–4 actionable points with timelines/metrics

Exceptions: small talk — easy and short.
Code always in ```js\ncode\n``` or appropriate language.
Remember context from previous messages.
        """
    }]

    result = await db.execute(
        text("SELECT sender, text FROM messages WHERE chat_id = :chat_id ORDER BY id DESC LIMIT 20"),
        {"chat_id": msg.chat_id}
    )
    last_msgs = result.fetchall()

    for m in reversed(last_msgs):
        history.append({"role": "user" if m.sender == "user" else "assistant", "content": m.text[:1500]})

    async def event_generator():
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=history,
                temperature=0.65,
                max_tokens=1200,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_reply += content
                    yield f"data: {content}\n\n"
                await asyncio.sleep(0.01)

            await db.execute(
                text("INSERT INTO messages (chat_id, user_id, sender, text) VALUES (:chat_id, :user_id, 'ai', :text)"),
                {"chat_id": msg.chat_id, "user_id": current_user.id, "text": full_reply.strip()}
            )
            await db.commit()

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Groq streaming error: {str(e)}")
            yield "data: Ошибка генерации ответа. Попробуй позже.\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- История чата ---
@app.get("/history/{chat_id}")
async def get_history(chat_id: int, current_user: User = Depends(get_current_user),
                      db: AsyncSession = Depends(async_session)):
    result = await db.execute(
        text("SELECT id FROM chats WHERE id = :chat_id AND user_id = :user_id"),
        {"chat_id": chat_id, "user_id": current_user.id}
    )
    if not result.scalar():
        raise HTTPException(status_code=403, detail="Chat not found or access denied")

    result = await db.execute(
        text("SELECT sender, text, created_at FROM messages WHERE chat_id = :chat_id ORDER BY id ASC"),
        {"chat_id": chat_id}
    )
    messages = result.fetchall()
    return {
        "messages": [{"sender": m.sender, "text": m.text, "created_at": m.created_at.isoformat()} for m in messages]}


# --- Быстрые кнопки ---
@app.get("/quick_buttons")
async def get_quick_buttons(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(async_session)):
    result = await db.execute(
        text("SELECT text FROM quick_buttons WHERE user_id = :user_id"),
        {"user_id": current_user.id}
    )
    buttons = result.fetchall()
    return {"buttons": [b.text for b in buttons]}


@app.post("/quick_buttons")
async def add_quick_button(btn: QuickButtonIn, current_user: User = Depends(get_current_user),
                           db: AsyncSession = Depends(async_session)):
    await db.execute(
        text("INSERT INTO quick_buttons (user_id, text) VALUES (:user_id, :text)"),
        {"user_id": current_user.id, "text": btn.text}
    )
    await db.commit()
    return {"status": "ok"}


@app.delete("/quick_buttons")
async def delete_quick_button(body: dict = Body(...), current_user: User = Depends(get_current_user),
                              db: AsyncSession = Depends(async_session)):
    text = body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Нет текста кнопки для удаления")
    await db.execute(
        text("DELETE FROM quick_buttons WHERE user_id = :user_id AND text = :text"),
        {"user_id": current_user.id, "text": text}
    )
    await db.commit()
    return {"status": "deleted"}


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int, current_user: User = Depends(get_current_user),
                      db: AsyncSession = Depends(async_session)):
    result = await db.execute(
        text("SELECT id FROM chats WHERE id = :chat_id AND user_id = :user_id"),
        {"chat_id": chat_id, "user_id": current_user.id}
    )
    if not result.scalar():
        raise HTTPException(status_code=403, detail="Chat not found or access denied")

    await db.execute(
        text("DELETE FROM messages WHERE chat_id = :chat_id"),
        {"chat_id": chat_id}
    )
    await db.execute(
        text("DELETE FROM chats WHERE id = :chat_id"),
        {"chat_id": chat_id}
    )
    await db.commit()
    return {"status": "ok"}