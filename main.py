from fastapi import FastAPI, HTTPException, Body, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import sqlite3
from passlib.context import CryptContext
from jose import JWTError, jwt
import logging

# === Логирование ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# --- Настройки ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

client = Groq(api_key=API_KEY)

# --- FastAPI + CORS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- SQLite база ---
conn = sqlite3.connect("coach.db", check_same_thread=False)

# --- Таблицы ---
with conn:
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT,
        created_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        created_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        user_id INTEGER,
        sender TEXT,
        text TEXT,
        created_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS quick_buttons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        text TEXT,
        created_at TEXT
    )""")

# --- Пароли ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Pydantic модели ---
class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ChatCreate(BaseModel):
    title: str

class Message(BaseModel):
    chat_id: int
    text: str

class QuickButton(BaseModel):
    text: str

# --- JWT ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный токен",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        cur = conn.cursor()
        cur.execute("SELECT id, email FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        cur.close()
        if not row:
            raise credentials_exception
        return {"id": row[0], "email": row[1]}
    except JWTError:
        raise credentials_exception

# --- Регистрация ---
@app.post("/register")
async def register(user: UserRegister):
    hashed_password = pwd_context.hash(user.password)
    try:
        with conn:
            conn.execute(
                "INSERT INTO users (email,password,created_at) VALUES (?,?,?)",
                (user.email, hashed_password, datetime.utcnow())
            )
        logger.info(f"Пользователь зарегистрирован: {user.email}")
        return {"status": "ok"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email уже зарегистрирован")

# --- Логин ---
@app.post("/login")
async def login(user: UserLogin):
    cur = conn.cursor()
    cur.execute("SELECT id, password, email FROM users WHERE email=?", (user.email,))
    row = cur.fetchone()
    cur.close()
    if row and pwd_context.verify(user.password, row[1]):
        token = create_access_token({"sub": row[2]})
        logger.info(f"Пользователь вошел: {row[2]}")
        return {"access_token": token, "token_type": "bearer", "user_id": row[0]}
    else:
        logger.warning(f"Неудачный вход: {user.email}")
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

# --- Создать чат ---
@app.post("/chats")
async def create_chat(chat: ChatCreate, current_user: dict = Depends(get_current_user)):
    logger.info(f"Создание чата: {current_user['email']} title={chat.title}")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chats (user_id, title, created_at) VALUES (?,?,?)",
        (current_user["id"], chat.title, datetime.utcnow())
    )
    conn.commit()
    chat_id = cur.lastrowid
    cur.close()
    return {"status": "ok", "chat_id": chat_id}

# --- Получить чаты пользователя ---
@app.get("/chats")
async def get_chats(current_user: dict = Depends(get_current_user)):
    logger.info(f"Получение списка чатов: {current_user['email']}")
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, created_at FROM chats WHERE user_id=? ORDER BY created_at DESC",
        (current_user["id"],)
    )
    rows = cur.fetchall()
    logger.info(f"Найденные чаты: {rows}")
    chats = [{"id": row[0], "title": row[1], "created_at": row[2]} for row in rows]
    cur.close()
    return {"chats": chats}

# --- AI Coach ---
@app.post("/coach")
async def coach_response(msg: Message, current_user: dict = Depends(get_current_user)):
    logger.info(f"Сообщение от {current_user['email']}: {msg.text}")
    with conn:
        conn.execute(
            "INSERT INTO messages (chat_id,user_id,sender,text,created_at) VALUES (?,?,?,?,?)",
            (msg.chat_id, current_user["id"], "user", msg.text, datetime.utcnow())
        )
    cur = conn.cursor()
    cur.execute(
        "SELECT sender, text FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT 10",
        (msg.chat_id,)
    )
    history_rows = cur.fetchall()[::-1]
    cur.close()
    history = [{"role": "system", "content": "Ты — персональный коуч. Поддерживай и мотивируй."}]
    for sender, text in history_rows:
        role = "user" if sender == "user" else "assistant"
        history.append({"role": role, "content": text[:1000]})
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=history,
        temperature=0.7,
        max_tokens=500
    )
    ai_text = response.choices[0].message.content.strip()
    with conn:
        conn.execute(
            "INSERT INTO messages (chat_id,user_id,sender,text,created_at) VALUES (?,?,?,?,?)",
            (msg.chat_id, current_user["id"], "ai", ai_text, datetime.utcnow())
        )
    logger.info(f"Ответ AI: {ai_text[:50]}...")
    return {"reply": ai_text}

# --- История чата ---
@app.get("/history/{chat_id}")
async def get_history(chat_id: int, current_user: dict = Depends(get_current_user)):
    cur = conn.cursor()
    cur.execute("SELECT sender, text, created_at FROM messages WHERE chat_id=? ORDER BY id ASC", (chat_id,))
    messages = [{"sender": row[0], "text": row[1], "created_at": row[2]} for row in cur.fetchall()]
    cur.close()
    logger.info(f"История чата {chat_id}: {len(messages)} сообщений")
    return {"messages": messages}

# --- Быстрые кнопки ---
@app.get("/quick_buttons")
async def get_quick_buttons(current_user: dict = Depends(get_current_user)):
    cur = conn.cursor()
    cur.execute("SELECT text FROM quick_buttons WHERE user_id=?", (current_user["id"],))
    buttons = [row[0] for row in cur.fetchall()]
    cur.close()
    return {"buttons": buttons}

@app.post("/quick_buttons")
async def add_quick_button(btn: QuickButton, current_user: dict = Depends(get_current_user)):
    with conn:
        conn.execute(
            "INSERT INTO quick_buttons (user_id,text,created_at) VALUES (?,?,?)",
            (current_user["id"], btn.text, datetime.utcnow())
        )
    logger.info(f"Добавлена кнопка: {btn.text} для {current_user['email']}")
    return {"status": "ok"}

@app.delete("/quick_buttons")
async def delete_quick_button(body: dict = Body(...), current_user: dict = Depends(get_current_user)):
    text = body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Нет текста кнопки для удаления")
    with conn:
        conn.execute("DELETE FROM quick_buttons WHERE user_id=? AND text=?", (current_user["id"], text))
    logger.info(f"Удалена кнопка: {text} для {current_user['email']}")
    return {"status": "deleted"}

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int, current_user: dict = Depends(get_current_user)):
    with conn:
        conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    logger.info(f"Удален чат {chat_id} для {current_user['email']}")
    return {"status": "ok"}
