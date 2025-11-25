🤖 AI Personal Coach - MVP
📖 Overview

AI Personal Coach — минимальный рабочий вариант (MVP) персонального AI ассистента, который помогает пользователю общаться с ИИ, отслеживать чаты и управлять ими.

✨ Features

User registration and login

Select, create, and delete chats

Chat with AI assistant

Multi-language support (RU / EN)

Minimalist responsive UI with subtle animations

🛠️ Installation

1. Clone the repository:

git clone https://github.com/NikoVlasov/ai-personal-coach.git
cd ai-personal-coach

2. Create virtual environment:

python -m venv .venv

3. Activate it:

Windows: .venv\Scripts\activate

Linux / Mac: source .venv/bin/activate

4. Install dependencies:

pip install -r requirements.txt

⚙️ Configuration

1. Create a .env file in the root folder.

2. Add required environment variables (e.g., API keys, configuration):

API_KEY=<your_api_key>

OTHER_CONFIG=<other_config>

3. Do not commit .env to the repository.

🖥️ Running the Application

There are two options:

1. Online via Render

Open the deployed Render URL in your browser.

Note: Free Render servers may “sleep” after inactivity. If so, wait a few seconds for the app to wake up.

2. Locally

Start backend server:

uvicorn main:app --host 0.0.0.0 --port 8080

Open index.html in browser or deploy frontend locally.

Messages and interactions respond immediately, no waiting for server wake-up.

🌐 Usage

Open the frontend in a browser.

Register a new account or log in.

Create or select a chat.

Send messages to the AI assistant.

Delete chats if needed.

📝 Notes

Sending messages by pressing Enter is currently not implemented (only via button).

Chat renaming is not implemented yet.

UI is minimal; future improvements planned: dark mode, animations, responsive tweaks.

 Project Structure

ai-personal-coach/

main.py       # Backend API

requirements.txt     # Python dependencies

.gitignore

README.md

📂Frontend

index.html      # Frontend MVP

📬 Contact / Feedback

For any questions, suggestions, or feedback, feel free to open an issue on GitHub or contact the developer directly.

⚖️ License

This project is licensed under the MIT License. See the LICENSE file for details.

