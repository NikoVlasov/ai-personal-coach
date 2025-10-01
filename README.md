🤖 AI Personal Coach - MVP



📖 Overview:


AI Personal Coach is a minimal viable product (MVP) of a personal AI assistant that helps users chat, track conversations, and use predefined quick buttons for common tasks.

✨ Features:
- User registration and login
- Select, create, and delete chats
- Chat with AI assistant
- Quick buttons for frequently used messages
- Multi-language support (RU / EN)
- Minimalist responsive UI

🛠️ Installation:
 # Clone the repository
git clone https://github.com/NikoVlasov/ai-personal-coach.git
cd ai-personal-coach

# Create virtual environment
python -m venv .venv

# Activate it
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


⚙️ Configuration:
1. Create a `.env` file in the root folder.
2. Add required environment variables (e.g., API keys, configuration):
   API_KEY=<your_api_key>
   OTHER_CONFIG=<other_config>
3. **Do not commit `.env` to the repository.**

🖥️ Running the Application:
# Start backend server
uvicorn main:app --host 0.0.0.0 --port 8080

# Open frontend
# Simply open index.html in browser or deploy on GitHub Pages / Netlify

🌐 Usage:
1. Open the frontend in a browser.
2. Register a new account or log in.
3. Create or select a chat.
4. Send messages to the AI assistant.
5. Use quick buttons for common predefined messages.
6. Delete chats if needed.

📝 Notes:
- Sending messages by pressing Enter is currently **not implemented** (only via button).
- Chat renaming is **not implemented yet**.
- UI is minimal, future improvements planned: dark mode, animations, responsive design.

📂 Project Structure:
ai-personal-coach/
├─ main.py           # Backend API
├─ requirements.txt  # Python dependencies
├─ index.html        # Frontend MVP
├─ .gitignore
├─ README.md
└─ frontend/         # Optional folder for assets

📬 Contact / Feedback:
For any questions, suggestions, or feedback, feel free to open an issue on GitHub or contact the developer directly.

⚖️ License:
This project is licensed under the MIT License. See the LICENSE file for details.


