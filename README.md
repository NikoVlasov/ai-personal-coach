# 🤖 AI Personal Coach - MVP

## 📖 Overview

AI Personal Coach —  (MVP) 


✨ Features

User registration and login

Select, create, and delete chats

Chat with AI assistant

## ✨ Features

- User registration and login  
- Select, create, and delete chats  
- Chat with AI assistant  
- Multi-language support (RU / EN)  
- Minimalist responsive UI with animations and tooltips  
- Floating icons and interactive buttons


## 🌐 Testing the Application

### For Users / Testers

You can test the application **directly via Render** without installing anything locally:  
[Open AI Personal Coach on Render](YOUR_RENDER_LINK_HERE)  

> ⚠️ Note: Free Render servers may go to sleep when inactive. The first request may take a few seconds to wake up the server.

### For Developers / Contributors

If you want to run the application locally or contribute to development:

#### Clone the repository
git clone https://github.com/NikoVlasov/ai-personal-coach.git
cd ai-personal-coach

Create a virtual environment:
python -m venv .venv

Activate it

Windows:

.venv\Scripts\activate

Linux / Mac:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Configuration

Create a .env file in the root folder and add required environment variables:

API_KEY=<your_api_key>
OTHER_CONFIG=<other_config>

Do not commit .env to the repository.

Running Locally:

uvicorn main:app --host 0.0.0.0 --port 8080

Open index.html in a browser or deploy locally using GitHub Pages / Netlify.


```markdown
## 🖥️ Usage

1. Open the frontend in a browser (either via Render or locally).  
2. Register a new account or log in.  
3. Create or select a chat.  
4. Send messages to the AI assistant.  
5. Delete chats if needed.  

> ⚠️ Notes:  
> - Sending messages by pressing Enter is currently not implemented (only via button).  
> - Chat renaming is not implemented yet.  
> - UI is minimal; future improvements include dark mode, animations, and enhanced responsive design.

## 📂 Project Structure

```text
ai-personal-coach/
├─ main.py             # Backend API
├─ requirements.txt    # Python dependencies
├─ index.html          # Frontend MVP
├─ .gitignore
├─ README.md
└─ frontend/           # Optional folder for assets


## 📬 Contact / Feedback

For any questions, suggestions, or feedback, feel free to open an issue on GitHub or contact the developer directly.

## ⚖️ License

This project is licensed under the MIT License. See the LICENSE file for details.

