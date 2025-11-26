# 🤖 AI Personal Coach — MVP
A minimal viable product (MVP) of a personal AI assistant that allows users to create chats, switch between them, and communicate with an AI model through a clean and responsive interface.  
The project is already deployed and can be used instantly — no installation required.

## 🌐 Live Demo

👉 **https://ai-personal-coach.onrender.com**

⚠️ Note:  
The backend is hosted on Render’s free tier.  
If the app loads slowly (10–20 seconds), the server is simply waking up.

## ✨ Features

- 🔐 User registration & login  
- 💬 Create, select, and delete chats  
- 🧠 AI assistant chat  
- 🌏 Multi-language UI (EN / RU)  
- 📱 Fully responsive design (mobile + desktop)  
- 💾 Chat history stored per account  
- 🎨 Clean modern UI with future expansion planned  

## 🛠️ Optional Local Installation

Users can test the app directly via the live link above.  
Local installation is only needed for developers.

### 1. Clone the repository
git clone https://github.com/NikoVlasov/ai-personal-coach.git  
cd ai-personal-coach

### 2. Create a virtual environment
python -m venv .venv

### 3. Activate the environment
Windows: .venv\Scripts\activate  
macOS/Linux: source .venv/bin/activate

### 4. Install dependencies
pip install -r requirements.txt

## ⚙️ Configuration

Create a `.env` file in the root folder:

API_KEY=<your_api_key>

Do **not** commit `.env` to the repository.

## ▶️ Running the App Locally

Start backend server:

uvicorn main:app --host 0.0.0.0 --port 8080

Backend will be available at:

http://localhost:8080

Then simply open `index.html` in your browser.

## 📂 Project Structure

ai-personal-coach/  
├─ main.py              # Backend API  
├─ requirements.txt     # Python dependencies  
├─ index.html           # Frontend  
├─ README.md  
├─ .gitignore  
└─ frontend/            # Assets and styles  

## 📝 Notes

- Enter-to-send is not yet implemented  
- UI/UX improvements planned: animations, redesign, additional themes  
- Safari/iOS layout may have minor issues (MVP stage)

## 📬 Feedback & Contact

Have suggestions or found a bug?  
Feel free to open an issue on GitHub — all feedback is welcome!

## ⚖️ License

This project is licensed under the MIT License.  
See the LICENSE file for details.



















