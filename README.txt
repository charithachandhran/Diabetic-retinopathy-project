DIABETIC RETINOPATHY DETECTION

BACKEND:
cd backend
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python init_db.py
uvicorn app:app --reload

FRONTEND (new terminal):
cd frontend
npm install
npm run dev

Frontend: http://localhost:5173
Backend: http://127.0.0.1:8000
API Docs: http://127.0.0.1:8000/docs
