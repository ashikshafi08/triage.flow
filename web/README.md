# triage.flow Web Interface

This directory contains the web interface for triage.flow, built with:
- Backend: FastAPI (Python)
- Frontend: React

## Setup

### Backend
1. Navigate to backend directory:
   ```bash
   cd web/backend
   ```
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set environment variables (create `.env` file):
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   GITHUB_TOKEN=your_github_token
   OPENAI_API_KEY=your_openai_api_key  # Optional if using OpenRouter
   ```
5. Run backend server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend
1. Navigate to frontend directory:
   ```bash
   cd web/frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run development server:
   ```bash
   npm run dev
   ```

## Usage
1. Access frontend at http://localhost:5173
2. Enter a GitHub issue URL
3. Select a prompt type
4. View analysis results

## Project Structure
```
web/
├── backend/
│   ├── main.py           # FastAPI server
│   ├── processing.py     # Issue processing logic
│   └── requirements.txt  # Python dependencies
└── frontend/
    ├── src/
    │   ├── components/   # React components
    │   ├── App.jsx       # Main application
    │   └── App.css       # Global styles
    └── package.json      # Frontend dependencies
