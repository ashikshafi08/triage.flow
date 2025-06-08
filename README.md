<p align="center">
  <img src="triage_flow_logo.png" alt="triage.flow logo" width="120"/>
</p>

# triage.flow

**AI-powered GitHub repo analysis and code review, in your browser.**  
*Instantly understand, explore, and triage any codebase with an agentic chat interface.*

---

## 🚀 Why triage.flow?

- **AI Agentic Chat:** Ask anything about your repo—architecture, bugs, PRs, code patterns—and get deep, actionable answers.
- **Smart File Explorer:** Instantly preview, search, and cross-link files and folders.
- **PR & Diff Insights:** Visualize pull request changes and code diffs inline.
- **Live Reasoning:** Watch the AI think, plan, and act in real time.
- **Modern UI:** Beautiful, responsive, and fast—works on any repo, any size.

---

## ⚡ Quick Start

```bash
# 1. Clone and install backend
git clone https://github.com/yourusername/triage.flow.git
cd triage.flow
uv pip sync requirements.txt  # or: pip install -r requirements.txt

# 2. Start backend
python -m uvicorn src.main:app --reload --port 8000

# 3. Launch frontend
cd issue-flow-ai-prompt
npm install
npm run dev
```

- Open [http://localhost:3000](http://localhost:3000)
- Paste a GitHub repo URL and start chatting!

---

## ✨ Features

- **Autonomous AI agent** for codebase Q&A and reasoning
- **Context-aware file/PR previews** and inline diffs
- **@-mention autocomplete** for files/folders
- **Integrated file viewer** with syntax highlighting
- **Streaming responses** and live agent steps

---

## 🛠 Tech Stack

- **Frontend:** React 18, TypeScript, Tailwind CSS
- **Backend:** FastAPI, Python 3.8+, FAISS, LlamaIndex
- **AI:** OpenAI, OpenRouter (Claude, Mistral, Llama)

---

## 📄 License

MIT

---

## 🤝 Contributing

PRs welcome! Fork, branch, and open a pull request.

---

<p align="center">
  <i>Built with ❤️ for developers, code reviewers, and teams.</i>
</p>
