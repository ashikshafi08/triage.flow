<p align="center">
  <img src="triage_flow_logo.png" alt="triage.flow logo" width="180" height="180"/>
</p>

# triage.flow

<p align="center">
  <b>AI-Powered Interactive GitHub Issue Analysis with Modern Chat Interface</b><br>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python"></a>
  <a href="https://reactjs.org"><img src="https://img.shields.io/badge/React-18%2B-blue.svg" alt="React"></a>
</p>

---

## 🚀 What is triage.flow?

A modern, web-based AI assistant that helps you **understand, analyze, and triage GitHub issues** through an interactive chat interface. Get deep, code-aware insights with beautiful presentation and seamless user experience.

### ✨ Key Highlights

- **🎯 Interactive Chat Interface** - ChatGPT-style conversation with GitHub issues
- **🌲 Codebase Explorer** - Collapsible sidebar with full repository tree navigation  
- **📋 Enhanced Markdown** - Context-aware emojis, file linking, and professional formatting
- **📎 Smart File Mentions** - Use `@filename.py` to reference specific files in chat
- **⚡ Real-time Streaming** - Live AI responses with typing indicators
- **🎨 Modern UI/UX** - Professional, investor-pitch ready interface
- **🔍 Smart Analysis** - RAG-powered context extraction from 20+ programming languages

---

## 🎨 Interface Features

| Feature | Description |
|---------|-------------|
| **Context-Aware Headers** | Headers automatically get relevant emojis (🐛 for bugs, 🔧 for fixes, etc.) |
| **Smart Bullet Points** | List items get contextual icons (❌ for errors, ✅ for success, 💡 for tips) |
| **File Cross-linking** | Automatic detection and highlighting of referenced files |
| **Numbered Step Badges** | Visual badges for step-by-step instructions |
| **Professional Code Blocks** | VS Code theme with copy buttons and language detection |
| **Enhanced Blockquotes** | Important notes with lightbulb indicators |
| **Responsive Design** | Works beautifully on desktop and mobile |

---

## 🛠 Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: Python FastAPI, FAISS Vector Store
- **AI Models**: OpenAI, OpenRouter, Claude, Mistral support
- **Code Analysis**: Tree-sitter parsers for 20+ languages
- **Real-time**: Server-Sent Events for streaming responses

---

## 🚀 Quick Start

### 1. Backend Setup

```bash
git clone https://github.com/yourusername/triage.flow.git
cd triage.flow

# Using uv (recommended)
uv pip sync requirements.txt

# Or using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the backend
python -m uvicorn src.main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd issue-flow-ai-prompt
npm install
npm run dev
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key  
GITHUB_TOKEN=your_github_token
LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
```

### 4. Start Analyzing Issues! 

1. Visit `http://localhost:3000`
2. Paste a GitHub issue URL
3. Start chatting with the AI about the issue
4. Use the sidebar to explore the codebase
5. Mention files with `@filename.py` for specific context

---

## 💬 How to Use the Chat Interface

### Basic Commands
- **Ask questions**: "What's causing this bug?"
- **Request analysis**: "Explain the root cause"
- **Get recommendations**: "How should we fix this?"
- **File references**: "Look at @config.py for the settings"

### Pro Tips
- Use the **sidebar toggle** (📁 icon) to browse the full codebase
- **@ mention files** to include them in your conversation context
- Watch for **emoji headers** that categorize information types
- **Copy code blocks** with the built-in copy button
- **Numbered steps** appear as visual badges for easy following

---

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions` | POST | Create new analysis session |
| `/sessions/{id}/messages` | POST | Send message to AI |
| `/api/files` | GET | Get repository file list |
| `/api/tree` | GET | Get repository tree structure |

---

## 🎯 Use Cases

### For Maintainers
- **Quickly triage** incoming issues with AI insights
- **Understand complex bugs** across large codebases  
- **Generate responses** to contributor questions
- **Prioritize** issues based on impact analysis

### For Contributors  
- **Get up to speed** on unfamiliar codebases
- **Understand** the context behind issues
- **Learn** best practices from AI analysis
- **Navigate** large repositories efficiently

### For Teams
- **Collaborative debugging** with shared AI sessions
- **Knowledge transfer** through AI-explained code
- **Consistent** issue analysis across team members
- **Documentation** generation from AI insights

---

## 🌟 Advanced Features

### Smart Context Extraction
- **Repository cloning** for complete code access
- **FAISS vector indexing** for semantic code search  
- **Multi-language support** via tree-sitter parsers
- **Issue + comments** extraction for full context

### Professional UI Elements
- **ChatGPT-style** message bubbles and avatars
- **Gradient headers** with status indicators
- **Hover effects** and smooth transitions
- **Loading animations** and streaming indicators
- **Responsive design** for all screen sizes

### File Navigation
- **Collapsible tree** structure in sidebar
- **File type icons** and folder indicators
- **Search functionality** within file picker
- **Quick file reference** with @ mentions

---

## 🔮 Roadmap

- [ ] **Real-time collaboration** - Multiple users in same session
- [ ] **Issue templates** - Pre-built analysis prompts  
- [ ] **Export functionality** - Save conversations as markdown
- [ ] **Plugin system** - Custom analysis tools
- [ ] **GitHub integration** - Direct issue commenting
- [ ] **Multi-repo support** - Analyze across repositories

---

## 🤝 Contributing

We love contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Backend development
cd triage.flow
python -m uvicorn src.main:app --reload --port 8000

# Frontend development  
cd issue-flow-ai-prompt
npm run dev

# Run both with hot reload for full-stack development
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [React](https://reactjs.org) and [FastAPI](https://fastapi.tiangolo.com)
- UI components from [shadcn/ui](https://ui.shadcn.com)
- Icons by [Lucide](https://lucide.dev)
- Syntax highlighting by [react-syntax-highlighter](https://github.com/react-syntax-highlighter/react-syntax-highlighter)
