<p align="center">
  <img src="triage_flow_logo.png" alt="triage.flow logo" width="180" height="180"/>
</p>

# triage.flow

<p align="center">
  <b>AI-Powered Interactive GitHub Repository Analysis with Advanced Chat Interface</b><br>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python"></a>
  <a href="https://reactjs.org"><img src="https://img.shields.io/badge/React-18%2B-blue.svg" alt="React"></a>
</p>

---

## 🚀 What is triage.flow?

A modern, web-based AI assistant that helps you **understand, analyze, and explore entire GitHub repositories** through an interactive chat interface. Get deep, code-aware insights with beautiful presentation, smart context retrieval, and cost-optimized AI interactions.

### ✨ Key Highlights

- **🎯 Interactive Repository Chat** - ChatGPT-style conversation with full repository context
- **🌲 Smart Codebase Explorer** - High-performance file picker with search and tree navigation  
- **📋 Enhanced Markdown** - Context-aware emojis, file linking, and professional formatting
- **📎 Smart File & Folder Mentions** - Use `@filename.py` or `@folder/path` to reference specific files/folders
- **⚡ Real-time Streaming** - Live AI responses with typing indicators and cost optimization
- **🎨 Modern UI/UX** - Professional, responsive interface with session management
- **🔍 Advanced RAG System** - Smart context sizing with 10-25 sources from indexed repositories
- **💰 Cost-Optimized AI** - OpenRouter prompt caching for 25-90% cost savings

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
| **Session Management** | Multiple concurrent repository sessions with history |
| **High-Performance File Picker** | Optimized for repositories with 3000+ files |

---

## 🛠 Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: Python FastAPI, FAISS Vector Store, Advanced RAG
- **AI Models**: OpenAI, OpenRouter (Claude, Mistral, Llama) with prompt caching
- **Code Analysis**: Tree-sitter parsers for 20+ languages with smart fallbacks
- **Real-time**: Server-Sent Events for streaming responses
- **Performance**: Multi-level caching, smart context sizing, async processing

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

# Install tree-sitter language parsers for optimal TypeScript/JavaScript parsing
pip install tree-sitter-javascript tree-sitter-typescript

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
# AI Provider Configuration
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key  # Recommended for cost savings
GITHUB_TOKEN=your_github_token

# Model Configuration
LLM_PROVIDER=openrouter  # or "openai"
DEFAULT_MODEL=anthropic/claude-3.5-sonnet  # or gpt-4o-mini

# Performance & Cost Optimization
ENABLE_PROMPT_CACHING=true  # 25-90% AI cost savings
ENABLE_SMART_SIZING=true    # Dynamic context sizing
MIN_RAG_SOURCES=10          # Minimum context sources
DEFAULT_RAG_SOURCES=15      # Default context sources
MAX_RAG_SOURCES=25          # Maximum context sources

# Feature Flags
ENABLE_RAG_CACHING=true
ENABLE_RESPONSE_CACHING=true
ENABLE_ASYNC_RAG=true
```

### 4. Start Exploring Repositories! 

1. Visit `http://localhost:3000`
2. Paste a GitHub repository URL (public repos supported)
3. Wait for repository cloning and indexing (progress shown)
4. Start chatting with the AI about the codebase
5. Use the sidebar to explore files and folders
6. Mention files with `@filename.py` or folders with `@folder/subfolder`

---

## 💬 How to Use the Chat Interface

### Repository Analysis Commands
- **Overview**: "What is this repository about?"
- **Architecture**: "Explain the project structure"
- **Code exploration**: "Show me the main components"
- **Debugging**: "Help me understand this error in @file.py"
- **Implementation**: "How does the authentication work?"

### Smart Context Features
- **File references**: "Look at @config.py for the settings"
- **Folder exploration**: "What's in @folder/src/components?"
- **Multi-file analysis**: "Compare @file1.js and @file2.js"
- **Complex queries**: "Find all database-related files and explain the schema"

### Pro Tips
- Use **@** for file autocomplete (optimized for large repos)
- **Folder mentions** automatically include all files in context
- The AI **remembers conversation history** across session reloads
- **Query complexity** automatically adjusts context size (10-25 sources)
- **Prompt caching** reduces costs for repetitive repository questions

---

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/assistant/sessions` | POST | Create new repository session |
| `/assistant/sessions` | GET | List all sessions |
| `/assistant/sessions/{id}/messages` | GET | Get conversation history |
| `/sessions/{id}/messages` | POST | Send message to AI (streaming) |
| `/api/files` | GET | Get repository file list |
| `/api/tree` | GET | Get repository tree structure |
| `/api/file-content` | GET | Get specific file content |
| `/cache-stats` | GET | Monitor performance metrics |

---

## 🎯 Use Cases

### For Developers
- **Understand new codebases** quickly with AI-guided exploration
- **Debug complex issues** across multiple files and components
- **Learn best practices** from AI analysis of well-structured projects
- **Navigate large repositories** efficiently with smart search

### For Code Reviewers  
- **Get context** on unfamiliar parts of the codebase
- **Understand impact** of proposed changes
- **Generate insights** for review comments
- **Explore dependencies** and related code

### For Technical Writers
- **Generate documentation** from AI understanding of code
- **Create tutorials** based on actual implementation
- **Explain complex systems** with AI assistance
- **Understand APIs** and integration patterns

### For Teams
- **Onboard new developers** with AI-guided codebase tours
- **Share knowledge** through AI-explained code patterns
- **Consistent analysis** across team members
- **Collaborative exploration** of system architecture

---

## 🌟 Advanced Features

### Smart Context & RAG System
- **Repository cloning** with full history and branch support
- **FAISS vector indexing** for semantic code search with 3000+ documents
- **Smart context sizing** - automatically adjusts 10-25 sources based on query complexity
- **Multi-language support** via tree-sitter parsers with intelligent fallbacks
- **Query complexity analysis** - detects repository overview vs. specific file questions

### Cost Optimization & Performance
- **OpenRouter prompt caching** - 25-90% cost savings on repetitive queries
- **Multi-level caching** - RAG context, responses, and folder structure
- **Async processing** - background repository indexing and context retrieval
- **Smart file picker** - optimized for repositories with 3000+ files (30 folders + 20 files initially)
- **Memory management** - intelligent conversation context compression

### Professional UI/UX
- **Session persistence** - conversations survive page reloads
- **Multiple repository sessions** - work with different repos simultaneously
- **Real-time status** - repository cloning and indexing progress
- **Advanced file picker** - search, folders, and performance optimization
- **Responsive chat interface** - mobile-friendly with typing indicators

### Enterprise-Ready Features
- **Configurable AI providers** - OpenAI, OpenRouter, multiple models
- **Performance monitoring** - cache statistics and usage metrics
- **Error handling** - graceful fallbacks and detailed logging
- **Security** - file access restrictions and path validation

---

## 📊 Performance & Cost Metrics

### Prompt Caching Savings (OpenRouter)
- **Anthropic Claude**: 90% savings on cached content (0.1x cost)
- **OpenAI GPT models**: 25-50% savings on cached content (0.5x-0.75x cost)
- **Automatic caching**: Triggers for repository context >1000 tokens

### RAG System Performance
- **Index size**: 1000-5000+ documents per repository
- **Context sources**: 10-25 dynamically selected based on query
- **Response time**: <2s for most queries with caching
- **Memory usage**: Optimized for large repositories

### File Picker Optimization
- **Large repos**: Handles 3000+ files smoothly
- **Initial load**: 30 folders + 20 files for fast startup
- **Search results**: Capped at 100 items for performance
- **Tree navigation**: Lazy loading for deep directory structures

---

## 🔮 Roadmap

- [ ] **Private repository support** - GitHub App integration for private repos
- [ ] **Multi-branch analysis** - Compare code across different branches
- [ ] **Code change analysis** - Understand diffs and pull requests
- [ ] **Integration with IDEs** - VS Code extension for seamless workflow
- [ ] **Team collaboration** - Shared sessions and annotations
- [ ] **Export functionality** - Save conversations and insights
- [ ] **Custom AI models** - Fine-tuned models for specific domains
- [ ] **API documentation** - Interactive API explorer and docs

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
# Backend development with hot reload
cd triage.flow
python -m uvicorn src.main:app --reload --port 8000

# Frontend development with Vite
cd issue-flow-ai-prompt
npm run dev

# Monitor performance and caching
curl http://localhost:8000/cache-stats
```

### Key Development Areas
- **RAG improvements** - Better context selection and ranking
- **UI/UX enhancements** - More intuitive file navigation
- **Performance optimization** - Faster indexing and response times
- **Cost optimization** - Better prompt caching strategies
- **New AI providers** - Support for additional model providers

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [React](https://reactjs.org) and [FastAPI](https://fastapi.tiangolo.com)
- UI components from [shadcn/ui](https://ui.shadcn.com)
- Icons by [Lucide](https://lucide.dev)
- AI providers: [OpenAI](https://openai.com) and [OpenRouter](https://openrouter.ai)
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Code parsing via [tree-sitter](https://tree-sitter.github.io/)
- Syntax highlighting by [react-syntax-highlighter](https://github.com/react-syntax-highlighter/react-syntax-highlighter)
