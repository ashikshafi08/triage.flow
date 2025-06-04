<p align="center">
  <img src="triage_flow_logo.png" alt="triage.flow logo" width="180" height="180"/>
</p>

# triage.flow

<p align="center">
  <b>Agentic AI-Powered Interactive GitHub Repository Analysis with Advanced Chat Interface</b><br>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python"></a>
  <a href="https://reactjs.org"><img src="https://img.shields.io/badge/React-18%2B-blue.svg" alt="React"></a>
</p>

---

## 🚀 What is triage.flow?

A modern, web-based **agentic AI assistant** that helps you **understand, analyze, and explore entire GitHub repositories** through an interactive chat interface. Powered by **ReAct (Reasoning + Acting) agents** that autonomously use tools, reason through complex problems, and provide deep, code-aware insights with beautiful presentation, smart context retrieval, and cost-optimized AI interactions.

### ✨ Key Highlights

- **🤖 Agentic AI System** - ReAct agents that think, plan, and use tools autonomously to solve complex coding questions
- **🧠 Real-time Reasoning** - Watch the AI think through problems with live Thought/Action/Observation streaming
- **🛠️ Autonomous Tool Usage** - AI independently explores files, searches code, and analyzes patterns across your repository
- **🎯 Interactive Repository Chat** - ChatGPT-style conversation with full repository context and agent reasoning
- **🌲 Smart Codebase Explorer** - High-performance file picker with search and tree navigation  
- **📋 Enhanced Markdown** - Context-aware emojis, file linking, and professional formatting
- **📎 Smart File & Folder Mentions** - Use `@filename.py` or `@folder/path` with autocomplete to reference specific files/folders
- **👀 Integrated File Viewer** - Side-by-side file viewing pane with syntax highlighting and quick actions
- **⚡ Real-time Streaming** - Live AI responses with typing indicators and cost optimization
- **🎨 Modern UI/UX** - Professional, responsive interface with session management
- **🔍 Advanced RAG System** - Smart context sizing with 10-25 sources from indexed repositories
- **💰 Cost-Optimized AI** - OpenRouter prompt caching for 25-90% cost savings

---

## 🤖 Agentic AI Capabilities

Our system uses **ReAct (Reasoning + Acting) agents** that don't just answer questions - they autonomously reason through complex problems and take actions to solve them:

### 🧠 Autonomous Reasoning
- **Multi-step problem solving** - Breaks down complex questions into logical steps
- **Dynamic tool selection** - Chooses the right tools for each task automatically
- **Context-aware decisions** - Adapts approach based on repository structure and user intent
- **Self-correction** - Re-evaluates and adjusts when initial approaches don't work

### 🛠️ Intelligent Tool Usage
- **File exploration** - Autonomously navigates and examines repository structure
- **Smart code search** - Semantically searches across files for relevant patterns
- **Pattern analysis** - Identifies architectural patterns and relationships
- **Code generation** - Creates context-aware examples based on existing codebase patterns

### 📊 Real-time Transparency
- **Live reasoning display** - See exactly how the AI thinks through problems
- **Tool execution tracking** - Watch as tools are selected and executed
- **Step-by-step breakdown** - Understand the agent's decision-making process
- **Interactive feedback** - Agent adapts based on your responses and clarifications

---

## 🎨 Interface Features

| Feature | Description |
|---------|-------------|
| **Real-time Agent Reasoning** | Live display of Thought/Action/Observation cycles as the AI works |
| **Context-Aware Headers** | Headers automatically get relevant emojis (🐛 for bugs, 🔧 for fixes, etc.) |
| **Smart Bullet Points** | List items get contextual icons (❌ for errors, ✅ for success, 💡 for tips) |
| **File Cross-linking** | Automatic detection and highlighting of referenced files |
| **@-Mention Autocomplete** | Type `@` to get intelligent file/folder suggestions with fuzzy search |
| **Integrated File Viewer** | Click any file in the tree to view it in a side pane with syntax highlighting |
| **Agentic Step Visualization** | Beautiful UI for displaying AI reasoning steps with type-specific styling |
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

## 🏗️ Agentic Architecture

### ReAct Agent Framework
- **LlamaIndex-powered agents** with autonomous tool selection
- **Multi-step reasoning** with self-correction capabilities  
- **Real-time streaming** of thought processes and tool executions
- **Context-aware memory** that remembers previous interactions and reasoning chains

### Intelligent Tool Ecosystem
- **`explore_directory`** - Autonomous file system navigation with metadata
- **`search_codebase`** - Semantic code search across entire repositories
- **`read_file`** - Smart file reading with size optimization
- **`analyze_file_structure`** - Deep structural analysis and pattern recognition
- **`find_related_files`** - Relationship discovery through imports and naming
- **`semantic_content_search`** - AI-powered content understanding
- **`generate_code_example`** - Context-aware code generation from repository patterns

### Live Reasoning Visualization
The UI provides real-time insight into the agent's decision-making:
- 🧠 **Purple Thought bubbles** - Shows reasoning and planning
- ⚡ **Blue Action blocks** - Displays tool selection and parameters  
- 👁️ **Green Observation panels** - Shows tool execution results
- 💬 **Teal Answer sections** - Final synthesized responses
- ⚠️ **Status indicators** - Real-time processing updates

---

## 🚀 Quick Start

1. **Clone and setup** the repository
2. **Start the backend** with repository indexing
3. **Launch the frontend** for the chat interface
4. **Create a new session** by entering any GitHub repo URL
5. **Watch the AI agent** clone, index, and analyze the repository
6. **Ask complex questions** and watch the agent reason through solutions step-by-step
7. **Use @-mentions** for specific file context and explore with the file tree
8. **Observe real-time reasoning** as the agent explores, searches, and synthesizes answers

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
6. **Click any file** in the tree to open it in the integrated viewer pane
7. **Type `@`** in the chat for intelligent file/folder autocomplete
8. Mention files with `@filename.py` or folders with `@folder/subfolder`

---

## 💬 How to Use the Agentic Chat Interface

### Agentic Repository Analysis Commands
- **Overview**: "What is this repository about?" - *Agent explores structure, reads key files, analyzes patterns*
- **Architecture**: "Explain the project structure" - *Agent navigates directories, examines relationships*
- **Code exploration**: "Show me the main components" - *Agent searches, categorizes, and explains findings*
- **Debugging**: "Help me understand this error in @file.py" - *Agent reads file, searches for related code, analyzes context*
- **Implementation**: "How does the authentication work?" - *Agent searches auth patterns, traces code flow*

### Watch the AI Think
- **Thought steps** show the agent's reasoning process
- **Action steps** display which tools the agent chooses to use
- **Observation steps** show the results of tool execution
- **Answer steps** provide the final synthesized response

### Smart Context Features
- **File references**: "Look at @config.py for the settings"
- **Folder exploration**: "What's in @folder/src/components?"
- **Multi-file analysis**: "Compare @file1.js and @file2.js"
- **Complex queries**: "Find all database-related files and explain the schema"
- **Autocomplete suggestions**: Type `@` to see file/folder suggestions with fuzzy search
- **Integrated file viewing**: Click files in the explorer to view them alongside chat

### Pro Tips
- Use **@** for file autocomplete (optimized for large repos)
- **Arrow keys** navigate autocomplete suggestions, **Enter/Tab** to select
- **Ask complex questions** - the agent will break them down automatically
- **Watch the reasoning** - each step shows how the AI approaches your problem
- **Folder mentions** automatically include all files in context
- **File viewer pane** opens when clicking files in the explorer tree
- The AI **remembers conversation history** and previous reasoning
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
- **@-mention autocomplete** - intelligent file/folder suggestions with keyboard navigation
- **Integrated file viewer** - side-by-side code viewing with syntax highlighting and actions
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

---

## 🤝 Contributing

We love contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [React](https://reactjs.org) and [FastAPI](https://fastapi.tiangolo.com)
- UI components from [shadcn/ui](https://ui.shadcn.com)
- AI providers: [OpenAI](https://openai.com) and [OpenRouter](https://openrouter.ai)
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Code parsing via [tree-sitter](https://tree-sitter.github.io/)
