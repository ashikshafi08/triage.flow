# OnboardAI - Intelligent Developer Onboarding Engine

OnboardAI transforms the existing triage.flow codebase into a comprehensive AI-powered developer onboarding platform. It provides personalized learning experiences, adaptive workflows, and intelligent guidance to reduce developer onboarding time by 70%.

## 🎯 Key Features

### 🌐 **GitHub URL Support** ✨ **NEW**
- **Direct GitHub Integration**: Works with any public GitHub repository URL
- **Automatic Cloning**: Seamlessly clones and analyzes repositories
- **Repository Caching**: Efficiently handles multiple sessions with the same repo
- **URL Examples**: `https://github.com/virattt/ai-hedge-fund/`, `https://github.com/microsoft/vscode/`

### 🧠 Intelligent AI Core (`OnboardingAICore`)
- **Personalized Learning**: Adapts to developer experience level, role, and learning style
- **Educational Responses**: Provides explanations that teach, not just answer
- **Progress Tracking**: Monitors learning journey with adaptive recommendations
- **Context-Aware**: Understands developer's current learning phase and goals

### 🤖 Enhanced Agentic Explorer (`OnboardingAgenticExplorer`)
- **Educational Tools**: Specialized tools for concept explanations and practice exercises
- **Learning Analytics**: Tracks understanding patterns and suggests improvements
- **Difficulty Adaptation**: Adjusts complexity based on real-time feedback
- **Structured Learning**: Organizes knowledge building systematically

### 📋 Smart Workflow Engine (`OnboardingWorkflowEngine`)
- **Personalized Workflows**: Creates custom onboarding paths for each developer
- **Adaptive Timing**: Adjusts time estimates based on experience and progress
- **Role-Specific Steps**: Tailors activities to frontend, backend, fullstack, etc.
- **Progress Dependencies**: Ensures prerequisite knowledge before advancing

### 👤 Developer Profiling (`DeveloperProfile`)
- **Experience Assessment**: Accurately categorizes skill levels and knowledge gaps
- **Learning Style Detection**: Adapts to visual, hands-on, reading, or auditory preferences
- **Goal Alignment**: Connects onboarding activities to developer career goals
- **Continuous Adaptation**: Updates profile based on learning feedback

## 🚀 Quick Start

### 1. Basic Usage

```python
from src.onboarding import OnboardingAICore, DeveloperProfile, ExperienceLevel, Role

# Create developer profile
profile = DeveloperProfile()
profile.experience_level = ExperienceLevel.MID
profile.role = Role.FULLSTACK
profile.programming_languages = ["python", "javascript"]
profile.learning_style = LearningStyle.HANDS_ON

# Initialize OnboardAI
ai_core = OnboardingAICore(
    workspace_id="my_company",
    user_id="new_developer",
    repo_path="/path/to/codebase",
    developer_profile=profile
)

# Start onboarding session
session = await ai_core.start_onboarding_session({
    "experience_level": "mid",
    "role": "fullstack",
    "goals": ["understand_architecture", "contribute_effectively"]
})

# Ask questions with educational context
response = await ai_core.ask_question(
    "How is this codebase organized and what are the main components?"
)

print(response["response"])
```

### 2. Using the REST API

```bash
# Create developer profile
curl -X POST "http://localhost:8000/api/onboarding/profile/survey" \
  -H "Content-Type: application/json" \
  -d '{
    "experience_level": "mid",
    "role": "fullstack",
    "programming_languages": ["python", "javascript"],
    "learning_style": "hands_on",
    "goals": ["understand_architecture"]
  }'

# Start onboarding session with GitHub URL
curl -X POST "http://localhost:8000/api/onboarding/session/start" \
  -d "user_id=dev123&workspace_id=company&repo_path=https://github.com/virattt/ai-hedge-fund/"

# Or with local path
curl -X POST "http://localhost:8000/api/onboarding/session/start" \
  -d "user_id=dev123&workspace_id=company&repo_path=/path/to/repo"

# Ask onboarding question
curl -X POST "http://localhost:8000/api/onboarding/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the overall architecture of this system?",
    "user_id": "dev123",
    "workspace_id": "company",
    "repo_path": "https://github.com/virattt/ai-hedge-fund/"
  }'
```

### 3. Run the Demo

```bash
# Run interactive demo with local repository
cd src/onboarding
python demo_onboarding.py

# Or specify custom local repository
python demo_onboarding.py --repo-path /path/to/your/repo

# NEW: GitHub URL Demo
python demo_github_onboarding.py --interactive

# Or run directly with a GitHub repository
python demo_github_onboarding.py --repo-url https://github.com/virattt/ai-hedge-fund/
```

## 🏗️ Architecture Overview

```
OnboardAI System Architecture

┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              onboarding_ai.py                           │   │
│  │  - REST endpoints for all onboarding functions         │   │
│  │  - Request/response models and validation              │   │
│  │  - Session management and error handling               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI Core Engine                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            OnboardingAICore                             │   │
│  │  - Orchestrates entire onboarding experience           │   │
│  │  - Personalizes responses based on profile             │   │
│  │  - Tracks learning progress and adapts               │   │
│  │  - Integrates all onboarding components               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Agentic        │  │   Workflow      │  │   Developer     │
│  Explorer       │  │   Engine        │  │   Profile       │
│                 │  │                 │  │                 │
│ - Educational   │  │ - Personalized  │  │ - Experience    │
│   tool usage    │  │   workflows     │  │   assessment    │
│ - Learning      │  │ - Adaptive      │  │ - Learning      │
│   analytics     │  │   timing        │  │   preferences   │
│ - Concept       │  │ - Role-specific │  │ - Progress      │
│   explanations  │  │   steps         │  │   tracking      │
│ - Practice      │  │ - Dependencies  │  │ - Feedback      │
│   exercises     │  │   management    │  │   adaptation    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Enhanced Agent Tools                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Context-Aware Tools + Onboarding-Specific Tools       │   │
│  │  - explain_concept - track_learning_progress           │   │
│  │  - difficulty_feedback - find_related_concepts         │   │
│  │  - generate_practice_exercise                          │   │
│  │  + All existing codebase exploration tools             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LlamaIndex Foundation                           │
│  - ReActAgent with educational prompts                         │
│  - Enhanced system prompts for teaching                        │
│  - Memory management for learning context                      │
│  - Multi-model LLM setup (cost-efficient + high-quality)       │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Components Deep Dive

### OnboardingAICore
The central orchestrator that coordinates all onboarding activities:

**Key Methods:**
- `start_onboarding_session()` - Initialize personalized onboarding
- `ask_question()` - Handle educational Q&A with context
- `explain_concept()` - Provide detailed concept explanations
- `generate_codebase_tour()` - Create interactive learning paths
- `suggest_first_tasks()` - Recommend appropriate starter tasks
- `track_progress()` - Monitor and adapt learning journey

**Personalization Features:**
- Adapts responses to experience level (junior/mid/senior)
- Customizes explanations for learning style (visual/hands-on/reading)
- Connects concepts to developer's programming background
- Provides role-specific guidance (frontend/backend/fullstack)

### OnboardingAgenticExplorer
Enhanced version of the base AgenticCodebaseExplorer with educational focus:

**Additional Tools:**
- `explain_concept` - Educational explanations with examples
- `track_learning_progress` - Progress monitoring with encouragement
- `difficulty_feedback` - Adaptive difficulty based on feedback
- `find_related_concepts` - Suggest connected learning topics
- `generate_practice_exercise` - Create hands-on learning activities

**Learning Analytics:**
- Tracks concepts learned and time spent
- Identifies learning patterns and preferences
- Generates achievement badges and milestones
- Provides personalized learning recommendations

### OnboardingWorkflowEngine
Creates and manages personalized onboarding workflows:

**Workflow Features:**
- **Role-Based Templates**: Different paths for different roles
- **Experience Adjustments**: Timing and complexity based on skill level
- **Learning Dependencies**: Ensures prerequisite knowledge
- **Adaptive Steps**: Modifies based on progress and feedback

**Step Types:**
- `WELCOME` - Introduction and orientation
- `ASSESSMENT` - Skill and preference evaluation
- `ENVIRONMENT_SETUP` - Development environment configuration
- `CODEBASE_EXPLORATION` - Guided code learning
- `FIRST_TASK` - Practical hands-on contribution
- `TEAM_INTEGRATION` - Social and process integration

### DeveloperProfile
Comprehensive profiling system for personalization:

**Profile Attributes:**
- **Experience Level**: Junior, Mid, Senior, Lead
- **Role Focus**: Frontend, Backend, Fullstack, Mobile, DevOps, Data, QA
- **Learning Style**: Visual, Hands-on, Reading, Auditory, Mixed
- **Technical Background**: Languages, frameworks, tools
- **Learning Preferences**: Pace, mentorship, structure preferences

**Adaptive Features:**
- Updates profile based on learning feedback
- Adjusts time estimates based on actual performance
- Modifies complexity based on comprehension patterns
- Personalizes resource recommendations

## 🎓 Educational Philosophy

OnboardAI is built on proven educational principles:

### 1. **Personalized Learning**
- Adapts to individual learning styles and preferences
- Adjusts complexity based on experience level
- Connects new concepts to existing knowledge

### 2. **Progressive Complexity**
- Starts with foundational concepts
- Gradually introduces advanced topics
- Ensures solid understanding before progression

### 3. **Active Learning**
- Encourages exploration and experimentation
- Provides hands-on practice exercises
- Promotes question-asking and curiosity

### 4. **Contextual Understanding**
- Explains not just "what" but "why"
- Connects code patterns to business logic
- Relates concepts to real-world applications

### 5. **Continuous Feedback**
- Monitors learning progress continuously
- Adapts based on difficulty feedback
- Provides encouragement and motivation

## 🔌 Integration Points

### Existing Triage.Flow Integration
OnboardAI builds on the existing codebase:
- **Reuses**: AgenticCodebaseExplorer, all existing tools, LLM configuration
- **Extends**: Enhanced prompts, educational tools, progress tracking
- **Adds**: Developer profiling, workflow management, learning analytics

### External Integrations
- **GitHub**: Repository analysis, issue tracking, PR management
- **Slack**: Team communication, progress notifications, help requests
- **Analytics**: Learning metrics, ROI calculations, performance tracking

## 📊 Learning Analytics

OnboardAI provides comprehensive learning analytics:

### Individual Metrics
- **Learning Velocity**: Concepts learned per hour
- **Comprehension Patterns**: Areas of strength and challenge
- **Engagement Level**: Question frequency and depth
- **Progress Milestones**: Achievements and certifications

### Team Metrics
- **Onboarding Efficiency**: Time to productivity comparison
- **Knowledge Gaps**: Common learning challenges
- **Resource Effectiveness**: Most helpful learning materials
- **Mentor Load**: Distribution of mentorship needs

### Organizational Metrics
- **ROI Calculations**: Cost savings from faster onboarding
- **Retention Impact**: Correlation with employee satisfaction
- **Process Optimization**: Workflow improvement opportunities
- **Scalability Metrics**: Capacity for handling growth

## 🚀 Getting Started with Development

### Prerequisites
- Python 3.9+
- LlamaIndex
- FastAPI
- OpenAI/Anthropic API keys

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Run demo
python src/onboarding/demo_onboarding.py
```

### Testing
```bash
# Run unit tests
pytest src/onboarding/tests/

# Run integration tests
pytest src/onboarding/tests/integration/

# Run full onboarding flow test
python src/onboarding/demo_onboarding.py --repo-path ./test-repo
```

## 🤝 Contributing

OnboardAI is designed to be extensible and customizable:

### Adding New Learning Tools
1. Create tool function in `OnboardingAgenticExplorer`
2. Add tool description and parameters
3. Update system prompts to reference new tool
4. Add tests and documentation

### Customizing Workflows
1. Define new step types in `StepType` enum
2. Create step templates in `OnboardingWorkflowEngine`
3. Add role-specific workflow templates
4. Test with different developer profiles

### Extending Personalization
1. Add new attributes to `DeveloperProfile`
2. Update survey collection in API
3. Modify adaptation logic in AI core
4. Add corresponding frontend changes

## 📈 Roadmap

### Phase 1: Core Platform (✅ Complete)
- ✅ AI-powered onboarding engine
- ✅ Developer profiling and personalization
- ✅ Adaptive workflow management
- ✅ Educational content generation
- ✅ Progress tracking and analytics

### Phase 2: Advanced Features (🚧 Next)
- 🔄 Advanced learning analytics dashboard
- 🔄 Integration with popular IDEs
- 🔄 Automated skill assessments
- 🔄 Peer learning and collaboration features
- 🔄 Advanced personalization algorithms

### Phase 3: Enterprise Features (📋 Planned)
- 📋 Multi-tenant SaaS platform
- 📋 Enterprise SSO integration
- 📋 Advanced compliance and audit trails
- 📋 Custom branding and white-labeling
- 📋 Advanced reporting and ROI analytics

## 💡 Use Cases

### For Individual Developers
- **New Team Members**: Structured learning path for new hires
- **Technology Transitions**: Learning new languages or frameworks
- **Codebase Familiarity**: Understanding existing systems
- **Skill Development**: Targeted learning for career growth

### For Engineering Teams
- **Onboarding Optimization**: Reduce time to productivity
- **Knowledge Sharing**: Capture and distribute team knowledge
- **Mentorship Support**: Augment human mentors with AI assistance
- **Standard Practices**: Ensure consistent learning experiences

### For Organizations
- **Scaling Engineering**: Handle rapid team growth effectively
- **Developer Experience**: Improve satisfaction and retention
- **Knowledge Management**: Preserve and transfer institutional knowledge
- **ROI Optimization**: Measure and improve onboarding investment

## 🏆 Success Metrics

OnboardAI targets measurable improvements in developer onboarding:

### Time to Productivity
- **Target**: 70% reduction in onboarding time
- **Measurement**: Days until first meaningful contribution
- **Baseline**: Industry average of 3-6 months

### Learning Effectiveness
- **Target**: 90% completion rate for onboarding workflows
- **Measurement**: Percentage completing all required steps
- **Quality**: Post-onboarding knowledge assessments

### Developer Satisfaction
- **Target**: 4.5+ rating on onboarding experience
- **Measurement**: Survey feedback and engagement metrics
- **Retention**: Correlation with long-term employment

### Cost Savings
- **Target**: $10,000+ savings per developer onboarded
- **Calculation**: Reduced mentor time + faster productivity
- **ROI**: Measurable return on onboarding investment

---

**OnboardAI transforms developer onboarding from a chaotic, inconsistent process into a structured, personalized, and highly effective learning journey. It combines the power of AI with educational best practices to create the future of developer onboarding.**