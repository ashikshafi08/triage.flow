"""
Onboarding-Specific Prompts for AI Assistant

Contains specialized prompts designed for developer onboarding scenarios,
educational guidance, and personalized learning experiences.
"""

from typing import Dict, List, Optional
from .developer_profile import DeveloperProfile, ExperienceLevel, LearningStyle

class OnboardingPrompts:
    """
    Collection of prompts specifically designed for developer onboarding
    
    All prompts are crafted to be:
    - Educational and encouraging
    - Personalized based on developer profile
    - Focused on learning outcomes
    - Practical and actionable
    """
    
    def get_system_prompt(self) -> str:
        """Core system prompt for onboarding AI assistant"""
        return """You are OnboardAI, an expert AI assistant specialized in developer onboarding. Your primary role is to help new developers learn, understand, and contribute to codebases effectively.

CORE PRINCIPLES:
1. **Educational Focus**: Always explain concepts clearly with appropriate detail level
2. **Encouragement**: Be supportive and encouraging, especially when developers struggle
3. **Practical Guidance**: Provide actionable steps and concrete examples
4. **Personalization**: Adapt explanations based on experience level and learning style
5. **Progressive Learning**: Build knowledge incrementally from basics to advanced concepts
6. **Safety First**: Prioritize code quality, security, and best practices

RESPONSE STYLE:
- Use clear, friendly, and professional language
- Include relevant code examples when helpful
- Provide context for why things work the way they do
- Offer multiple approaches when appropriate
- Ask clarifying questions when needed
- Reference documentation and learning resources

TEACHING APPROACH:
- Start with the big picture, then dive into details
- Use analogies and real-world examples
- Connect new concepts to existing knowledge
- Provide immediate feedback on progress
- Suggest practice exercises and next steps

Remember: Your goal is not just to answer questions, but to help developers become confident, capable contributors to the codebase."""

    def get_welcome_prompt(self, profile: DeveloperProfile) -> str:
        """Generate personalized welcome message"""
        experience_context = {
            ExperienceLevel.JUNIOR: "As someone new to professional development, I'll provide detailed explanations and plenty of examples to help you build confidence.",
            ExperienceLevel.MID: "With your existing development experience, I'll focus on helping you understand this specific codebase and team practices.",
            ExperienceLevel.SENIOR: "Given your senior experience, I'll highlight the unique architectural patterns and advanced concepts in this codebase."
        }.get(profile.experience_level, "I'll adapt my guidance to your experience level.")
        
        learning_context = {
            LearningStyle.VISUAL: "I'll include diagrams and visual examples to help you understand the concepts.",
            LearningStyle.HANDS_ON: "I'll provide plenty of interactive examples and practical exercises.",
            LearningStyle.READING: "I'll reference comprehensive documentation and detailed explanations.",
            LearningStyle.AUDITORY: "I'll explain concepts verbally with clear step-by-step instructions."
        }.get(profile.learning_style, "I'll use a mixed approach to help you learn effectively.")
        
        return f"""Welcome to the team! I'm OnboardAI, your personal onboarding assistant. I'm here to help you navigate this codebase and become a productive team member.

Based on your profile:
- Experience Level: {profile.experience_level.value.title()}
- Role Focus: {profile.role.value.title()}  
- Learning Style: {profile.learning_style.value.replace('_', ' ').title()}
- Languages: {', '.join(profile.programming_languages)}

{experience_context}

{learning_context}

I can help you with:
🏗️ Understanding the codebase architecture
🛠️ Setting up your development environment  
📝 Finding good first tasks to work on
🧭 Navigating team processes and practices
🤝 Connecting concepts to your existing knowledge
📚 Learning new technologies used in this project

What would you like to explore first? Feel free to ask me anything about the codebase, development setup, or team practices!"""

    def get_codebase_overview_prompt(self, profile: DeveloperProfile) -> str:
        """Generate codebase overview tailored to developer profile"""
        detail_level = "high-level overview" if profile.experience_level == ExperienceLevel.SENIOR else "detailed explanation"
        focus_areas = {
            "frontend": "Pay special attention to UI components, state management, and user interaction patterns.",
            "backend": "Focus on API design, data flow, business logic, and system architecture.",
            "fullstack": "Cover both frontend and backend aspects, highlighting how they integrate.",
            "mobile": "Explain mobile-specific patterns, platform considerations, and app architecture.",
            "devops": "Emphasize deployment, infrastructure, CI/CD, and operational aspects.",
            "data": "Focus on data processing, analytics, storage patterns, and data flow.",
            "qa": "Highlight testing strategies, quality processes, and validation approaches."
        }.get(profile.role.value, "Provide a comprehensive overview of all system components.")
        
        return f"""Analyze this codebase and provide a {detail_level} suitable for a {profile.experience_level.value} {profile.role.value} developer.

{focus_areas}

Structure your response to include:

1. **Architecture Overview**: Main architectural patterns and design principles
2. **Key Components**: Most important modules, services, or components
3. **Data Flow**: How information moves through the system
4. **Technology Stack**: Main frameworks, libraries, and tools used
5. **Development Workflow**: How code changes are made and deployed
6. **Entry Points**: Where to start exploring the code
7. **Common Patterns**: Recurring code patterns and conventions
8. **Integration Points**: How different parts of the system connect

For a {profile.learning_style.value.replace('_', ' ')} learner, emphasize practical examples and clear explanations. Highlight areas that would be most relevant for someone with experience in {', '.join(profile.programming_languages)}.

Make this overview engaging and actionable - not just descriptive, but educational."""

    def get_codebase_analysis_prompt(self, profile: DeveloperProfile) -> str:
        """Analyze codebase for tour generation"""
        return f"""Analyze this codebase to create a personalized learning tour for a {profile.experience_level.value} {profile.role.value} developer.

Focus on identifying:

1. **Critical Learning Paths**: What sequence of files/concepts should they explore?
2. **Complexity Levels**: Which areas are beginner-friendly vs advanced?
3. **Role-Relevant Areas**: What parts are most important for a {profile.role.value} developer?
4. **Learning Dependencies**: What concepts must be understood before others?
5. **Practical Examples**: Where are the best examples of key patterns?

Consider their background in {', '.join(profile.programming_languages)} and learning style preference for {profile.learning_style.value.replace('_', ' ')} approaches.

Provide a structured analysis that can be used to create an optimal learning journey."""

    def get_tour_generation_prompt(self, profile: DeveloperProfile, analysis: str) -> str:
        """Generate interactive codebase tour"""
        return f"""Based on this analysis:

{analysis}

Create an interactive codebase tour for a {profile.experience_level.value} {profile.role.value} developer. Structure the tour as a series of steps, each with:

1. **Step Title**: Clear, engaging title
2. **Learning Objective**: What they'll understand after this step
3. **Files to Explore**: Specific files to examine
4. **Key Concepts**: Main concepts to focus on
5. **Guided Questions**: Questions to ask themselves while exploring
6. **Practical Exercise**: Small task to reinforce learning
7. **Connection to Previous**: How this builds on earlier steps
8. **Estimated Time**: How long this step should take

Make each step:
- **Digestible**: Not overwhelming, appropriate for their experience level
- **Interactive**: Encourages active exploration, not passive reading
- **Progressive**: Builds knowledge systematically
- **Practical**: Connects to real development work

Tailor the complexity and depth to someone with {profile.years_of_experience} years of experience who prefers {profile.learning_style.value.replace('_', ' ')} learning.

Generate 5-8 tour steps that create a comprehensive introduction to the codebase."""

    def get_first_task_analysis_prompt(self, profile: DeveloperProfile) -> str:
        """Analyze codebase for first task suggestions"""
        complexity_level = {
            ExperienceLevel.JUNIOR: "beginner-friendly tasks that build confidence",
            ExperienceLevel.MID: "intermediate tasks that leverage existing skills",
            ExperienceLevel.SENIOR: "meaningful tasks that provide architectural insight"
        }.get(profile.experience_level, "appropriately challenging tasks")
        
        return f"""Analyze this codebase to identify {complexity_level} for a {profile.role.value} developer.

Look for tasks that are:

1. **Self-Contained**: Can be completed without extensive system knowledge
2. **Educational**: Teach important patterns or concepts
3. **Safe**: Low risk of breaking existing functionality  
4. **Relevant**: Match their role focus ({profile.role.value})
5. **Achievable**: Appropriate for {profile.experience_level.value} level
6. **Engaging**: Interesting and motivating to work on

Consider areas like:
- Documentation improvements
- Test coverage additions
- Small feature implementations
- Bug fixes with clear scope
- Code refactoring opportunities
- Performance improvements
- Developer experience enhancements

For each potential task, evaluate:
- **Complexity**: How challenging is it?
- **Learning Value**: What will they learn?
- **Risk Level**: How safe is it to modify?
- **Time Estimate**: How long should it take?
- **Prerequisites**: What knowledge is needed?

Focus on tasks that would help someone with experience in {', '.join(profile.programming_languages)} learn this specific codebase effectively."""

    def get_concept_explanation_prompt(
        self, 
        concept: str, 
        experience_level: ExperienceLevel,
        learning_style: LearningStyle,
        file_context: Optional[str] = None
    ) -> str:
        """Generate educational concept explanations"""
        
        detail_level = {
            ExperienceLevel.JUNIOR: "detailed with basic concepts explained",
            ExperienceLevel.MID: "thorough with focus on practical application", 
            ExperienceLevel.SENIOR: "concise with emphasis on advanced patterns"
        }.get(experience_level, "comprehensive")
        
        teaching_approach = {
            LearningStyle.VISUAL: "Use diagrams, code examples, and visual representations",
            LearningStyle.HANDS_ON: "Provide interactive examples and exercises",
            LearningStyle.READING: "Give comprehensive explanations with references",
            LearningStyle.AUDITORY: "Explain step-by-step with clear verbal descriptions"
        }.get(learning_style, "Use multiple approaches")
        
        context_addition = f"\n\nUse this specific code context to make the explanation concrete:\n{file_context}" if file_context else ""
        
        return f"""Explain the concept "{concept}" in a {detail_level} way suitable for a {experience_level.value} developer.

{teaching_approach} to make the explanation engaging and effective.

Structure your explanation with:

1. **What it is**: Clear definition in simple terms
2. **Why it matters**: Practical importance and benefits
3. **How it works**: Step-by-step breakdown
4. **Real examples**: Concrete code examples from this codebase
5. **Common patterns**: How it's typically used
6. **Potential pitfalls**: What to watch out for
7. **Best practices**: Recommended approaches
8. **Related concepts**: What connects to this
9. **Practice suggestions**: How to reinforce learning

Make this explanation:
- **Clear**: Use language appropriate for the experience level
- **Practical**: Focus on real-world application
- **Engaging**: Keep it interesting and motivating
- **Actionable**: Include specific next steps

{context_addition}

Remember: The goal is education and understanding, not just information transfer."""

    def get_contextualized_question_prompt(
        self,
        question: str,
        question_type: str,
        profile: DeveloperProfile,
        current_phase: str,
        recent_context: Dict
    ) -> str:
        """Contextualize developer questions for onboarding"""
        return f"""You're helping a {profile.experience_level.value} {profile.role.value} developer who is currently in the {current_phase} phase of onboarding.

Their question: "{question}"
Question type: {question_type}

Developer context:
- Experience with: {', '.join(profile.programming_languages)}
- Learning style: {profile.learning_style.value.replace('_', ' ')}
- Goals: {', '.join(profile.goals)}
- Recent activity: {recent_context.get('recent_questions', [])}

Please provide an educational response that:

1. **Directly answers** their specific question
2. **Explains the reasoning** behind the answer
3. **Connects to their experience** with {', '.join(profile.programming_languages)}
4. **Provides context** about why this matters in this codebase
5. **Suggests next steps** for deeper learning
6. **Includes practical examples** when helpful

Tailor the complexity and detail level to their {profile.experience_level.value} experience level, and present information in a way that works well for {profile.learning_style.value.replace('_', ' ')} learners.

If this question indicates they might be struggling, provide extra encouragement and alternative approaches."""

    def get_question_classification_prompt(self, question: str) -> str:
        """Classify developer questions for appropriate handling"""
        return f"""Classify this developer question into one of these categories:

Question: "{question}"

Categories:
1. **setup** - Environment setup, installation, configuration
2. **architecture** - System design, patterns, overall structure  
3. **code_explanation** - How specific code works
4. **debugging** - Error resolution, troubleshooting
5. **workflow** - Development processes, git, deployment
6. **concept** - General programming or framework concepts
7. **task_guidance** - How to approach a specific task
8. **navigation** - Finding files, understanding organization
9. **best_practices** - Coding standards, conventions
10. **stuck** - General confusion or frustration

Return only the category name that best fits this question."""

    def get_stuck_assistance_prompt(
        self,
        current_task: str,
        stuck_reason: str,
        profile: DeveloperProfile,
        recent_context: Dict
    ) -> str:
        """Generate assistance for stuck developers"""
        return f"""A {profile.experience_level.value} {profile.role.value} developer is stuck and needs help.

Current task: {current_task}
Why they're stuck: {stuck_reason}
Recent context: {recent_context}

Provide supportive, actionable assistance that includes:

1. **Immediate Relief**: Address their frustration with encouragement
2. **Problem Breakdown**: Break down what might be causing the issue
3. **Step-by-Step Help**: Concrete next steps to try
4. **Alternative Approaches**: Different ways to tackle the problem
5. **Learning Opportunity**: What they can learn from this situation
6. **When to Seek Help**: When to escalate to a human mentor

Be especially supportive and encouraging. Remember that being stuck is a normal part of learning, and your role is to help them move forward while building confidence.

Tailor your assistance to:
- Their {profile.experience_level.value} experience level
- Their preference for {profile.learning_style.value.replace('_', ' ')} learning
- Their background in {', '.join(profile.programming_languages)}

Make this feel like having a patient, knowledgeable mentor by their side."""

    def get_learning_goals_prompt(self, profile: DeveloperProfile) -> str:
        """Generate personalized learning goals"""
        return f"""Based on this developer profile, suggest 5-7 personalized learning goals for their onboarding journey:

Profile:
- Role: {profile.role.value}
- Experience: {profile.experience_level.value}
- Languages: {', '.join(profile.programming_languages)}
- Frameworks: {', '.join(profile.frameworks)}
- Stated goals: {', '.join(profile.goals)}

Generate SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound) that:

1. **Build on existing skills** while learning new ones
2. **Progress logically** from basic to advanced understanding
3. **Align with their role** as a {profile.role.value} developer  
4. **Consider their experience level** ({profile.experience_level.value})
5. **Include both technical and soft skills**
6. **Are achievable** within a typical onboarding timeline

For each goal, include:
- **Goal statement**: What they'll achieve
- **Why it matters**: Relevance to their role and growth
- **Success criteria**: How they'll know they've achieved it
- **Estimated timeline**: When to complete it
- **Key milestones**: Intermediate checkpoints

Make these goals motivating and clearly connected to becoming an effective contributor to this specific codebase and team."""

    def get_progress_assessment_prompt(
        self,
        completed_steps: List[str],
        time_spent: Dict[str, int],
        help_requests: List[Dict],
        profile: DeveloperProfile
    ) -> str:
        """Assess learning progress and provide feedback"""
        return f"""Assess the onboarding progress for this {profile.experience_level.value} {profile.role.value} developer:

Completed steps: {completed_steps}
Time spent per step: {time_spent}
Help requests: {len(help_requests)} total
Recent help topics: {[req.get('question_type', 'unknown') for req in help_requests[-3:]]}

Expected timeline for {profile.experience_level.value} level: {self._get_expected_timeline(profile)}

Provide a comprehensive progress assessment including:

1. **Overall Progress**: How are they doing compared to expectations?
2. **Strengths Observed**: What are they doing well?
3. **Learning Patterns**: What does their help-seeking behavior indicate?
4. **Areas for Focus**: Where should they concentrate next?
5. **Pace Analysis**: Are they moving too fast/slow for their level?
6. **Recommendations**: Specific suggestions for improvement
7. **Encouragement**: Positive reinforcement and motivation
8. **Risk Factors**: Any concerning patterns to address

Be honest but encouraging, and provide actionable insights that help both the developer and their mentor understand the learning journey."""

    def get_adaptive_difficulty_prompt(
        self,
        current_task: str,
        difficulty_feedback: str,
        profile: DeveloperProfile
    ) -> str:
        """Adjust task difficulty based on feedback"""
        return f"""A {profile.experience_level.value} developer found the current task "{current_task}" to be {difficulty_feedback}.

Based on this feedback, provide recommendations for:

1. **Immediate Adjustment**: How to modify the current task
2. **Future Calibration**: How to better match difficulty to their level
3. **Skill Gap Analysis**: What skills might need development
4. **Alternative Approaches**: Different ways to tackle similar tasks
5. **Confidence Building**: How to maintain motivation and progress

If the task was "too_hard":
- Break it into smaller, manageable pieces
- Identify prerequisite knowledge gaps
- Suggest preparatory exercises
- Provide additional resources

If the task was "too_easy":
- Add complexity or additional requirements
- Suggest related advanced concepts to explore
- Propose extension activities
- Connect to more challenging work

Always maintain a growth mindset and help them see challenges as learning opportunities."""

    def get_team_integration_prompt(self, profile: DeveloperProfile) -> str:
        """Guide team integration and collaboration"""
        return f"""Help this {profile.experience_level.value} {profile.role.value} developer integrate effectively with the team.

Provide guidance on:

1. **Communication Patterns**: How to engage with team members
2. **Meeting Participation**: How to contribute in standups, reviews, etc.
3. **Code Reviews**: How to give and receive feedback effectively  
4. **Question Asking**: When and how to ask for help
5. **Knowledge Sharing**: How to share learnings with others
6. **Collaboration Tools**: Effective use of team communication platforms
7. **Cultural Fit**: Understanding team dynamics and values

Consider their:
- Experience level: {profile.experience_level.value}
- Communication style: {profile.communication_style}
- Mentorship preference: {"seeks mentorship" if profile.prefers_mentorship else "prefers independence"}

Provide specific, actionable advice that helps them become a valued team member while staying true to their working style and preferences."""

    # Helper methods
    
    def _get_expected_timeline(self, profile: DeveloperProfile) -> str:
        """Get expected timeline based on profile"""
        timelines = {
            ExperienceLevel.JUNIOR: "6-8 weeks for full productivity",
            ExperienceLevel.MID: "3-4 weeks for full productivity", 
            ExperienceLevel.SENIOR: "2-3 weeks for full productivity"
        }
        return timelines.get(profile.experience_level, "4-6 weeks for full productivity")