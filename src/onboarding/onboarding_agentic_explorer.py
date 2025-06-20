"""
Onboarding-Specific Agentic Explorer

Extends the core AgenticCodebaseExplorer with onboarding-focused capabilities,
educational guidance, and personalized learning experiences.
"""

import logging
import json
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from ..agent_tools.core import AgenticCodebaseExplorer
from ..agent_tools.response_handling import format_agentic_response
from ..config import settings
from .developer_profile import DeveloperProfile, ExperienceLevel, LearningStyle
from .onboarding_prompts import OnboardingPrompts

if TYPE_CHECKING:
    from ..issue_rag import IssueAwareRAG

logger = logging.getLogger(__name__)

class OnboardingAgenticExplorer(AgenticCodebaseExplorer):
    """
    Specialized agentic explorer for developer onboarding
    
    Enhances the base explorer with:
    - Educational-focused system prompts
    - Personalized explanations based on developer profile
    - Learning progress tracking
    - Onboarding-specific tool behaviors
    - Structured educational responses
    """
    
    def __init__(
        self,
        session_id: str,
        repo_path: str,
        developer_profile: DeveloperProfile,
        issue_rag_system: Optional['IssueAwareRAG'] = None,
        custom_system_prompt: Optional[str] = None
    ):
        # Initialize base explorer first
        super().__init__(session_id, repo_path, issue_rag_system)
        
        # Onboarding-specific attributes
        self.developer_profile = developer_profile
        self.onboarding_prompts = OnboardingPrompts()
        
        # Learning state tracking
        self.learning_interactions = []
        self.current_learning_context = {}
        self.difficulty_adjustments = {}
        
        # Override system prompt with onboarding-focused one
        self.onboarding_system_prompt = custom_system_prompt or self._create_onboarding_system_prompt()
        
        # Create onboarding-specialized agent
        self.onboarding_agent = self._create_onboarding_agent(self.onboarding_system_prompt)
        
        # Add onboarding-specific tools
        self._add_onboarding_tools()
        
        logger.info(f"OnboardingAgenticExplorer initialized for {developer_profile.experience_level.value} {developer_profile.role.value}")
    
    def _create_onboarding_system_prompt(self) -> str:
        """Create specialized system prompt for onboarding"""
        base_prompt = self.onboarding_prompts.get_system_prompt()
        
        # Add profile-specific context
        profile_context = f"""

DEVELOPER PROFILE CONTEXT:
- Experience Level: {self.developer_profile.experience_level.value.title()}
- Role Focus: {self.developer_profile.role.value.title()}
- Learning Style: {self.developer_profile.learning_style.value.replace('_', ' ').title()}
- Programming Languages: {', '.join(self.developer_profile.programming_languages)}
- Years of Experience: {self.developer_profile.years_of_experience}
- Learning Goals: {', '.join(self.developer_profile.goals)}

PERSONALIZATION GUIDELINES:
{self._get_personalization_guidelines()}

RESPONSE APPROACH:
- Adjust technical depth to {self.developer_profile.experience_level.value} level
- Use {self.developer_profile.learning_style.value.replace('_', ' ')} teaching methods
- Connect concepts to their {', '.join(self.developer_profile.programming_languages)} background
- Provide examples relevant to {self.developer_profile.role.value} development
- Encourage and support their learning journey

Remember: You're not just exploring code, you're teaching and guiding a developer's learning journey."""
        
        return base_prompt + profile_context
    
    def _get_personalization_guidelines(self) -> str:
        """Get personalization guidelines based on profile"""
        guidelines = []
        
        # Experience-based guidelines
        if self.developer_profile.experience_level == ExperienceLevel.JUNIOR:
            guidelines.extend([
                "- Explain concepts thoroughly with detailed context",
                "- Break down complex topics into digestible steps",
                "- Provide more examples and analogies",
                "- Be encouraging and patient with questions",
                "- Define technical terms when first mentioned"
            ])
        elif self.developer_profile.experience_level == ExperienceLevel.SENIOR:
            guidelines.extend([
                "- Focus on architectural patterns and design decisions",
                "- Highlight advanced concepts and trade-offs",
                "- Compare approaches to industry best practices",
                "- Assume familiarity with basic programming concepts",
                "- Emphasize strategic and high-level thinking"
            ])
        else:  # MID level
            guidelines.extend([
                "- Balance detailed explanations with practical application",
                "- Connect new concepts to existing development knowledge",
                "- Provide real-world examples and use cases",
                "- Focus on both implementation and reasoning"
            ])
        
        # Learning style guidelines
        if self.developer_profile.learning_style == LearningStyle.VISUAL:
            guidelines.append("- Include ASCII diagrams, code structure visuals, and clear formatting")
        elif self.developer_profile.learning_style == LearningStyle.HANDS_ON:
            guidelines.append("- Provide interactive examples and suggest practice exercises")
        elif self.developer_profile.learning_style == LearningStyle.READING:
            guidelines.append("- Include comprehensive explanations and reference documentation")
        
        return "\n".join(guidelines)
    
    def _create_onboarding_agent(self, system_prompt: str) -> ReActAgent:
        """Create specialized agent for onboarding interactions"""
        return ReActAgent.from_tools(
            tools=self.tools,
            llm=self.base_llm,
            memory=self.memory,
            verbose=True,
            max_iterations=self._get_onboarding_max_iterations(),
            system_prompt=system_prompt
        )
    
    def _get_onboarding_max_iterations(self) -> int:
        """Get appropriate max iterations for onboarding scenarios"""
        # Onboarding queries often require more thorough exploration
        base_iterations = getattr(settings, 'AGENTIC_MAX_ITERATIONS', 12)
        
        # Adjust based on experience level
        if self.developer_profile.experience_level == ExperienceLevel.JUNIOR:
            return int(base_iterations * 1.5)  # More thorough for juniors
        elif self.developer_profile.experience_level == ExperienceLevel.SENIOR:
            return base_iterations  # Standard for seniors
        else:
            return int(base_iterations * 1.2)  # Slightly more for mid-level
    
    def _add_onboarding_tools(self):
        """Add onboarding-specific tools to the tool set"""
        onboarding_tools = [
            self._create_concept_explanation_tool(),
            self._create_learning_progress_tool(),
            self._create_difficulty_feedback_tool(),
            self._create_related_concepts_tool(),
            self._create_practice_exercise_tool()
        ]
        
        # Add new tools to existing toolset
        self.tools.extend(onboarding_tools)
        
        # Recreate agent with updated toolset
        self.onboarding_agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.base_llm,
            memory=self.memory,
            verbose=True,
            max_iterations=self._get_onboarding_max_iterations(),
            system_prompt=self.onboarding_system_prompt
        )
    
    def _create_concept_explanation_tool(self) -> FunctionTool:
        """Tool for explaining programming concepts in educational context"""
        
        def explain_concept(
            concept: str,
            file_context: Optional[str] = None,
            difficulty_level: str = "auto"
        ) -> str:
            """
            Provide educational explanation of a programming concept
            
            Args:
                concept: The concept to explain (e.g., "dependency injection", "async/await")
                file_context: Optional code context to make explanation concrete
                difficulty_level: Explanation depth (beginner/intermediate/advanced/auto)
            """
            try:
                # Auto-determine difficulty if not specified
                if difficulty_level == "auto":
                    difficulty_level = self._get_auto_difficulty_level()
                
                # Generate explanation using onboarding prompts
                explanation_prompt = self.onboarding_prompts.get_concept_explanation_prompt(
                    concept=concept,
                    experience_level=self.developer_profile.experience_level,
                    learning_style=self.developer_profile.learning_style,
                    file_context=file_context
                )
                
                # Track learning interaction
                self._track_learning_interaction("concept_explanation", {
                    "concept": concept,
                    "difficulty_level": difficulty_level,
                    "has_context": bool(file_context)
                })
                
                return f"Educational explanation for '{concept}' at {difficulty_level} level:\n\n" + explanation_prompt
                
            except Exception as e:
                logger.error(f"Error in concept explanation: {e}")
                return f"Error explaining concept '{concept}': {str(e)}"
        
        return FunctionTool.from_defaults(
            fn=explain_concept,
            name="explain_concept",
            description="Provide educational explanations of programming concepts tailored to the developer's experience level"
        )
    
    def _create_learning_progress_tool(self) -> FunctionTool:
        """Tool for tracking and reporting learning progress"""
        
        def track_learning_progress(
            topic: str,
            understanding_level: str,
            time_spent_minutes: int = 0,
            questions: Optional[str] = None
        ) -> str:
            """
            Track learning progress for a specific topic
            
            Args:
                topic: What was learned (e.g., "React components", "API design")
                understanding_level: How well understood (beginner/intermediate/advanced/mastered)
                time_spent_minutes: Time spent learning this topic
                questions: Any remaining questions or areas of confusion
            """
            try:
                progress_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "topic": topic,
                    "understanding_level": understanding_level,
                    "time_spent_minutes": time_spent_minutes,
                    "questions": questions,
                    "session_id": self.session_id
                }
                
                self.learning_interactions.append(progress_entry)
                
                # Generate progress summary
                summary = f"Learning progress recorded for '{topic}':\n"
                summary += f"- Understanding level: {understanding_level}\n"
                summary += f"- Time spent: {time_spent_minutes} minutes\n"
                
                if questions:
                    summary += f"- Questions/concerns: {questions}\n"
                
                # Provide encouragement and next steps
                encouragement = self._generate_learning_encouragement(understanding_level, topic)
                summary += f"\n{encouragement}"
                
                return summary
                
            except Exception as e:
                logger.error(f"Error tracking learning progress: {e}")
                return f"Error tracking progress: {str(e)}"
        
        return FunctionTool.from_defaults(
            fn=track_learning_progress,
            name="track_learning_progress",
            description="Track learning progress and provide personalized encouragement"
        )
    
    def _create_difficulty_feedback_tool(self) -> FunctionTool:
        """Tool for collecting and adapting to difficulty feedback"""
        
        def provide_difficulty_feedback(
            current_task: str,
            difficulty_rating: str,
            specific_challenges: Optional[str] = None,
            suggested_adjustments: Optional[str] = None
        ) -> str:
            """
            Provide feedback on task difficulty to adapt future explanations
            
            Args:
                current_task: The task being worked on
                difficulty_rating: too_easy/just_right/too_hard/overwhelming
                specific_challenges: What specifically was challenging
                suggested_adjustments: How to improve the experience
            """
            try:
                feedback_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "task": current_task,
                    "difficulty_rating": difficulty_rating,
                    "challenges": specific_challenges,
                    "suggestions": suggested_adjustments,
                    "profile_snapshot": self.developer_profile.get_personalization_context()
                }
                
                # Store feedback for adaptation
                if current_task not in self.difficulty_adjustments:
                    self.difficulty_adjustments[current_task] = []
                self.difficulty_adjustments[current_task].append(feedback_entry)
                
                # Generate adaptive response
                response = f"Thank you for the feedback on '{current_task}'!\n\n"
                
                if difficulty_rating == "too_hard":
                    response += "I'll adjust future explanations to:\n"
                    response += "- Break down concepts into smaller steps\n"
                    response += "- Provide more background context\n"
                    response += "- Include additional examples\n"
                    response += "- Suggest preparatory learning resources\n"
                    
                elif difficulty_rating == "too_easy":
                    response += "I'll make future content more challenging by:\n"
                    response += "- Diving deeper into advanced concepts\n"
                    response += "- Introducing related complex topics\n"
                    response += "- Providing less step-by-step guidance\n"
                    response += "- Focusing on architectural patterns\n"
                    
                elif difficulty_rating == "just_right":
                    response += "Great! I'll maintain this level of complexity for similar topics.\n"
                
                if specific_challenges:
                    response += f"\nI'll pay special attention to: {specific_challenges}\n"
                
                return response
                
            except Exception as e:
                logger.error(f"Error processing difficulty feedback: {e}")
                return f"Error processing feedback: {str(e)}"
        
        return FunctionTool.from_defaults(
            fn=provide_difficulty_feedback,
            name="difficulty_feedback",
            description="Collect feedback on task difficulty to personalize future learning experiences"
        )
    
    def _create_related_concepts_tool(self) -> FunctionTool:
        """Tool for finding and suggesting related learning concepts"""
        
        def find_related_concepts(
            current_concept: str,
            learning_depth: str = "moderate",
            include_prerequisites: bool = True,
            include_advanced: bool = True
        ) -> str:
            """
            Find concepts related to current learning topic
            
            Args:
                current_concept: The concept being learned
                learning_depth: How deep to go (shallow/moderate/deep)
                include_prerequisites: Include foundational concepts
                include_advanced: Include advanced related topics
            """
            try:
                related_concepts = {
                    "prerequisites": [],
                    "related": [],
                    "advanced": [],
                    "practical_applications": []
                }
                
                # This would typically use embeddings or knowledge graphs
                # For now, providing a structured template
                
                concept_lower = current_concept.lower()
                
                # Example mappings (in real implementation, this would be more sophisticated)
                if "component" in concept_lower and "react" in concept_lower:
                    related_concepts["prerequisites"] = ["JSX syntax", "JavaScript ES6", "Virtual DOM"]
                    related_concepts["related"] = ["Props", "State", "Event handling", "Component lifecycle"]
                    related_concepts["advanced"] = ["Higher-order components", "React hooks", "Context API"]
                    related_concepts["practical_applications"] = ["Form components", "List rendering", "Conditional rendering"]
                
                elif "api" in concept_lower:
                    related_concepts["prerequisites"] = ["HTTP methods", "JSON", "REST principles"]
                    related_concepts["related"] = ["Endpoints", "Status codes", "Authentication", "Rate limiting"]
                    related_concepts["advanced"] = ["GraphQL", "API versioning", "Caching strategies"]
                    related_concepts["practical_applications"] = ["CRUD operations", "Data fetching", "Error handling"]
                
                # Format response based on learning depth and preferences
                response = f"Related concepts for '{current_concept}':\n\n"
                
                if include_prerequisites and related_concepts["prerequisites"]:
                    response += "📚 Prerequisites to review:\n"
                    for prereq in related_concepts["prerequisites"]:
                        response += f"  - {prereq}\n"
                    response += "\n"
                
                if related_concepts["related"]:
                    response += "🔗 Related concepts to explore:\n"
                    for related in related_concepts["related"]:
                        response += f"  - {related}\n"
                    response += "\n"
                
                if include_advanced and related_concepts["advanced"]:
                    response += "🚀 Advanced topics for later:\n"
                    for advanced in related_concepts["advanced"]:
                        response += f"  - {advanced}\n"
                    response += "\n"
                
                if related_concepts["practical_applications"]:
                    response += "💡 Practical applications:\n"
                    for app in related_concepts["practical_applications"]:
                        response += f"  - {app}\n"
                
                return response
                
            except Exception as e:
                logger.error(f"Error finding related concepts: {e}")
                return f"Error finding related concepts: {str(e)}"
        
        return FunctionTool.from_defaults(
            fn=find_related_concepts,
            name="find_related_concepts",
            description="Find and suggest related learning concepts to expand understanding"
        )
    
    def _create_practice_exercise_tool(self) -> FunctionTool:
        """Tool for generating practice exercises"""
        
        def generate_practice_exercise(
            concept: str,
            difficulty: str = "auto",
            time_limit_minutes: int = 30,
            include_solution: bool = False
        ) -> str:
            """
            Generate a practice exercise for a specific concept
            
            Args:
                concept: The concept to practice
                difficulty: Exercise difficulty (beginner/intermediate/advanced/auto)
                time_limit_minutes: Suggested time limit
                include_solution: Whether to include solution hints
            """
            try:
                if difficulty == "auto":
                    difficulty = self._get_auto_difficulty_level()
                
                # Generate exercise based on concept and difficulty
                exercise = f"Practice Exercise: {concept}\n"
                exercise += f"Difficulty: {difficulty.title()}\n"
                exercise += f"Estimated time: {time_limit_minutes} minutes\n\n"
                
                # Exercise content would be generated based on concept
                # This is a template - real implementation would use AI generation
                exercise += f"Practice implementing {concept} in the context of this codebase.\n\n"
                exercise += "Steps:\n"
                exercise += "1. Identify relevant files and patterns\n"
                exercise += "2. Create a small, focused implementation\n"
                exercise += "3. Test your implementation\n"
                exercise += "4. Compare with existing patterns in the codebase\n\n"
                
                if include_solution:
                    exercise += "💡 Hints:\n"
                    exercise += f"- Look for similar patterns in the codebase\n"
                    exercise += f"- Start with the simplest possible implementation\n"
                    exercise += f"- Don't forget to handle edge cases\n"
                
                # Track exercise generation
                self._track_learning_interaction("practice_exercise", {
                    "concept": concept,
                    "difficulty": difficulty,
                    "time_limit": time_limit_minutes
                })
                
                return exercise
                
            except Exception as e:
                logger.error(f"Error generating practice exercise: {e}")
                return f"Error generating exercise: {str(e)}"
        
        return FunctionTool.from_defaults(
            fn=generate_practice_exercise,
            name="generate_practice_exercise",
            description="Generate personalized practice exercises for learning concepts"
        )
    
    async def ask_onboarding_question(
        self, 
        question: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main interface for onboarding questions with enhanced educational responses
        """
        try:
            logger.info(f"Onboarding question: {question[:100]}...")
            
            # Set current context for tools
            self.current_learning_context = context or {}
            
            # Use the onboarding agent for educational responses
            response = await self.onboarding_agent.achat(question)
            
            # Extract educational metadata from the response
            educational_metadata = self._extract_educational_metadata(str(response))
            
            # Generate additional learning resources
            related_concepts = await self._suggest_related_concepts(question)
            next_steps = self._suggest_next_learning_steps(question, educational_metadata)
            
            # Track this interaction
            self._track_learning_interaction("question_answer", {
                "question": question,
                "response_length": len(str(response)),
                "educational_metadata": educational_metadata
            })
            
            return {
                "response": str(response),
                "educational_metadata": educational_metadata,
                "related_concepts": related_concepts,
                "next_steps": next_steps,
                "learning_context": self.current_learning_context,
                "personalization_applied": self._get_applied_personalizations()
            }
            
        except Exception as e:
            logger.error(f"Error in onboarding question: {e}")
            return {
                "response": f"I encountered an error while processing your question: {str(e)}",
                "error": True
            }
    
    async def generate_learning_summary(self) -> Dict[str, Any]:
        """Generate a summary of learning progress and achievements"""
        try:
            total_interactions = len(self.learning_interactions)
            
            if total_interactions == 0:
                return {
                    "message": "No learning interactions recorded yet. Start exploring the codebase!",
                    "suggestions": [
                        "Ask about the overall architecture",
                        "Explore key files and directories",
                        "Learn about the development workflow"
                    ]
                }
            
            # Analyze learning patterns
            concepts_learned = self._extract_learned_concepts()
            time_spent = self._calculate_total_learning_time()
            difficulty_patterns = self._analyze_difficulty_patterns()
            
            return {
                "total_interactions": total_interactions,
                "concepts_learned": concepts_learned,
                "total_time_minutes": time_spent,
                "difficulty_patterns": difficulty_patterns,
                "achievements": self._generate_achievements(),
                "recommendations": self._generate_learning_recommendations(),
                "next_milestones": self._suggest_next_milestones()
            }
            
        except Exception as e:
            logger.error(f"Error generating learning summary: {e}")
            return {"error": f"Failed to generate learning summary: {str(e)}"}
    
    # Helper methods
    
    def _get_auto_difficulty_level(self) -> str:
        """Automatically determine appropriate difficulty level"""
        level_map = {
            ExperienceLevel.JUNIOR: "beginner",
            ExperienceLevel.MID: "intermediate",
            ExperienceLevel.SENIOR: "advanced",
            ExperienceLevel.LEAD: "expert"
        }
        return level_map.get(self.developer_profile.experience_level, "intermediate")
    
    def _track_learning_interaction(self, interaction_type: str, metadata: Dict[str, Any]):
        """Track learning interactions for progress analysis"""
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": interaction_type,
            "metadata": metadata,
            "session_id": self.session_id,
            "developer_profile": self.developer_profile.get_personalization_context()
        }
        self.learning_interactions.append(interaction)
    
    def _generate_learning_encouragement(self, understanding_level: str, topic: str) -> str:
        """Generate personalized encouragement based on progress"""
        encouragements = {
            "beginner": f"Great start with {topic}! Every expert was once a beginner. Keep exploring and asking questions.",
            "intermediate": f"You're making solid progress with {topic}! You're building a strong foundation.",
            "advanced": f"Excellent understanding of {topic}! You're ready to tackle more complex aspects.",
            "mastered": f"Outstanding mastery of {topic}! Consider helping others learn or exploring related advanced topics."
        }
        
        base_encouragement = encouragements.get(understanding_level, "Keep up the great learning!")
        
        # Add profile-specific encouragement
        if self.developer_profile.experience_level == ExperienceLevel.JUNIOR:
            base_encouragement += " Remember, it's normal to feel challenged - that's how we grow!"
        
        return base_encouragement
    
    def _extract_educational_metadata(self, response: str) -> Dict[str, Any]:
        """Extract educational metadata from response"""
        return {
            "concepts_mentioned": self._extract_concepts_from_text(response),
            "complexity_level": self._assess_response_complexity(response),
            "educational_elements": self._identify_educational_elements(response),
            "code_examples_count": response.count("```"),
            "explanation_depth": len(response.split('.'))  # Simple approximation
        }
    
    async def _suggest_related_concepts(self, question: str) -> List[str]:
        """Suggest concepts related to the question"""
        # Simplified implementation - would use embeddings in production
        concepts = []
        question_lower = question.lower()
        
        if "component" in question_lower:
            concepts.extend(["props", "state", "lifecycle", "hooks"])
        if "api" in question_lower:
            concepts.extend(["endpoints", "authentication", "middleware", "validation"])
        if "database" in question_lower:
            concepts.extend(["queries", "migrations", "relationships", "indexing"])
        
        return concepts[:5]  # Limit to top 5
    
    def _suggest_next_learning_steps(self, question: str, metadata: Dict[str, Any]) -> List[str]:
        """Suggest next learning steps based on current question"""
        steps = []
        
        # Base suggestions
        steps.append("Practice implementing the concepts discussed")
        steps.append("Explore related files in the codebase")
        
        # Add specific suggestions based on complexity
        if metadata.get("complexity_level", "medium") == "high":
            steps.append("Break down complex concepts into smaller parts")
            steps.append("Review prerequisites if needed")
        
        return steps
    
    def _get_applied_personalizations(self) -> List[str]:
        """Get list of personalizations applied to the response"""
        personalizations = []
        
        personalizations.append(f"Adjusted for {self.developer_profile.experience_level.value} experience level")
        personalizations.append(f"Optimized for {self.developer_profile.learning_style.value.replace('_', ' ')} learning style")
        
        if self.developer_profile.programming_languages:
            personalizations.append(f"Connected to {', '.join(self.developer_profile.programming_languages)} background")
        
        return personalizations
    
    def _extract_learned_concepts(self) -> List[str]:
        """Extract concepts that have been learned from interactions"""
        concepts = set()
        for interaction in self.learning_interactions:
            if interaction["type"] == "concept_explanation":
                concepts.add(interaction["metadata"].get("concept", ""))
            elif interaction["type"] == "question_answer":
                # Extract concepts from questions
                concepts.update(self._extract_concepts_from_text(interaction["metadata"].get("question", "")))
        
        return list(filter(None, concepts))
    
    def _calculate_total_learning_time(self) -> int:
        """Calculate total time spent learning"""
        total_time = 0
        for interaction in self.learning_interactions:
            if "time_spent_minutes" in interaction["metadata"]:
                total_time += interaction["metadata"]["time_spent_minutes"]
        
        return total_time
    
    def _analyze_difficulty_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in difficulty feedback"""
        feedback_items = []
        for adjustments in self.difficulty_adjustments.values():
            feedback_items.extend(adjustments)
        
        if not feedback_items:
            return {"message": "No difficulty feedback recorded yet"}
        
        ratings = [item["difficulty_rating"] for item in feedback_items]
        most_common = max(set(ratings), key=ratings.count) if ratings else "unknown"
        
        return {
            "total_feedback_items": len(feedback_items),
            "most_common_rating": most_common,
            "needs_adjustment": most_common in ["too_hard", "too_easy"]
        }
    
    def _generate_achievements(self) -> List[str]:
        """Generate achievements based on learning progress"""
        achievements = []
        
        total_concepts = len(self._extract_learned_concepts())
        if total_concepts >= 5:
            achievements.append(f"🎓 Concept Explorer - Learned {total_concepts} concepts")
        
        total_time = self._calculate_total_learning_time()
        if total_time >= 60:
            achievements.append(f"⏰ Dedicated Learner - {total_time // 60} hours of learning time")
        
        if len(self.learning_interactions) >= 10:
            achievements.append("💪 Active Participant - Asked 10+ questions")
        
        return achievements
    
    def _generate_learning_recommendations(self) -> List[str]:
        """Generate personalized learning recommendations"""
        recommendations = []
        
        # Based on learning patterns
        if self._calculate_total_learning_time() < 30:
            recommendations.append("Consider spending more time on hands-on exploration")
        
        concepts_learned = len(self._extract_learned_concepts())
        if concepts_learned < 3:
            recommendations.append("Try exploring fundamental concepts of the codebase")
        
        # Based on profile
        if self.developer_profile.learning_style == LearningStyle.HANDS_ON:
            recommendations.append("Look for practice exercises and coding challenges")
        elif self.developer_profile.learning_style == LearningStyle.VISUAL:
            recommendations.append("Create diagrams to visualize system architecture")
        
        return recommendations
    
    def _suggest_next_milestones(self) -> List[str]:
        """Suggest next learning milestones"""
        milestones = []
        
        concepts_count = len(self._extract_learned_concepts())
        if concepts_count < 5:
            milestones.append("Learn 5 core concepts")
        elif concepts_count < 10:
            milestones.append("Master 10 fundamental concepts")
        
        if self._calculate_total_learning_time() < 120:
            milestones.append("Complete 2 hours of focused learning")
        
        milestones.append("Complete your first code contribution")
        milestones.append("Successfully review someone else's code")
        
        return milestones[:3]  # Return top 3 milestones
    
    # Utility methods for text analysis
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract programming concepts from text"""
        # Simple keyword extraction - would use NLP in production
        concepts = []
        programming_terms = [
            "component", "function", "class", "method", "variable", "api", "endpoint",
            "database", "query", "migration", "model", "controller", "service", "middleware",
            "authentication", "authorization", "validation", "testing", "deployment"
        ]
        
        text_lower = text.lower()
        for term in programming_terms:
            if term in text_lower:
                concepts.append(term)
        
        return list(set(concepts))
    
    def _assess_response_complexity(self, response: str) -> str:
        """Assess the complexity level of a response"""
        # Simple heuristic - would use more sophisticated analysis in production
        word_count = len(response.split())
        code_blocks = response.count("```")
        technical_terms = len(self._extract_concepts_from_text(response))
        
        complexity_score = word_count / 100 + code_blocks * 2 + technical_terms
        
        if complexity_score > 10:
            return "high"
        elif complexity_score > 5:
            return "medium"
        else:
            return "low"
    
    def _identify_educational_elements(self, response: str) -> List[str]:
        """Identify educational elements in the response"""
        elements = []
        
        if "example" in response.lower():
            elements.append("examples")
        if "```" in response:
            elements.append("code_samples")
        if any(word in response.lower() for word in ["because", "since", "therefore", "this is why"]):
            elements.append("explanations")
        if any(word in response.lower() for word in ["step", "first", "next", "then"]):
            elements.append("step_by_step")
        if "?" in response:
            elements.append("questions")
        
        return elements