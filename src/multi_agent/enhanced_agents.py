"""
Enhanced Multi-Agent System
Integrates AgenticRAGSystem (multi-RAG + agentic tools) with structured multi-agent orchestration
"""

import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser

from ..config import settings
from ..agentic_rag import AgenticRAGSystem
from ..agentic_tools import AgenticCodebaseExplorer
from .events import ResearchTask, CodeComponent, ImplementationStep
from .structured_outputs import (
    QueryAnalysis, ImplementationStrategy, MultiRAGResearchResult,
    RAGContext, RAGSource, IssueContext, AgenticAnalysis, 
    ToolExecutionResult, ValidationFeedback, EnhancedMultiAgentResult
)
from .validators import CodeValidator, SafetyGate

logger = logging.getLogger(__name__)


class EnhancedResearchAgent:
    """Enhanced research agent using AgenticRAGSystem for comprehensive context retrieval"""
    
    def __init__(self, session_id: str, agentic_rag: AgenticRAGSystem):
        self.session_id = session_id
        self.agentic_rag = agentic_rag
        
        # Get the agentic explorer from the RAG system
        self.agentic_explorer = agentic_rag.agentic_explorer
        
        # Create structured output parser for query analysis
        self.query_analyzer = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(QueryAnalysis),
            prompt_template_str=(
                "Analyze this software development query and return structured analysis.\n"
                "Query: {query}\n"
                "Return only the structured analysis as JSON."
            ),
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get LLM for structured outputs"""
        from llama_index.llms.openrouter import OpenRouter
        return OpenRouter(
            model="google/gemini-2.5-flash-preview-05-20",
            api_key=settings.openrouter_api_key,
            max_tokens=2000
        )
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query intent and complexity with structured output"""
        try:
            # Use LlamaIndex structured output
            result = await self.query_analyzer.acall(query=query)
            logger.info(f"[EnhancedResearch] Query analysis completed: {result.query_type}")
            return result
        except Exception as e:
            logger.warning(f"Structured query analysis failed: {e}")
            # Fallback to default
            return QueryAnalysis(
                query_type="general",
                scope="multiple_files", 
                technical_domains=["general"],
                estimated_complexity=5
            )
    
    async def perform_research(self, query: str, query_analysis: QueryAnalysis) -> MultiRAGResearchResult:
        """Perform comprehensive research using multi-RAG and agentic tools"""
        start_time = time.time()
        
        try:
            # Get enhanced context from AgenticRAGSystem
            rag_context_raw = await self.agentic_rag.get_enhanced_context(
                query=query,
                use_agentic_tools=True,
                include_issue_context=True
            )
            
            # Convert to structured format
            rag_context = self._convert_to_structured_rag_context(rag_context_raw)
            
            # Perform additional agentic analysis if high complexity
            agentic_analysis = None
            if query_analysis.estimated_complexity >= 7:
                agentic_analysis = await self._perform_agentic_analysis(query, query_analysis)
            
            # Combine insights
            combined_insights = self._combine_insights(rag_context, agentic_analysis)
            
            # Calculate research quality score
            research_quality = self._calculate_research_quality(rag_context, agentic_analysis, query_analysis)
            
            processing_time = time.time() - start_time
            logger.info(f"[EnhancedResearch] Research completed in {processing_time:.2f}s")
            
            return MultiRAGResearchResult(
                rag_context=rag_context,
                agentic_analysis=agentic_analysis,
                combined_insights=combined_insights,
                research_quality_score=research_quality
            )
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            # Return minimal research result
            return MultiRAGResearchResult(
                rag_context=RAGContext(
                    sources=[],
                    search_type="error",
                    complexity=query_analysis.estimated_complexity,
                    processing_time=time.time() - start_time
                ),
                combined_insights=[f"Research failed: {str(e)}"],
                research_quality_score=1.0
            )
    
    def _convert_to_structured_rag_context(self, raw_context: Dict[str, Any]) -> RAGContext:
        """Convert raw RAG context to structured format"""
        sources = []
        for source in raw_context.get("sources", []):
            sources.append(RAGSource(
                file=source.get("file", "unknown"),
                language=source.get("language", "unknown"),
                content=source.get("content", ""),
                description=source.get("description"),
                match_reasons=source.get("match_reasons", [])
            ))
        
        related_issues = []
        issue_data = raw_context.get("related_issues", {})
        for issue in issue_data.get("issues", []):
            related_issues.append(IssueContext(
                number=issue.get("number", 0),
                title=issue.get("title", ""),
                state=issue.get("state", "unknown"),
                url=issue.get("url", ""),
                similarity=issue.get("similarity", 0.0),
                labels=issue.get("labels", []),
                body_preview=issue.get("body_preview", "")
            ))
        
        return RAGContext(
            sources=sources,
            related_issues=related_issues,
            search_type=raw_context.get("search_type", "regular"),
            complexity=raw_context.get("complexity", 5),
            processing_time=raw_context.get("processing_time", 0.0),
            repo_info=raw_context.get("repo_info", {})
        )
    
    async def _perform_agentic_analysis(self, query: str, query_analysis: QueryAnalysis) -> AgenticAnalysis:
        """Perform deep analysis using agentic tools"""
        tools_used = []
        tool_results = []
        
        try:
            # Determine which tools to use based on query type
            if query_analysis.query_type == "bug_investigation":
                tools_to_run = ["find_related_files", "analyze_file_structure", "related_issues"]
            elif query_analysis.query_type == "feature_development":
                tools_to_run = ["semantic_content_search", "generate_code_example", "analyze_file_structure"]
            elif query_analysis.query_type == "architecture_decision":
                tools_to_run = ["analyze_file_structure", "find_related_files", "semantic_content_search"]
            else:
                tools_to_run = ["semantic_content_search", "find_related_files"]
            
            # Execute tools
            for tool_name in tools_to_run:
                tool_start = time.time()
                try:
                    if tool_name == "semantic_content_search":
                        result = self.agentic_explorer.semantic_content_search(query)
                    elif tool_name == "analyze_file_structure":
                        result = self.agentic_explorer.analyze_file_structure("")
                    elif tool_name == "find_related_files":
                        # Use first source file if available
                        first_file = "main.py"  # fallback
                        result = self.agentic_explorer.find_related_files(first_file)
                    elif tool_name == "related_issues":
                        result = self.agentic_explorer.related_issues(query, k=3)
                    elif tool_name == "generate_code_example":
                        result = self.agentic_explorer.generate_code_example(query, [])
                    else:
                        continue
                    
                    tools_used.append(tool_name)
                    tool_results.append(ToolExecutionResult(
                        tool_name=tool_name,
                        success=True,
                        result=result[:1000],  # Truncate for structured output
                        execution_time=time.time() - tool_start
                    ))
                    
                except Exception as e:
                    tool_results.append(ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        result="",
                        execution_time=time.time() - tool_start,
                        error=str(e)
                    ))
            
            # Extract insights from tool results
            insights = []
            recommendations = []
            
            for result in tool_results:
                if result.success and result.result:
                    try:
                        parsed = json.loads(result.result)
                        if isinstance(parsed, dict):
                            insights.extend(parsed.get("insights", [])[:2])  # Limit insights
                            recommendations.extend(parsed.get("recommendations", [])[:2])
                    except:
                        # Extract some basic insights from text
                        if len(result.result) > 50:
                            insights.append(f"Analysis from {result.tool_name}: {result.result[:100]}...")
            
            # Calculate confidence based on successful tool executions
            successful_tools = sum(1 for r in tool_results if r.success)
            confidence = min(0.9, successful_tools / len(tools_to_run))
            
            return AgenticAnalysis(
                tools_used=tools_used,
                tool_results=tool_results,
                insights=insights[:5],  # Limit to 5 insights
                recommendations=recommendations[:5],  # Limit to 5 recommendations
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Agentic analysis failed: {e}")
            return AgenticAnalysis(
                tools_used=[],
                tool_results=[],
                insights=[f"Agentic analysis failed: {str(e)}"],
                recommendations=[],
                confidence_score=0.1
            )
    
    def _combine_insights(self, rag_context: RAGContext, agentic_analysis: Optional[AgenticAnalysis]) -> List[str]:
        """Combine insights from RAG and agentic analysis"""
        insights = []
        
        # Insights from RAG sources
        if rag_context.sources:
            insights.append(f"Found {len(rag_context.sources)} relevant source files")
            
        # Insights from related issues
        if rag_context.related_issues:
            insights.append(f"Found {len(rag_context.related_issues)} related GitHub issues")
            open_issues = [i for i in rag_context.related_issues if i.state == "open"]
            if open_issues:
                insights.append(f"{len(open_issues)} related issues are still open")
        
        # Insights from agentic analysis
        if agentic_analysis and agentic_analysis.insights:
            insights.extend(agentic_analysis.insights[:3])  # Add top 3 agentic insights
        
        return insights[:7]  # Limit total insights
    
    def _calculate_research_quality(self, rag_context: RAGContext, agentic_analysis: Optional[AgenticAnalysis], query_analysis: QueryAnalysis) -> float:
        """Calculate overall research quality score"""
        quality = 0.0
        
        # Base score from RAG sources
        if rag_context.sources:
            quality += min(5.0, len(rag_context.sources))  # Up to 5 points for sources
        
        # Points for issue context
        if rag_context.related_issues:
            quality += min(2.0, len(rag_context.related_issues) * 0.5)  # Up to 2 points for issues
        
        # Points for agentic analysis
        if agentic_analysis:
            quality += agentic_analysis.confidence_score * 2.0  # Up to 2 points
            if agentic_analysis.insights:
                quality += min(1.0, len(agentic_analysis.insights) * 0.2)  # Up to 1 point
        
        return min(10.0, quality)


class EnhancedImplementationAgent:
    """Enhanced implementation agent using agentic tools for code generation"""
    
    def __init__(self, session_id: str, agentic_explorer: AgenticCodebaseExplorer):
        self.session_id = session_id
        self.agentic_explorer = agentic_explorer
        
        # Create structured output parser
        self.strategy_planner = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(ImplementationStrategy),
            prompt_template_str=(
                "Based on the research findings, create an implementation strategy.\n"
                "Query: {query}\n"
                "Research Summary: {research_summary}\n"
                "Return structured implementation strategy as JSON."
            ),
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get LLM for structured outputs"""
        from llama_index.llms.openrouter import OpenRouter
        return OpenRouter(
            model="google/gemini-2.5-flash-preview-05-20",
            api_key=settings.openrouter_api_key,
            max_tokens=2000
        )
    
    async def create_implementation_strategy(self, query: str, research_results: MultiRAGResearchResult) -> ImplementationStrategy:
        """Create structured implementation strategy"""
        try:
            # Prepare research summary
            research_summary = {
                "sources_found": len(research_results.rag_context.sources),
                "related_issues": len(research_results.rag_context.related_issues),
                "insights": research_results.combined_insights,
                "quality_score": research_results.research_quality_score
            }
            
            result = await self.strategy_planner.acall(
                query=query,
                research_summary=json.dumps(research_summary, indent=2)
            )
            
            logger.info(f"[EnhancedImplementation] Strategy created: {result.high_level_approach}")
            return result
            
        except Exception as e:
            logger.warning(f"Structured strategy planning failed: {e}")
            # Fallback strategy
            return ImplementationStrategy(
                high_level_approach="Implement solution based on research findings",
                technology_choices=["python"],
                file_organization=["main.py"],
                integration_points=[],
                testing_strategy="unit_tests"
            )
    
    async def generate_implementation(self, query: str, strategy: ImplementationStrategy, research_results: MultiRAGResearchResult) -> List[CodeComponent]:
        """Generate code implementation using agentic tools"""
        try:
            # Use the powerful write_complete_code tool
            context_files = [source.file for source in research_results.rag_context.sources[:5]]
            
            # Determine primary language from strategy
            primary_language = strategy.technology_choices[0] if strategy.technology_choices else "python"
            
            # Generate complete code
            code_result = self.agentic_explorer.write_complete_code(
                description=f"Implementation for: {query}",
                context_files=context_files if context_files else None,
                language=primary_language,
                output_format="raw"
            )
            
            # Create code components
            components = []
            for file_name in strategy.file_organization:
                components.append(CodeComponent(
                    name=file_name,
                    language=primary_language,
                    code=code_result[:2000],  # Truncate for structured output
                    description=f"Implementation for {query}",
                    dependencies=strategy.additional_components
                ))
            
            logger.info(f"[EnhancedImplementation] Generated {len(components)} code components")
            return components
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            # Return minimal implementation
            return [CodeComponent(
                name="implementation.py",
                language="python",
                code=f"# Implementation for: {query}\n# Error: {str(e)}",
                description=f"Failed implementation for {query}",
                dependencies=[]
            )]


class EnhancedWorkflowOrchestrator:
    """Enhanced workflow orchestrator that combines everything"""
    
    def __init__(self, session_id: str, agentic_rag: AgenticRAGSystem):
        self.session_id = session_id
        self.agentic_rag = agentic_rag
        
        # Initialize enhanced agents
        self.research_agent = EnhancedResearchAgent(session_id, agentic_rag)
        self.implementation_agent = EnhancedImplementationAgent(session_id, agentic_rag.agentic_explorer)
        
        # Keep existing validators
        self.code_validator = CodeValidator()
        self.safety_gate = SafetyGate()
    
    async def execute_enhanced_workflow(self, query: str) -> EnhancedMultiAgentResult:
        """Execute the complete enhanced multi-agent workflow"""
        start_time = time.time()
        performance_metrics = {}
        
        try:
            # Step 1: Analyze Query
            step_start = time.time()
            query_analysis = await self.research_agent.analyze_query(query)
            performance_metrics["query_analysis"] = time.time() - step_start
            
            # Step 2: Perform Research
            step_start = time.time() 
            research_results = await self.research_agent.perform_research(query, query_analysis)
            performance_metrics["research"] = time.time() - step_start
            
            # Step 3: Create Implementation Strategy
            step_start = time.time()
            implementation_strategy = await self.implementation_agent.create_implementation_strategy(
                query, research_results
            )
            performance_metrics["strategy"] = time.time() - step_start
            
            # Step 4: Generate Code
            step_start = time.time()
            code_components = await self.implementation_agent.generate_implementation(
                query, implementation_strategy, research_results
            )
            performance_metrics["implementation"] = time.time() - step_start
            
            # Step 5: Validate and Assess Risk
            step_start = time.time()
            validation_feedback = await self._validate_implementation(code_components, query_analysis)
            performance_metrics["validation"] = time.time() - step_start
            
            total_time = time.time() - start_time
            
            # Determine approval status
            approved = validation_feedback.overall_status == "approved"
            
            # Generate next steps
            next_steps = self._generate_next_steps(validation_feedback, research_results, approved)
            
            # Track systems used
            rag_systems_used = ["LocalRepoContextExtractor"]
            if research_results.rag_context.related_issues:
                rag_systems_used.append("IssueAwareRAG")
            if research_results.agentic_analysis:
                rag_systems_used.append("AgenticCodebaseExplorer")
            
            tools_executed = []
            if research_results.agentic_analysis:
                tools_executed = research_results.agentic_analysis.tools_used
            
            return EnhancedMultiAgentResult(
                query=query,
                query_analysis=query_analysis,
                research_results=research_results,
                implementation_strategy=implementation_strategy,
                validation_feedback=validation_feedback,
                performance_metrics=performance_metrics,
                session_id=self.session_id,
                total_execution_time=total_time,
                approved=approved,
                next_steps=next_steps,
                rag_systems_used=rag_systems_used,
                tools_executed=tools_executed
            )
            
        except Exception as e:
            logger.error(f"Enhanced workflow failed: {e}")
            # Return error result
            return self._create_error_result(query, str(e), time.time() - start_time)
    
    async def _validate_implementation(self, code_components: List[CodeComponent], query_analysis: QueryAnalysis) -> ValidationFeedback:
        """Validate implementation and provide feedback"""
        start_time = time.time()
        
        try:
            feedback_items = []
            required_changes = []
            code_quality_score = 5.0
            security_score = 5.0
            
            if code_components:
                # Validate first component
                main_component = code_components[0]
                validation_result = self.code_validator.validate_code(
                    main_component.code, 
                    main_component.language
                )
                
                if validation_result.valid:
                    code_quality_score = 7.0
                    feedback_items.append("Code validation passed")
                else:
                    code_quality_score = 3.0
                    feedback_items.extend([f"Validation error: {err}" for err in validation_result.errors[:3]])
                    required_changes.extend(validation_result.errors[:3])
                
                # Safety assessment
                risk_assessment = self.safety_gate.assess_risk({
                    "original_query": query_analysis.query_type,
                    "main_implementation": main_component.code,
                    "implementation_steps": []
                })
                
                if risk_assessment.requires_approval:
                    security_score = 3.0
                    feedback_items.append(f"Security concern: {risk_assessment.approval_reason}")
                    required_changes.append("Address security risks")
                else:
                    security_score = 8.0
            
            # Determine overall status
            if required_changes:
                overall_status = "requires_review"
            elif code_quality_score >= 7.0 and security_score >= 7.0:
                overall_status = "approved"
            else:
                overall_status = "requires_review"
            
            return ValidationFeedback(
                overall_status=overall_status,
                code_quality_score=code_quality_score,
                security_score=security_score,
                feedback_items=feedback_items,
                required_changes=required_changes,
                optional_improvements=[],
                validation_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationFeedback(
                overall_status="requires_review",
                feedback_items=[f"Validation error: {str(e)}"],
                required_changes=["Fix validation errors"],
                validation_time_seconds=time.time() - start_time
            )
    
    def _generate_next_steps(self, validation_feedback: ValidationFeedback, research_results: MultiRAGResearchResult, approved: bool) -> List[str]:
        """Generate contextual next steps"""
        next_steps = []
        
        if approved:
            next_steps.append("Implementation approved and ready for deployment")
            next_steps.append("Consider running integration tests")
        else:
            if validation_feedback.required_changes:
                next_steps.extend([f"Fix: {change}" for change in validation_feedback.required_changes[:3]])
            
            next_steps.append("Review generated code thoroughly")
            next_steps.append("Test implementation in development environment")
        
        # Add research-based suggestions
        if research_results.rag_context.related_issues:
            open_issues = [i for i in research_results.rag_context.related_issues if i.state == "open"]
            if open_issues:
                next_steps.append(f"Check related open issues: {len(open_issues)} found")
        
        return next_steps[:5]  # Limit to 5 next steps
    
    def _create_error_result(self, query: str, error: str, execution_time: float) -> EnhancedMultiAgentResult:
        """Create error result when workflow fails"""
        return EnhancedMultiAgentResult(
            query=query,
            query_analysis=QueryAnalysis(),  # Default analysis
            research_results=MultiRAGResearchResult(
                rag_context=RAGContext(sources=[], search_type="error", processing_time=0.0),
                combined_insights=[f"Workflow error: {error}"],
                research_quality_score=0.0
            ),
            implementation_strategy=ImplementationStrategy(),  # Default strategy
            validation_feedback=ValidationFeedback(
                overall_status="rejected",
                feedback_items=[f"Workflow failed: {error}"],
                required_changes=["Fix workflow errors"],
                validation_time_seconds=0.0
            ),
            performance_metrics={"error": execution_time},
            session_id=self.session_id,
            total_execution_time=execution_time,
            approved=False,
            next_steps=["Debug workflow error", "Retry with simpler query"],
            rag_systems_used=[],
            tools_executed=[]
        ) 