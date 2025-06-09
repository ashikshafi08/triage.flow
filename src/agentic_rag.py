"""
Agentic RAG Integration
Combines LocalRepoContextExtractor with AgenticCodebaseExplorer for enhanced context retrieval
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import re

from .new_rag import LocalRepoContextExtractor
from .agentic_tools import AgenticCodebaseExplorer
from .config import settings
from .issue_rag import IssueAwareRAG
from .patch_linkage import PatchLinkageBuilder

logger = logging.getLogger(__name__)

class AgenticRAGSystem:
    """
    Enhanced RAG system that combines semantic retrieval with agentic tool capabilities
    for more intelligent context extraction and query processing
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.rag_extractor = None
        self.agentic_explorer = None
        self.issue_rag = None  # Add issue-aware RAG
        self.repo_path = None
        self.repo_info = None
        self._query_cache = {}  # Cache for repeated queries

    async def initialize_core_systems(self, repo_url: str, branch: str = "main") -> None:
        """Initializes the core RAG (LocalRepoContextExtractor) and basic agentic explorer."""
        try:
            # Initialize RAG system (core code RAG)
            self.rag_extractor = LocalRepoContextExtractor()
            await self.rag_extractor.load_repository(repo_url, branch)
            
            self.repo_path = self.rag_extractor.current_repo_path
            self.repo_info = self.rag_extractor.repo_info
            
            # Initialize agentic explorer (can function with just code RAG initially)
            # Issue RAG will be linked later if successfully initialized.
            self.agentic_explorer = AgenticCodebaseExplorer(self.session_id, self.repo_path, issue_rag_system=None)
            
            # Initialize commit index for commit-level analysis
            try:
                logger.info(f"Initializing commit index for session {self.session_id}")
                # Force rebuild for new sessions to ensure fresh data
                await self.agentic_explorer.initialize_commit_index(force_rebuild=True)
                
                # File statistics will be built automatically during index creation
                
                # Verify initialization worked
                if hasattr(self.agentic_explorer, 'commit_index_manager'):
                    stats = self.agentic_explorer.commit_index_manager.get_statistics()
                    logger.info(f"Commit index initialized successfully for session {self.session_id}: {stats}")
                    
                    # Check if we got a reasonable number of commits
                    if stats.get('total_commits', 0) < 10:
                        logger.warning(f"Suspiciously low commit count ({stats.get('total_commits', 0)}) for session {self.session_id}")
                else:
                    logger.warning(f"Commit index manager not available for session {self.session_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize commit index for session {self.session_id}: {e}")
                import traceback
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                # Continue without commit index - other tools will still work
            
            logger.info(f"AgenticRAG core systems initialized for session {self.session_id}.")
            
        except Exception as e:
            logger.error(f"Failed to initialize AgenticRAG core systems: {e}")
            raise

    async def initialize_issue_rag_async(self, session: Dict[str, Any]) -> None:
        """Initializes the IssueAwareRAG system asynchronously and updates session status."""
        from .patch_linkage import ProgressUpdate
        
        if not self.repo_info:
            logger.error(f"Session {self.session_id}: Cannot initialize IssueAwareRAG: repo_info is missing.")
            session["metadata"]["status"] = "error_repo_info_missing"
            session["metadata"]["message"] = "Error: Repository information missing, cannot load issue context."
            return

        owner = self.repo_info.get("owner")
        repo_name = self.repo_info.get("repo") # Renamed to avoid conflict with 'repo' module

        if not (owner and repo_name):
            logger.error(f"Session {self.session_id}: Cannot initialize IssueAwareRAG: owner or repo name is missing from repo_info.")
            session["metadata"]["status"] = "error_repo_details_missing"
            session["metadata"]["message"] = "Error: Repository owner/name missing, cannot load issue context."
            return

        def progress_callback(update: ProgressUpdate):
            """Update session metadata with detailed progress information"""
            try:
                # Update session metadata with detailed progress
                session["metadata"]["status"] = "issue_linking"
                session["metadata"]["progress_stage"] = update.stage
                session["metadata"]["progress_step"] = update.current_step
                session["metadata"]["progress_percentage"] = update.progress_percentage
                session["metadata"]["progress_items_processed"] = update.items_processed
                session["metadata"]["progress_total_items"] = update.total_items
                session["metadata"]["progress_current_item"] = update.current_item
                session["metadata"]["progress_estimated_time"] = update.estimated_time_remaining
                session["metadata"]["progress_details"] = update.details
                
                # Create user-friendly message
                if update.current_item:
                    message = f"{update.current_step}: {update.current_item}"
                else:
                    message = f"{update.current_step} ({update.items_processed}/{update.total_items})"
                
                if update.estimated_time_remaining:
                    minutes = update.estimated_time_remaining // 60
                    seconds = update.estimated_time_remaining % 60
                    if minutes > 0:
                        message += f" - ~{minutes}m {seconds}s remaining"
                    else:
                        message += f" - ~{seconds}s remaining"
                
                session["metadata"]["message"] = message
                
                logger.info(f"Progress: {update.stage} - {update.current_step} ({update.progress_percentage:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

        try:
            logger.info(f"Session {self.session_id}: Starting IssueAwareRAG initialization for {owner}/{repo_name}.")
            session["metadata"]["status"] = "issue_linking" 
            session["metadata"]["message"] = f"Starting issue linking and indexing for {owner}/{repo_name}..."
            
            # Create IssueAwareRAG with progress callback
            self.issue_rag = IssueAwareRAG(owner, repo_name, progress_callback)
            await self.issue_rag.initialize(force_rebuild=False) 
            
            if self.agentic_explorer: 
                self.agentic_explorer.issue_rag_system = self.issue_rag
            
            logger.info(f"Session {self.session_id}: IssueAwareRAG for {owner}/{repo_name} initialized successfully.")
            session["metadata"]["issue_rag_ready"] = True
            session["metadata"]["status"] = "ready" 
            session["metadata"]["message"] = "All systems ready. Full repository context available."
            
            # Clear progress fields since we're done
            for key in ["progress_stage", "progress_step", "progress_percentage", "progress_items_processed", 
                       "progress_total_items", "progress_current_item", "progress_estimated_time", "progress_details"]:
                session["metadata"].pop(key, None)

        except Exception as e:
            logger.error(f"Session {self.session_id}: Failed to initialize IssueAwareRAG for {owner}/{repo_name}: {e}")
            self.issue_rag = None # Ensure it's None if init failed
            session["metadata"]["issue_rag_ready"] = False
            session["metadata"]["status"] = "warning_issue_rag_failed"
            session["metadata"]["message"] = f"Core chat is ready. Issue context for {owner}/{repo_name} failed to load: {str(e)}"
            session["metadata"]["error_details_issue_rag"] = str(e) # Store specific error
            
            # Clear progress fields on error
            for key in ["progress_stage", "progress_step", "progress_percentage", "progress_items_processed", 
                       "progress_total_items", "progress_current_item", "progress_estimated_time", "progress_details"]:
                session["metadata"].pop(key, None)

        # NOTE: The 'session' dict is modified here. The caller (SessionManager)
        # is responsible for persisting these changes if needed (e.g., to disk or notifying UI).
    
    async def get_enhanced_context(
        self, 
        query: str, 
        restrict_files: Optional[List[str]] = None,
        use_agentic_tools: bool = True,
        include_issue_context: bool = True
    ) -> Dict[str, Any]:
        """
        Get enhanced context using both RAG and agentic capabilities with issue awareness
        """
        if not self.rag_extractor or not self.agentic_explorer:
            raise ValueError("AgenticRAG not properly initialized")
        
        try:
            # Analyze query to determine best approach
            query_analysis = await self._analyze_query(query)
            
            # Get base RAG context
            base_context = await self.rag_extractor.get_relevant_context(query, restrict_files)
            
            # Get issue context if enabled and available
            if include_issue_context and self.issue_rag and self.issue_rag.is_initialized():
                try:
                    issue_context = await self.issue_rag.get_issue_context(query, max_issues=3)
                    if issue_context.related_issues:
                        base_context["related_issues"] = {
                            "issues": [
                                {
                                    "number": result.issue.id,
                                    "title": result.issue.title,
                                    "state": result.issue.state,
                                    "url": f"https://github.com/{self.repo_info.get('owner')}/{self.repo_info.get('repo')}/issues/{result.issue.id}",
                                    "similarity": result.similarity,
                                    "labels": result.issue.labels,
                                    "body_preview": result.issue.body[:150] + "..." if len(result.issue.body) > 150 else result.issue.body
                                }
                                for result in issue_context.related_issues
                            ],
                            "total_found": issue_context.total_found,
                            "processing_time": issue_context.processing_time
                        }
                        logger.info(f"Added {len(issue_context.related_issues)} related issues to context")
                except Exception as e:
                    logger.warning(f"Failed to get issue context: {e}")
            
            # Enhance with agentic tools if beneficial
            if use_agentic_tools and query_analysis["should_use_agentic"]:
                enhanced_context = await self._enhance_with_agentic_tools(
                    query, base_context, query_analysis
                )
                return enhanced_context
            else:
                # Just add query analysis to base context
                base_context["query_analysis"] = query_analysis
                return base_context
                
        except Exception as e:
            logger.error(f"Error in enhanced context retrieval: {e}")
            # Fallback to base RAG
            return await self.rag_extractor.get_relevant_context(query, restrict_files)
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal processing approach"""
        query_lower = query.lower()
        
        # Classify query type
        query_type = self._classify_query_type(query)
        
        # Determine if agentic tools would be beneficial
        should_use_agentic = self._should_use_agentic_approach(query)
        
        # Extract technical requirements
        tech_requirements = self._extract_technical_requirements(query)
        
        # Extract keywords using agentic tools
        keywords = self._extract_issue_keywords(query)
        
        # Detect file/code references
        code_references = self._extract_code_references(query)
        
        return {
            "query_type": query_type,
            "should_use_agentic": should_use_agentic,
            "technical_requirements": tech_requirements,
            "keywords": keywords,
            "code_references": code_references,
            "complexity_score": self._calculate_complexity_score(query),
            "processing_strategy": self._determine_processing_strategy(query_type, should_use_agentic)
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query for optimal processing"""
        query_lower = query.lower()
        
        # File/directory exploration
        if any(pattern in query_lower for pattern in [
            "explore", "what's in", "show me", "list files", "directory", "folder"
        ]):
            return "exploration"
        
        # Code analysis and explanation
        if any(pattern in query_lower for pattern in [
            "explain", "what does", "how does", "analyze", "understand", "describe"
        ]):
            return "analysis"
        
        # Implementation and examples
        if any(pattern in query_lower for pattern in [
            "implement", "create", "build", "example", "how to", "generate"
        ]):
            return "implementation"
        
        # Debugging and troubleshooting
        if any(pattern in query_lower for pattern in [
            "debug", "fix", "error", "issue", "problem", "bug", "troubleshoot"
        ]):
            return "debugging"
        
        # Architecture and structure
        if any(pattern in query_lower for pattern in [
            "architecture", "structure", "design", "pattern", "relationship", "dependency"
        ]):
            return "architecture"
        
        # Search and finding
        if any(pattern in query_lower for pattern in [
            "find", "search", "locate", "where", "which files"
        ]):
            return "search"
        
        return "general"
    
    def _should_use_agentic_approach(self, query: str) -> bool:
        """Determine if agentic tools would improve the response"""
        query_lower = query.lower()
        
        # Complex exploration queries benefit from agentic tools
        exploration_patterns = [
            "explore", "analyze", "find all", "search for", "trace", "follow",
            "investigate", "deep dive", "comprehensive", "detailed"
        ]
        
        # Multi-step reasoning queries
        multi_step_indicators = [
            "step by step", "walk through", "process", "flow", "sequence",
            "first", "then", "next", "how does"
        ]
        
        # Code generation and examples
        generation_patterns = [
            "implement", "create", "build", "generate", "example", "show me how"
        ]
        
        # Architecture and relationship queries
        relationship_patterns = [
            "relationship", "dependency", "connected", "related", "structure",
            "architecture", "design", "pattern"
        ]
        
        # Check patterns
        if any(pattern in query_lower for pattern in exploration_patterns):
            return True
        if any(pattern in query_lower for pattern in multi_step_indicators):
            return True
        if any(pattern in query_lower for pattern in generation_patterns):
            return True
        if any(pattern in query_lower for pattern in relationship_patterns):
            return True
        
        # Long, complex queries likely benefit from agentic approach
        if len(query.split()) > 15:
            return True
        
        return False
    
    def _extract_technical_requirements(self, query: str) -> Dict[str, List[str]]:
        """Extract technical requirements from query using agentic tools"""
        if self.agentic_explorer:
            # Use the existing method from agentic tools
            fake_issue = type('Issue', (), {
                'title': '',
                'body': query,
                'labels': []
            })()
            return self.agentic_explorer._extract_technical_requirements(fake_issue)
        return {}
    
    def _extract_issue_keywords(self, query: str) -> Dict[str, List[str]]:
        """Extract keywords using agentic tools"""
        if self.agentic_explorer:
            return self.agentic_explorer._extract_issue_keywords(query)
        return {"primary": [], "contextual": [], "file_patterns": [], "all_terms": []}
    
    def _extract_code_references(self, query: str) -> Dict[str, List[str]]:
        """Extract code references (files, functions, classes) from query"""
        code_refs = {
            "files": [],
            "functions": [],
            "classes": [],
            "variables": [],
            "patterns": []
        }
        
        # File references (@file, file.py, etc.)
        file_patterns = [
            r'@([\w\-/\\.]+)',  # @filename
            r'`([^`]+\.(py|js|jsx|ts|tsx|java|go|rs|rb|php|html|css|json|yaml|yml|md))`',  # `file.ext`
            r'\b([\w\-/]+\.(py|js|jsx|ts|tsx|java|go|rs|rb|php|html|css|json|yaml|yml|md))\b'  # file.ext
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    code_refs["files"].append(match[0])
                else:
                    code_refs["files"].append(match)
        
        # Function/method references
        function_patterns = [
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # function_name(
            r'`([a-zA-Z_][a-zA-Z0-9_]*)\(\)`',   # `function()`
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)',   # def function_name
            r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)'  # function function_name
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, query)
            code_refs["functions"].extend(matches)
        
        # Class references
        class_patterns = [
            r'\bclass\s+([A-Z][a-zA-Z0-9_]*)',  # class ClassName
            r'\b([A-Z][a-zA-Z0-9_]*)\s*\(',     # ClassName(
            r'`([A-Z][a-zA-Z0-9_]*)`'           # `ClassName`
        ]
        
        for pattern in class_patterns:
            matches = re.findall(pattern, query)
            code_refs["classes"].extend(matches)
        
        # Remove duplicates and clean up
        for key in code_refs:
            code_refs[key] = list(set([ref for ref in code_refs[key] if ref and len(ref) > 1]))
        
        return code_refs
    
    def _calculate_complexity_score(self, query: str) -> int:
        """Calculate query complexity score"""
        score = 0
        
        # Word count factor
        word_count = len(query.split())
        if word_count > 20:
            score += 3
        elif word_count > 10:
            score += 2
        else:
            score += 1
        
        # Question complexity
        question_words = ['how', 'why', 'what', 'where', 'when', 'which']
        score += sum(1 for word in question_words if word in query.lower())
        
        # Technical terms
        tech_indicators = ['implement', 'debug', 'analyze', 'architecture', 'pattern', 'design']
        score += sum(1 for term in tech_indicators if term in query.lower())
        
        # File/code references
        if '@' in query or any(ext in query for ext in ['.py', '.js', '.ts', '.java']):
            score += 2
        
        return min(10, score)  # Cap at 10
    
    def _determine_processing_strategy(self, query_type: str, should_use_agentic: bool) -> str:
        """Determine the best processing strategy"""
        if should_use_agentic:
            if query_type in ["exploration", "architecture", "debugging"]:
                return "agentic_deep"
            elif query_type in ["implementation", "analysis"]:
                return "agentic_focused"
            else:
                return "agentic_light"
        else:
            return "rag_only"
    
    async def _enhance_with_agentic_tools(
        self, 
        query: str, 
        base_context: Dict[str, Any], 
        query_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance RAG context using agentic tools"""
        try:
            enhanced_context = base_context.copy()
            enhanced_context["query_analysis"] = query_analysis
            
            processing_strategy = query_analysis["processing_strategy"]
            
            if processing_strategy == "agentic_deep":
                # Deep analysis with multiple tools
                enhanced_context = await self._deep_agentic_enhancement(query, enhanced_context, query_analysis)
            elif processing_strategy == "agentic_focused":
                # Focused enhancement for specific needs
                enhanced_context = await self._focused_agentic_enhancement(query, enhanced_context, query_analysis)
            elif processing_strategy == "agentic_light":
                # Light enhancement with minimal overhead
                enhanced_context = await self._light_agentic_enhancement(query, enhanced_context, query_analysis)
            
            return enhanced_context
            
        except Exception as e:
            logger.error(f"Error enhancing with agentic tools: {e}")
            return base_context
    
    async def _deep_agentic_enhancement(
        self, 
        query: str, 
        context: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep enhancement using multiple agentic tools"""
        
        # Analyze file structure if architecture query
        if analysis["query_type"] == "architecture":
            try:
                structure_analysis = self.agentic_explorer.analyze_file_structure("")
                context["structure_analysis"] = json.loads(structure_analysis)
            except Exception as e:
                logger.warning(f"Failed to get structure analysis: {e}")
        
        # Find related files for all sources
        if context.get("sources"):
            try:
                related_files_map = {}
                for source in context["sources"][:3]:  # Limit to top 3 to avoid overhead
                    related_files = self.agentic_explorer.find_related_files(source["file"])
                    related_files_map[source["file"]] = json.loads(related_files)
                context["related_files"] = related_files_map
            except Exception as e:
                logger.warning(f"Failed to get related files: {e}")
        
        # Semantic content search for additional context
        if analysis["keywords"]["primary"]:
            try:
                for keyword in analysis["keywords"]["primary"][:2]:  # Limit to 2 keywords
                    semantic_results = self.agentic_explorer.semantic_content_search(keyword)
                    if "semantic_context" not in context:
                        context["semantic_context"] = {}
                    context["semantic_context"][keyword] = json.loads(semantic_results)
            except Exception as e:
                logger.warning(f"Failed to get semantic context: {e}")
        
        return context
    
    async def _focused_agentic_enhancement(
        self, 
        query: str, 
        context: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Focused enhancement for implementation/analysis queries"""
        
        # For implementation queries, find examples and patterns
        if analysis["query_type"] == "implementation":
            try:
                # Generate code examples if sources are available
                if context.get("sources"):
                    source_files = [source["file"] for source in context["sources"][:3]]
                    code_example = self.agentic_explorer.generate_code_example(
                        query, source_files
                    )
                    context["code_examples"] = json.loads(code_example)
            except Exception as e:
                logger.warning(f"Failed to generate code examples: {e}")
        
        # For analysis queries, provide deeper file analysis
        elif analysis["query_type"] == "analysis":
            try:
                if context.get("sources"):
                    analyzed_files = {}
                    for source in context["sources"][:2]:  # Limit to 2 files
                        file_analysis = self.agentic_explorer.analyze_file_structure(source["file"])
                        analyzed_files[source["file"]] = json.loads(file_analysis)
                    context["file_analysis"] = analyzed_files
            except Exception as e:
                logger.warning(f"Failed to analyze files: {e}")
        
        return context
    
    async def _light_agentic_enhancement(
        self, 
        query: str, 
        context: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Light enhancement with minimal processing overhead"""
        
        # Just add some basic semantic search if keywords are available
        if analysis["keywords"]["primary"]:
            try:
                primary_keyword = analysis["keywords"]["primary"][0]
                semantic_results = self.agentic_explorer.semantic_content_search(primary_keyword)
                context["additional_context"] = json.loads(semantic_results)
            except Exception as e:
                logger.warning(f"Failed light semantic search: {e}")
        
        return context
    
    def get_repo_info(self) -> Optional[Dict[str, Any]]:
        """Get repository information"""
        return self.repo_info
    
    def get_repo_path(self) -> Optional[str]:
        """Get repository path"""
        return self.repo_path
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.agentic_explorer:
                self.agentic_explorer.reset_memory()
            
            # Clear caches
            self._query_cache.clear()
            
            logger.info(f"AgenticRAG cleaned up for session {self.session_id}")
            
        except Exception as e:
            logger.warning(f"Error during AgenticRAG cleanup: {e}")
