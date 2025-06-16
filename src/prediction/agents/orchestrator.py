"""
Prediction orchestrator that coordinates multiple agents for comprehensive analysis
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Import existing Triage.Flow components
from ...llm_client import LLMClient
from ...agent_tools.core import AgenticCodebaseExplorer
from ...config import settings

# Import prediction components
from ..data_models.prediction_models import PredictionReport, RiskFactor
from ..tools.data_collector import PredictiveDataCollector
from ..tools.pattern_detector import PatternDetectionEngine
from ..tools.semantic_analyzer import SemanticCodeAnalyzer
from ..tools.risk_scorer import HybridRiskScorer

# Note: Using existing AgenticRAG system instead of separate specialized agents

logger = logging.getLogger(__name__)

class PredictionOrchestrator:
    """Orchestrates multiple agents for comprehensive predictive analysis"""
    
    def __init__(self, repo_path: str, repo_owner: str, repo_name: str):
        self.repo_path = repo_path
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        
        # Initialize existing Triage.Flow components
        self.llm_client = LLMClient()
        self.agentic_explorer = AgenticCodebaseExplorer(
            session_id=f"prediction_{repo_owner}_{repo_name}",
            repo_path=repo_path,
            issue_rag_system=None
        )
        
        # Initialize prediction components
        self.data_collector = PredictiveDataCollector(repo_path, repo_owner, repo_name)
        self.pattern_detector = PatternDetectionEngine(self.data_collector)
        self.semantic_analyzer = SemanticCodeAnalyzer(repo_path, repo_owner, repo_name)
        self.risk_scorer = HybridRiskScorer(repo_path, repo_owner, repo_name)
        
        # Note: Using existing AgenticCodebaseExplorer for intelligent analysis
        # instead of separate specialized agents
        
        logger.info(f"Initialized PredictionOrchestrator for {repo_owner}/{repo_name}")
    
    async def generate_prediction_report(self, prediction_horizon_days: int = 14) -> PredictionReport:
        """Generate comprehensive prediction report using multi-agent analysis"""
        start_time = datetime.now()
        logger.info(f"Starting prediction analysis for {self.repo_owner}/{self.repo_name}")
        
        try:
            # Phase 1: Parallel data collection, pattern detection, and semantic analysis
            logger.info("Phase 1: Data collection, pattern detection, and semantic analysis")
            data_tasks = [
                self.data_collector.collect_code_metrics(),
                self.data_collector.collect_team_metrics(),
                self.data_collector.collect_deployment_metrics(),
                self.pattern_detector.detect_bug_patterns(),
                self.pattern_detector.detect_team_patterns(),
                self.semantic_analyzer.analyze_code_quality_risks()
            ]
            
            code_metrics, team_metrics, deployment_metrics, bug_patterns, team_patterns, semantic_risks = await asyncio.gather(*data_tasks)
            
            # Phase 2: Hybrid risk scoring
            logger.info("Phase 2: Hybrid risk scoring")
            scoring_results = await self.risk_scorer.score_risks(
                semantic_risks=semantic_risks,
                pattern_risks=[pattern.__dict__ if hasattr(pattern, '__dict__') else pattern for pattern in bug_patterns],
                repository_context={
                    'code_metrics': code_metrics,
                    'team_metrics': team_metrics,
                    'deployment_metrics': deployment_metrics
                }
            )
            
            # Phase 3: LLM-based analysis using existing agentic RAG
            logger.info("Phase 3: LLM-based analysis")
            analysis_results = await self._perform_llm_analysis(
                code_metrics, team_metrics, deployment_metrics, bug_patterns, team_patterns, semantic_risks, scoring_results
            )
            
            # Phase 4: Synthesis and report generation
            logger.info("Phase 4: Report synthesis")
            prediction_report = await self._synthesize_report(
                prediction_horizon_days=prediction_horizon_days,
                code_metrics=code_metrics,
                team_metrics=team_metrics,
                deployment_metrics=deployment_metrics,
                bug_patterns=bug_patterns,
                team_patterns=team_patterns,
                semantic_risks=semantic_risks,
                scoring_results=scoring_results,
                analysis_results=analysis_results,
                start_time=start_time
            )
            
            analysis_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Prediction analysis completed in {analysis_duration:.2f} seconds")
            
            return prediction_report
            
        except Exception as e:
            logger.error(f"Error during prediction analysis: {e}")
            raise
    
    async def _perform_llm_analysis(
        self, 
        code_metrics: Dict[str, Any],
        team_metrics: Dict[str, Any], 
        deployment_metrics: Dict[str, Any],
        bug_patterns: List,
        team_patterns: List,
        semantic_risks: List[RiskFactor],
        scoring_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform LLM-based analysis using existing agentic RAG system"""
        
        # Prepare analysis context
        analysis_context = f"""
        Repository: {self.repo_owner}/{self.repo_name}
        
        Code Metrics Summary:
        - Hotspot files: {len(code_metrics.get('hotspot_files', []))}
        - Test coverage estimate: {code_metrics.get('test_coverage_estimate', {}).get('estimated_coverage', 0):.2f}
        - Dependency health score: {code_metrics.get('dependency_health', {}).get('dependency_freshness_score', 0):.2f}
        
        Team Metrics Summary:
        - Active authors: {team_metrics.get('collaboration_patterns', {}).get('total_active_authors', 0)}
        - Knowledge distribution: {team_metrics.get('collaboration_patterns', {}).get('knowledge_distribution', 'unknown')}
        - Average review time: {team_metrics.get('review_patterns', {}).get('average_review_time', 0)} hours
        
        Deployment Metrics Summary:
        - Merge frequency: {deployment_metrics.get('merge_frequency', {}).get('merge_frequency', 0):.2f} per day
        - Rollback frequency: {deployment_metrics.get('rollback_indicators', {}).get('rollback_frequency', 0):.2f} per month
        
        Detected Patterns:
        - Bug patterns: {len(bug_patterns)}
        - Team patterns: {len(team_patterns)}
        - Semantic code risks: {len(semantic_risks)}
        
        Hybrid Risk Scoring Results:
        - Overall Risk Score: {scoring_results.get('final_scores', {}).get('overall_risk_score', 0):.2f}
        - Scoring Confidence: {scoring_results.get('final_scores', {}).get('overall_confidence', 0):.2f}
        - Risk Level: {scoring_results.get('insights', {}).get('risk_level_assessment', 'Unknown')}
        
        Top Semantic Risks:
        {self._format_semantic_risks(semantic_risks[:3])}
        """
        
        # Use existing agentic RAG for analysis
        analysis_prompt = f"""
        Based on the following repository analysis data, predict potential issues and provide risk assessment:
        
        {analysis_context}
        
        Please analyze:
        1. What are the top 3 most likely issues to occur in the next 2 weeks?
        2. What is the overall risk level (low/medium/high/critical)?
        3. What are the key risk factors?
        4. What immediate actions should be taken?
        5. What long-term strategies would help prevent issues?
        
        Provide specific, actionable insights based on the data patterns.
        """
        
        try:
            # Use the existing agentic explorer for analysis
            llm_response = await self.agentic_explorer.query(analysis_prompt)
            
            # Parse the LLM response (simplified parsing for now)
            return {
                'llm_analysis': llm_response,
                'predicted_issues': self._extract_predicted_issues(llm_response),
                'risk_factors': self._extract_risk_factors(llm_response),
                'immediate_actions': self._extract_immediate_actions(llm_response),
                'long_term_strategies': self._extract_long_term_strategies(llm_response),
                'overall_confidence': 0.7  # Would be calculated based on data quality
            }
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                'llm_analysis': f"Error in analysis: {str(e)}",
                'predicted_issues': [],
                'risk_factors': [],
                'immediate_actions': ["Review system health manually"],
                'long_term_strategies': ["Implement monitoring"],
                'overall_confidence': 0.3
            }
    
    def _extract_predicted_issues(self, llm_response: str) -> List[Dict[str, Any]]:
        """Extract predicted issues from LLM response"""
        # Simplified extraction - in production, would use more sophisticated parsing
        issues = []
        
        # Look for numbered lists or bullet points about issues
        lines = llm_response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['issue', 'problem', 'bug', 'failure']):
                if len(line.strip()) > 10:  # Avoid very short lines
                    issues.append({
                        'type': 'llm_predicted',
                        'description': line.strip(),
                        'probability': 0.6,  # Default probability
                        'impact': 0.5,  # Default impact
                        'source': 'llm_analysis'
                    })
        
        return issues[:5]  # Return top 5 issues
    
    def _extract_risk_factors(self, llm_response: str) -> List[Dict[str, Any]]:
        """Extract risk factors from LLM response"""
        risk_factors = []
        
        # Look for risk-related content
        lines = llm_response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['risk', 'concern', 'warning', 'critical']):
                if len(line.strip()) > 10:
                    severity = 'medium'
                    if 'critical' in line.lower() or 'high' in line.lower():
                        severity = 'high'
                    elif 'low' in line.lower():
                        severity = 'low'
                    
                    risk_factors.append({
                        'type': 'llm_identified',
                        'severity': severity,
                        'confidence': 0.6,
                        'description': line.strip(),
                        'affected_files': [],
                        'mitigation_actions': []
                    })
        
        return risk_factors[:5]  # Return top 5 risk factors
    
    def _extract_immediate_actions(self, llm_response: str) -> List[str]:
        """Extract immediate actions from LLM response"""
        actions = []
        
        lines = llm_response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['immediate', 'action', 'should', 'must', 'urgent']):
                if len(line.strip()) > 10:
                    actions.append(line.strip())
        
        return actions[:5]  # Return top 5 actions
    
    def _extract_long_term_strategies(self, llm_response: str) -> List[str]:
        """Extract long-term strategies from LLM response"""
        strategies = []
        
        lines = llm_response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['long-term', 'strategy', 'prevent', 'improve', 'implement']):
                if len(line.strip()) > 10:
                    strategies.append(line.strip())
        
        return strategies[:5]  # Return top 5 strategies
    
    def _format_semantic_risks(self, risks: List[RiskFactor]) -> str:
        """Format semantic risks for LLM analysis"""
        if not risks:
            return "No semantic risks detected"
        
        formatted = []
        for risk in risks:
            formatted.append(f"- {risk.severity.upper()}: {risk.description[:100]}...")
        
        return "\n".join(formatted)
    
    async def _synthesize_report(
        self,
        prediction_horizon_days: int,
        code_metrics: Dict[str, Any],
        team_metrics: Dict[str, Any],
        deployment_metrics: Dict[str, Any],
        bug_patterns: List,
        team_patterns: List,
        semantic_risks: List[RiskFactor],
        scoring_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        start_time: datetime
    ) -> PredictionReport:
        """Synthesize all analysis results into a comprehensive report"""
        
        # Combine predicted issues from patterns and LLM analysis
        predicted_issues = []
        
        # From pattern analysis
        for pattern in bug_patterns:
            predicted_issues.append({
                'type': 'pattern_based',
                'description': pattern.description,
                'probability': pattern.confidence,
                'impact': pattern.risk_score,
                'files_affected': pattern.files_affected,
                'source': 'pattern_detection'
            })
        
        # From LLM analysis
        predicted_issues.extend(analysis_results.get('predicted_issues', []))
        
        # Extract risk factors
        risk_factors = []
        
        # Add semantic risks first (highest priority)
        risk_factors.extend(semantic_risks)
        
        # Add LLM-identified risks
        for risk in analysis_results.get('risk_factors', []):
            risk_factors.append(RiskFactor(
                factor_type=risk.get('type', 'unknown'),
                severity=risk.get('severity', 'medium'),
                confidence=risk.get('confidence', 0.5),
                description=risk.get('description', ''),
                affected_files=risk.get('affected_files', []),
                mitigation_actions=risk.get('mitigation_actions', [])
            ))
        
        # Calculate overall confidence score (use hybrid scoring confidence)
        overall_confidence = scoring_results.get('final_scores', {}).get('overall_confidence', 
                                                analysis_results.get('overall_confidence', 0.5))
        
        # Extract prevention strategies
        immediate_actions = analysis_results.get('immediate_actions', [])
        long_term_strategies = analysis_results.get('long_term_strategies', [])
        
        # Data sources used
        data_sources = [
            'commit_index',
            'patch_linkage', 
            'agentic_explorer',
            'code_metrics',
            'team_metrics',
            'deployment_metrics'
        ]
        
        analysis_duration = (datetime.now() - start_time).total_seconds()
        
        return PredictionReport(
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
            analysis_timestamp=datetime.now(),
            prediction_horizon_days=prediction_horizon_days,
            predicted_issues=predicted_issues,
            risk_factors=risk_factors,
            confidence_score=overall_confidence,
            detected_patterns=bug_patterns,
            team_patterns=team_patterns,
            immediate_actions=immediate_actions,
            long_term_strategies=long_term_strategies,
            analysis_duration_seconds=analysis_duration,
            data_sources_used=data_sources
        )
    
    async def generate_prevention_strategies(self, context_description: str = None) -> Dict[str, Any]:
        """Generate targeted prevention strategies based on current analysis"""
        logger.info("Generating prevention strategies")
        
        # Get current patterns and risks
        bug_patterns = await self.pattern_detector.detect_bug_patterns()
        team_patterns = await self.pattern_detector.detect_team_patterns()
        
        # Create context for LLM
        context = f"""
        Repository: {self.repo_owner}/{self.repo_name}
        Context: {context_description or 'General prevention strategies'}
        
        Detected Bug Patterns:
        {[f"- {p.description}" for p in bug_patterns[:5]]}
        
        Detected Team Patterns:
        {[f"- {p.pattern_type}: {', '.join(p.team_members)}" for p in team_patterns[:5]]}
        """
        
        prevention_prompt = f"""
        Based on the following analysis, provide specific prevention strategies:
        
        {context}
        
        Please provide:
        1. Immediate actions (can be implemented within 1 week)
        2. Short-term strategies (1-4 weeks)
        3. Long-term strategies (1-6 months)
        4. Monitoring and alerting recommendations
        
        Focus on actionable, specific recommendations.
        """
        
        try:
            llm_response = await self.agentic_explorer.query(prevention_prompt)
            
            return {
                'immediate_actions': self._extract_immediate_actions(llm_response),
                'short_term_strategies': self._extract_short_term_strategies(llm_response),
                'long_term_strategies': self._extract_long_term_strategies(llm_response),
                'monitoring_recommendations': self._extract_monitoring_recommendations(llm_response),
                'full_analysis': llm_response
            }
            
        except Exception as e:
            logger.error(f"Error generating prevention strategies: {e}")
            return {
                'immediate_actions': ["Review current system health"],
                'short_term_strategies': ["Implement basic monitoring"],
                'long_term_strategies': ["Establish prevention processes"],
                'monitoring_recommendations': ["Set up basic alerts"],
                'error': str(e)
            }
    
    def _extract_short_term_strategies(self, llm_response: str) -> List[str]:
        """Extract short-term strategies from LLM response"""
        strategies = []
        
        lines = llm_response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['short-term', 'short term', 'weeks', '1-4']):
                if len(line.strip()) > 10:
                    strategies.append(line.strip())
        
        return strategies[:5]
    
    def _extract_monitoring_recommendations(self, llm_response: str) -> List[str]:
        """Extract monitoring recommendations from LLM response"""
        recommendations = []
        
        lines = llm_response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['monitor', 'alert', 'track', 'observe']):
                if len(line.strip()) > 10:
                    recommendations.append(line.strip())
        
        return recommendations[:5]
    
    async def get_dashboard_data(self, progress_callback=None) -> Dict[str, Any]:
        """Get data for the prediction dashboard using hybrid scoring"""
        logger.info("Generating dashboard data with hybrid scoring")
        
        try:
            if progress_callback:
                await progress_callback({
                    "type": "progress",
                    "phase": "initialization",
                    "message": "Starting dashboard analysis...",
                    "progress": 5
                })
            
            # Get base data for analysis
            if progress_callback:
                await progress_callback({
                    "type": "progress",
                    "phase": "data_collection",
                    "message": "Collecting repository metrics...",
                    "progress": 15
                })
            
            code_metrics = await self.data_collector.collect_code_metrics()
            team_metrics = await self.data_collector.collect_team_metrics()
            deployment_metrics = await self.data_collector.collect_deployment_metrics()
            
            if progress_callback:
                await progress_callback({
                    "type": "progress",
                    "phase": "semantic_analysis",
                    "message": "Analyzing code quality with AI...",
                    "progress": 35
                })
            
            # Get semantic risks and patterns for hybrid scoring
            semantic_risks = await self.semantic_analyzer.analyze_code_quality_risks(progress_callback=progress_callback)
            
            if progress_callback:
                await progress_callback({
                    "type": "progress",
                    "phase": "pattern_detection",
                    "message": "Detecting bug and team patterns...",
                    "progress": 55
                })
            
            bug_patterns = await self.pattern_detector.detect_bug_patterns()
            team_patterns = await self.pattern_detector.detect_team_patterns()
            
            if progress_callback:
                await progress_callback({
                    "type": "progress",
                    "phase": "risk_scoring",
                    "message": "Running hybrid AI + algorithmic risk scoring...",
                    "progress": 75
                })
            
            # CORE: Use hybrid scoring as the primary risk assessment
            scoring_results = await self.risk_scorer.score_risks(
                semantic_risks=semantic_risks,
                pattern_risks=[pattern.__dict__ if hasattr(pattern, '__dict__') else pattern for pattern in bug_patterns],
                repository_context={
                    'code_metrics': code_metrics,
                    'team_metrics': team_metrics,
                    'deployment_metrics': deployment_metrics
                }
            )
            
            logger.info(f"Hybrid scoring completed: Overall risk={scoring_results.get('final_scores', {}).get('overall_risk_score', 0):.3f}")
            
            if progress_callback:
                await progress_callback({
                    "type": "progress",
                    "phase": "synthesis",
                    "message": "Synthesizing dashboard data...",
                    "progress": 90
                })
            
            # Extract key metrics from hybrid scoring
            final_scores = scoring_results.get('final_scores', {})
            insights = scoring_results.get('insights', {})
            
            # Calculate summary statistics from hybrid scoring
            total_risks = len(semantic_risks) + len(bug_patterns)
            high_risk_count = len([r for r in semantic_risks if r.severity in ['high', 'critical']])
            critical_risk_count = len([r for r in semantic_risks if r.severity == 'critical'])
            
            # Use hybrid scoring confidence as primary confidence
            primary_confidence = final_scores.get('overall_confidence', 0.7)
            overall_risk_score = final_scores.get('overall_risk_score', 0.5)
            
            # Risk breakdown from semantic analysis
            risk_breakdown = {
                'critical': critical_risk_count,
                'high': high_risk_count - critical_risk_count,
                'medium': len([r for r in semantic_risks if r.severity == 'medium']),
                'low': len([r for r in semantic_risks if r.severity == 'low'])
            }
            
            # Pattern insights
            pattern_insights = {
                'bug_patterns_detected': len(bug_patterns),
                'team_patterns_detected': len(team_patterns),
                'most_common_pattern_type': self._get_most_common_pattern_type(bug_patterns),
                'confidence_score': primary_confidence
            }
            
            # Action items from hybrid scoring insights
            action_items = {
                'immediate': insights.get('recommendations', [])[:5],  # Top 5 from hybrid scoring
                'long_term': [
                    'Implement continuous code quality monitoring',
                    'Establish regular technical debt review cycles',
                    'Enhance automated testing coverage',
                    'Set up proactive risk alerting system'
                ][:5]
            }
            
            # Recent predictions (current analysis)
            recent_predictions = [{
                'timestamp': datetime.now().isoformat(),
                'confidence': primary_confidence,
                'risk_count': total_risks,
                'predicted_issues': len(insights.get('top_concerns', []))
            }]
            
            # Trend data based on risk level
            risk_level = insights.get('risk_level_assessment', 'MEDIUM')
            trend_data = {
                'risk_trend': 'increasing' if 'CRITICAL' in risk_level or 'HIGH' in risk_level else 'stable',
                'pattern_trend': 'increasing' if len(bug_patterns) > 5 else 'stable',
                'confidence_trend': 'increasing' if primary_confidence > 0.8 else 'stable'
            }
            
            # Extract risky files from code metrics
            risky_files = []
            try:
                if 'hotspot_files' in code_metrics:
                    # hotspot_files is a list of file paths, not a dict
                    hotspot_file_paths = code_metrics['hotspot_files']
                    file_statistics = code_metrics.get('file_statistics', {})
                    
                    for file_path in hotspot_file_paths:
                        if file_path in file_statistics:
                            stats = file_statistics[file_path]
                            risky_files.append({
                                'path': file_path,
                                'risk_score': min(stats.get('touch_frequency', 0) / 50.0, 1.0),  # Normalize to 0-1
                                'touch_count': stats.get('touch_frequency', 0),
                                'unique_authors': stats.get('unique_authors', 0),
                                'recent_changes': stats.get('recent_commits', 0),
                                'complexity_score': stats.get('complexity_trend', {}).get('complexity_score', 0.5),
                                'last_modified': '',  # Would need to get from git
                                'primary_author': 'Unknown',  # Would need to calculate from commits
                                'issues_linked': 0  # Would need to correlate with issues
                            })
                        else:
                            # Fallback for files without detailed stats
                            risky_files.append({
                                'path': file_path,
                                'risk_score': 0.5,
                                'touch_count': 0,
                                'unique_authors': 0,
                                'recent_changes': 0,
                                'complexity_score': 0.5,
                                'last_modified': '',
                                'primary_author': 'Unknown',
                                'issues_linked': 0
                            })
            except Exception as e:
                logger.error(f"Error extracting risky files: {e}")
                risky_files = []
            
            # Sort risky files by risk score
            risky_files.sort(key=lambda x: x['risk_score'], reverse=True)
            risky_files = risky_files[:10]  # Top 10 risky files
            
            # Extract team insights from team metrics
            team_insights = []
            try:
                if 'author_statistics' in team_metrics:
                    for author, stats in team_metrics['author_statistics'].items():
                        # Calculate risk contribution based on commit frequency and file diversity
                        commits = stats.get('commits', 0)
                        files_touched = stats.get('files_touched', 0)
                        risk_contribution = min((commits * files_touched) / 1000.0, 1.0)
                        
                        # Get velocity trend info
                        velocity_info = stats.get('velocity_trend', {})
                        velocity_trend = 'stable'
                        if velocity_info.get('recent_spike', False):
                            velocity_trend = 'increasing'
                        elif velocity_info.get('average_daily_commits', 0) < 0.1:
                            velocity_trend = 'decreasing'
                        
                        team_insights.append({
                            'name': author,
                            'commits': commits,
                            'files_touched': files_touched,
                            'risk_contribution': risk_contribution,
                            'velocity_trend': velocity_trend,
                            'knowledge_areas': [],  # Would need to infer from file types
                            'collaboration_score': team_metrics.get('collaboration_patterns', {}).get('collaboration_score', 0.5)
                        })
            except Exception as e:
                logger.error(f"Error extracting team insights: {e}")
                team_insights = []
            
            # Sort team insights by risk contribution
            team_insights.sort(key=lambda x: x['risk_contribution'], reverse=True)
            team_insights = team_insights[:10]  # Top 10 contributors
            
            # Convert predicted issues to frontend format
            predicted_issues = []
            for i, concern in enumerate(insights.get('top_concerns', [])):  # Top 5 concerns
                # top_concerns is a list of strings, not dictionaries
                if isinstance(concern, str):
                    predicted_issues.append({
                        'id': f'pred-{i+1:03d}',
                        'title': concern[:100],  # Use concern string as title
                        'description': concern,  # Use concern string as description
                        'severity': 'high' if any(keyword in concern.lower() for keyword in ['critical', 'security', 'urgent']) else 'medium',
                        'confidence': final_scores.get('overall_confidence', 0.7),
                        'affected_files': [],  # Would need to extract from semantic risks
                        'estimated_impact': f"Risk level: {insights.get('risk_level_assessment', 'MEDIUM')}",
                        'prevention_actions': [
                            'Review affected files for potential issues',
                            'Implement additional testing',
                            'Monitor system metrics closely',
                            'Consider refactoring high-risk areas'
                        ],
                        'timeline': 'Next 14 days'
                    })
                else:
                    # Fallback for unexpected format
                    predicted_issues.append({
                        'id': f'pred-{i+1:03d}',
                        'title': f'Risk Concern #{i+1}',
                        'description': str(concern),
                        'severity': 'medium',
                        'confidence': final_scores.get('overall_confidence', 0.7),
                        'affected_files': [],
                        'estimated_impact': f"Risk level: {insights.get('risk_level_assessment', 'MEDIUM')}",
                        'prevention_actions': [
                            'Review affected files for potential issues',
                            'Implement additional testing',
                            'Monitor system metrics closely',
                            'Consider refactoring high-risk areas'
                        ],
                        'timeline': 'Next 14 days'
                    })
            
            if progress_callback:
                await progress_callback({
                    "type": "complete",
                    "phase": "complete",
                    "message": f"Analysis complete! Found {total_risks} risks with {primary_confidence:.0%} confidence",
                    "progress": 100
                })
            
            return {
                'summary': {
                    'total_risks': total_risks,
                    'high_priority_risks': high_risk_count + critical_risk_count,
                    'confidence_score': primary_confidence,
                    'last_analysis': datetime.now().isoformat(),
                    'repository_health_score': max(0.0, 1.0 - (total_risks / 20.0)),  # Simple health calculation
                    'predicted_issues_count': len(predicted_issues)
                },
                'risk_breakdown': risk_breakdown,
                'pattern_insights': pattern_insights,
                'hybrid_scoring': scoring_results,  # Add hybrid scoring results
                'semantic_analysis': {
                    'total_risks': len(semantic_risks),
                    'risk_categories': self._categorize_semantic_risks(semantic_risks),
                    'severity_distribution': {
                        'critical': len([r for r in semantic_risks if r.severity == 'critical']),
                        'high': len([r for r in semantic_risks if r.severity == 'high']),
                        'medium': len([r for r in semantic_risks if r.severity == 'medium']),
                        'low': len([r for r in semantic_risks if r.severity == 'low'])
                    },
                    'top_risks': [
                        {
                            'type': risk.factor_type,
                            'severity': risk.severity,
                            'description': risk.description,
                            'files': risk.affected_files
                        }
                        for risk in sorted(semantic_risks, key=lambda r: (
                            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(r.severity, 1) * r.confidence
                        ), reverse=True)[:5]
                    ],
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'action_items': action_items,
                'recent_predictions': recent_predictions,
                'trend_data': trend_data,
                'risky_files': risky_files,
                'team_insights': team_insights,
                'predicted_issues': predicted_issues,
                'deployment_metrics': {
                    'success_rate': deployment_metrics.get('success_rate', 95.0),
                    'average_deployment_time': deployment_metrics.get('average_deployment_time', 15.0),
                    'rollback_frequency': deployment_metrics.get('rollback_frequency', 0.1),
                    'last_deployment': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {
                'error': str(e),
                'summary': {
                    'total_risks': 0, 
                    'high_priority_risks': 0, 
                    'confidence_score': 0.0,
                    'last_analysis': datetime.now().isoformat(),
                    'repository_health_score': 0.0,
                    'predicted_issues_count': 0
                },
                'risk_breakdown': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                'pattern_insights': {},
                'hybrid_scoring': {},
                'semantic_analysis': {
                    'total_risks': 0,
                    'risk_categories': [],
                    'severity_distribution': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                    'top_risks': [],
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'action_items': {'immediate': [], 'long_term': []},
                'recent_predictions': [],
                'trend_data': {},
                'risky_files': [],
                'team_insights': [],
                'predicted_issues': [],
                'deployment_metrics': {
                    'success_rate': 0.0,
                    'average_deployment_time': 0.0,
                    'rollback_frequency': 0.0,
                    'last_deployment': datetime.now().isoformat()
                }
            }
    
    def _get_most_common_pattern_type(self, patterns: List) -> str:
        """Get the most common pattern type from detected patterns"""
        if not patterns:
            return "none"
        
        pattern_types = [p.pattern_type for p in patterns]
        return max(set(pattern_types), key=pattern_types.count) if pattern_types else "none"
    
    def _categorize_semantic_risks(self, semantic_risks: List[RiskFactor]) -> Dict[str, int]:
        """Categorize semantic risks by type"""
        categories = {}
        for risk in semantic_risks:
            risk_type = risk.factor_type
            if risk_type not in categories:
                categories[risk_type] = 0
            categories[risk_type] += 1
        return categories 