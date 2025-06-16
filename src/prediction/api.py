"""
API endpoints for the prediction system
"""

import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException

from .agents.orchestrator import PredictionOrchestrator
from .tools.semantic_analyzer import SemanticCodeAnalyzer
from .data_models.prediction_models import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

class PredictionAPI:
    """API interface for the prediction system"""
    
    def __init__(self):
        self.orchestrators = {}  # Cache orchestrators by repo
        self.semantic_analyzers = {}  # Cache semantic analyzers by repo
    
    def _get_orchestrator(self, repo_path: str, repo_owner: str, repo_name: str) -> PredictionOrchestrator:
        """Get or create orchestrator for repository"""
        key = f"{repo_owner}/{repo_name}"
        
        if key not in self.orchestrators:
            self.orchestrators[key] = PredictionOrchestrator(repo_path, repo_owner, repo_name)
        
        return self.orchestrators[key]
    
    def _get_semantic_analyzer(self, repo_path: str, repo_owner: str, repo_name: str) -> SemanticCodeAnalyzer:
        """Get or create semantic analyzer for repository"""
        key = f"{repo_owner}/{repo_name}"
        
        if key not in self.semantic_analyzers:
            self.semantic_analyzers[key] = SemanticCodeAnalyzer(repo_path, repo_owner, repo_name)
        
        return self.semantic_analyzers[key]
    
    async def generate_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """Generate prediction report for repository"""
        try:
            logger.info(f"Generating prediction for {request.repo_owner}/{request.repo_name}")
            
            orchestrator = self._get_orchestrator(
                request.repo_path or f"./{request.repo_name}",
                request.repo_owner,
                request.repo_name
            )
            
            prediction_report = await orchestrator.generate_prediction_report(
                prediction_horizon_days=request.prediction_horizon_days
            )
            
            return PredictionResponse(
                status="success",
                prediction_report=prediction_report.dict(),
                analysis_duration_seconds=prediction_report.analysis_duration_seconds
            )
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return PredictionResponse(
                status="error",
                error=str(e)
            )
    
    async def analyze_semantic_risks(
        self, 
        repo_path: str, 
        repo_owner: str, 
        repo_name: str,
        file_paths: Optional[list] = None
    ) -> Dict[str, Any]:
        """Analyze semantic code quality risks"""
        try:
            logger.info(f"Analyzing semantic risks for {repo_owner}/{repo_name}")
            
            analyzer = self._get_semantic_analyzer(repo_path, repo_owner, repo_name)
            
            # Get detailed risk analysis
            risks = await analyzer.analyze_code_quality_risks(file_paths)
            
            # Get summary
            summary = await analyzer.get_risk_summary()
            
            return {
                'status': 'success',
                'summary': summary,
                'detailed_risks': [
                    {
                        'type': risk.factor_type,
                        'severity': risk.severity,
                        'confidence': risk.confidence,
                        'description': risk.description,
                        'affected_files': risk.affected_files,
                        'mitigation_actions': risk.mitigation_actions
                    }
                    for risk in risks
                ],
                'total_risks': len(risks)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing semantic risks: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'total_risks': 0
            }
    
    async def get_dashboard_data(
        self, 
        repo_path: str, 
        repo_owner: str, 
        repo_name: str
    ) -> Dict[str, Any]:
        """Get dashboard data with semantic analysis and hybrid scoring"""
        try:
            logger.info(f"Getting dashboard data for {repo_owner}/{repo_name}")
            
            orchestrator = self._get_orchestrator(repo_path, repo_owner, repo_name)
            
            # Get standard dashboard data
            dashboard_data = await orchestrator.get_dashboard_data()
            
            # Add semantic analysis
            semantic_analyzer = self._get_semantic_analyzer(repo_path, repo_owner, repo_name)
            semantic_summary = await semantic_analyzer.get_risk_summary()
            
            # Get hybrid scoring results by running a quick analysis
            try:
                # Get semantic risks for scoring
                semantic_risks = await semantic_analyzer.analyze_code_quality_risks()
                
                # Run hybrid scoring
                scoring_results = await orchestrator.risk_scorer.score_risks(
                    semantic_risks=semantic_risks,
                    pattern_risks=[],  # Would be populated from pattern detection
                    repository_context={}  # Would include metrics
                )
                
                # Add hybrid scoring to dashboard
                dashboard_data['hybrid_scoring'] = scoring_results
                
            except Exception as scoring_error:
                logger.warning(f"Could not get hybrid scoring results: {scoring_error}")
                # Continue without hybrid scoring data
            
            # Enhance dashboard with semantic data
            dashboard_data['semantic_analysis'] = semantic_summary
            
            # Update risk breakdown to include semantic risks
            if 'risk_breakdown' in dashboard_data:
                semantic_severity = semantic_summary.get('severity_distribution', {})
                for severity, count in semantic_severity.items():
                    if severity in dashboard_data['risk_breakdown']:
                        dashboard_data['risk_breakdown'][severity] += count
                    else:
                        dashboard_data['risk_breakdown'][severity] = count
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

# Global API instance
prediction_api = PredictionAPI() 