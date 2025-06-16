"""
Hybrid risk scoring system combining fixed metrics with LLM-based judgment
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from ...llm_client import LLMClient
from ...agent_tools.core import AgenticCodebaseExplorer
from ..data_models.prediction_models import RiskFactor

logger = logging.getLogger(__name__)

class HybridRiskScorer:
    """
    Combines multiple scoring approaches for accurate risk assessment:
    1. Fixed algorithmic scoring for objective metrics
    2. LLM-based judgment for subjective/contextual assessment
    3. Weighted combination based on confidence levels
    """
    
    def __init__(self, repo_path: str, repo_owner: str, repo_name: str):
        self.repo_path = repo_path
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        
        self.llm_client = LLMClient()
        self.agentic_explorer = AgenticCodebaseExplorer(
            session_id=f"risk_scoring_{repo_owner}_{repo_name}",
            repo_path=repo_path,
            issue_rag_system=None
        )
        
        # INTUITIVE SCORING FORMULA:
        # 1. Semantic Analysis (70% weight) - Direct code quality issues
        # 2. Pattern Detection (30% weight) - Repository-level patterns  
        # 3. LLM Judge (Final arbitrator) - Contextual assessment
        
        # Semantic risk weights (high impact on actual code quality)
        self.semantic_risk_weights = {
            'security_vulnerability': 1.0,      # Critical - immediate security risk
            'performance_issue': 0.9,           # High - affects user experience
            'error_handling': 0.8,              # High - affects reliability
            'code_complexity': 0.7,             # Medium-high - affects maintainability
            'maintainability_issue': 0.6,       # Medium - long-term technical debt
            'code_quality': 0.5                 # Medium - general quality issues
        }
        
        # Pattern risk weights (lower impact, more about process/trends)
        self.pattern_risk_weights = {
            'file_hotspot': 0.4,                # Medium-low - indicates change frequency
            'knowledge_concentration': 0.5,      # Medium - bus factor risk
            'velocity_spike': 0.3,              # Low - might indicate rushed work
            'review_rush': 0.4,                 # Medium-low - process issue
            'dependency_health': 0.6,           # Medium - external risk
            'test_coverage_drop': 0.7           # Medium-high - quality assurance risk
        }
        
        # Severity multipliers (exponential scale for intuitive impact)
        self.severity_multipliers = {
            'critical': 1.0,    # 100% impact
            'high': 0.75,       # 75% impact  
            'medium': 0.5,      # 50% impact
            'low': 0.25         # 25% impact
        }
        
        # Component weights in final score
        self.component_weights = {
            'semantic_analysis': 0.7,   # 70% - Direct code issues
            'pattern_detection': 0.3,   # 30% - Repository patterns
            'llm_adjustment': 0.2       # 20% adjustment factor (can boost or reduce)
        }
        
        logger.info(f"Initialized HybridRiskScorer with intuitive formula for {repo_owner}/{repo_name}")
    
    async def score_risks(
        self, 
        semantic_risks: List[RiskFactor], 
        pattern_risks: List[Dict[str, Any]],
        repository_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Score risks using hybrid approach
        
        Args:
            semantic_risks: Risks from semantic analysis
            pattern_risks: Risks from pattern detection
            repository_context: Additional context about the repository
            
        Returns:
            Comprehensive risk scoring results
        """
        logger.info("Starting hybrid risk scoring")
        
        try:
            # Phase 1: Fixed algorithmic scoring
            algorithmic_scores = await self._calculate_algorithmic_scores(
                semantic_risks, pattern_risks, repository_context
            )
            
            # Phase 2: LLM-based contextual judgment
            llm_scores = await self._get_llm_risk_judgment(
                semantic_risks, pattern_risks, repository_context, algorithmic_scores
            )
            
            # Phase 3: Hybrid combination
            final_scores = await self._combine_scores(
                algorithmic_scores, llm_scores, semantic_risks, pattern_risks
            )
            
            # Phase 4: Generate insights and recommendations
            insights = await self._generate_scoring_insights(final_scores, semantic_risks, pattern_risks)
            
            return {
                'final_scores': final_scores,
                'algorithmic_scores': algorithmic_scores,
                'llm_scores': llm_scores,
                'insights': insights,
                'scoring_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_risks_analyzed': len(semantic_risks) + len(pattern_risks),
                    'scoring_confidence': final_scores.get('overall_confidence', 0.7)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid risk scoring: {e}")
            return {
                'error': str(e),
                'final_scores': {'overall_risk_score': 0.5, 'overall_confidence': 0.3}
            }
    
    async def _calculate_algorithmic_scores(
        self, 
        semantic_risks: List[RiskFactor], 
        pattern_risks: List[Dict[str, Any]],
        repository_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate objective, algorithmic risk scores using intuitive formula"""
        
        scores = {
            'semantic_component': {'score': 0.0, 'confidence': 0.9, 'risk_count': len(semantic_risks)},
            'pattern_component': {'score': 0.0, 'confidence': 0.8, 'risk_count': len(pattern_risks)},
            'risk_category_scores': {},
            'severity_distribution': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'file_risk_scores': {},
            'overall_risk_score': 0.0,
            'algorithmic_confidence': 0.85
        }
        
        # STEP 1: Calculate Semantic Analysis Score (70% weight)
        semantic_score = 0.0
        semantic_total_weight = 0.0
        
        for risk in semantic_risks:
            risk_type = risk.factor_type
            severity = risk.severity
            confidence = risk.confidence
            
            # Intuitive formula: Weight × Severity × Confidence
            base_weight = self.semantic_risk_weights.get(risk_type, 0.5)
            severity_multiplier = self.severity_multipliers.get(severity, 0.5)
            risk_impact = base_weight * severity_multiplier * confidence
            
            semantic_score += risk_impact
            semantic_total_weight += base_weight
            
            # Track for detailed analysis
            if risk_type not in scores['risk_category_scores']:
                scores['risk_category_scores'][risk_type] = []
            scores['risk_category_scores'][risk_type].append(risk_impact)
            
            # Update severity distribution
            scores['severity_distribution'][severity] += 1
            
            # Track file-level risks
            for file_path in risk.affected_files:
                if file_path not in scores['file_risk_scores']:
                    scores['file_risk_scores'][file_path] = []
                scores['file_risk_scores'][file_path].append(risk_impact)
        
        # Normalize semantic score (0-1 scale)
        if semantic_total_weight > 0:
            scores['semantic_component']['score'] = min(semantic_score / semantic_total_weight, 1.0)
        
        # STEP 2: Calculate Pattern Detection Score (30% weight)
        pattern_score = 0.0
        pattern_total_weight = 0.0
        
        for pattern in pattern_risks:
            pattern_type = pattern.get('pattern_type', 'unknown')
            risk_score = pattern.get('risk_score', 0.5)
            confidence = pattern.get('confidence', 0.7)
            
            # Lower weight for patterns (they're indicators, not direct issues)
            base_weight = self.pattern_risk_weights.get(pattern_type, 0.3)
            pattern_impact = base_weight * risk_score * confidence
            
            pattern_score += pattern_impact
            pattern_total_weight += base_weight
            
            # Track for analysis
            mapped_type = self._map_pattern_to_risk_type(pattern_type)
            if mapped_type not in scores['risk_category_scores']:
                scores['risk_category_scores'][mapped_type] = []
            scores['risk_category_scores'][mapped_type].append(pattern_impact)
        
        # Normalize pattern score (0-1 scale)
        if pattern_total_weight > 0:
            scores['pattern_component']['score'] = min(pattern_score / pattern_total_weight, 1.0)
        
        # STEP 3: Combine using component weights
        semantic_weight = self.component_weights['semantic_analysis']
        pattern_weight = self.component_weights['pattern_detection']
        
        combined_score = (
            scores['semantic_component']['score'] * semantic_weight +
            scores['pattern_component']['score'] * pattern_weight
        )
        
        scores['overall_risk_score'] = min(combined_score, 1.0)
        
        # Calculate category averages
        for category, category_scores in scores['risk_category_scores'].items():
            if category_scores:
                scores['risk_category_scores'][category] = {
                    'average_score': sum(category_scores) / len(category_scores),
                    'max_score': max(category_scores),
                    'count': len(category_scores),
                    'total_impact': sum(category_scores)
                }
        
        # Calculate file risk scores
        for file_path, file_scores in scores['file_risk_scores'].items():
            if file_scores:
                scores['file_risk_scores'][file_path] = {
                    'average_score': sum(file_scores) / len(file_scores),
                    'max_score': max(file_scores),
                    'risk_count': len(file_scores),
                    'total_impact': sum(file_scores)
                }
        
        logger.info(f"Algorithmic scoring: Semantic={scores['semantic_component']['score']:.3f}, "
                   f"Pattern={scores['pattern_component']['score']:.3f}, "
                   f"Combined={scores['overall_risk_score']:.3f}")
        
        return scores
    
    async def _get_llm_risk_judgment(
        self,
        semantic_risks: List[RiskFactor],
        pattern_risks: List[Dict[str, Any]],
        repository_context: Optional[Dict[str, Any]],
        algorithmic_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get LLM-based contextual risk judgment as final arbitrator"""
        
        # Extract key metrics for LLM analysis
        semantic_score = algorithmic_scores['semantic_component']['score']
        pattern_score = algorithmic_scores['pattern_component']['score']
        combined_score = algorithmic_scores['overall_risk_score']
        
        # Sample top risks for detailed analysis
        top_semantic_risks = sorted(
            semantic_risks, 
            key=lambda r: self.severity_multipliers.get(r.severity, 0.5) * r.confidence, 
            reverse=True
        )[:3]
        
        judgment_prompt = f"""
        You are an expert software engineering consultant analyzing repository risk assessment.
        
        REPOSITORY: {self.repo_owner}/{self.repo_name}
        
        ALGORITHMIC ANALYSIS RESULTS:
        • Semantic Analysis Score: {semantic_score:.2f}/1.0 (70% weight - direct code issues)
        • Pattern Detection Score: {pattern_score:.2f}/1.0 (30% weight - repository patterns)  
        • Combined Algorithmic Score: {combined_score:.2f}/1.0
        
        RISK BREAKDOWN:
        • Total Semantic Risks: {len(semantic_risks)} (direct code quality issues)
        • Total Pattern Risks: {len(pattern_risks)} (repository-level indicators)
        • Severity Distribution: {algorithmic_scores['severity_distribution']}
        
        TOP SEMANTIC RISKS (Most Critical):
        {self._format_risks_for_llm(top_semantic_risks)}
        
        PATTERN INDICATORS:
        {self._format_patterns_for_llm(pattern_risks[:3])}
        
        As the FINAL ARBITRATOR, provide your expert judgment:
        
        1. **Risk Score Adjustment**: Should the combined score of {combined_score:.2f} be:
           - INCREASED (if context makes it worse than calculated)
           - DECREASED (if context makes it better than calculated)  
           - MAINTAINED (if algorithmic assessment is accurate)
           
        2. **Adjustment Factor**: If adjusting, by how much? (0.0 to 0.3 adjustment range)
        
        3. **Confidence Level**: How confident are you in this assessment? (0.0-1.0)
        
        4. **Business Impact**: What would be the real-world business consequences?
        
        5. **Priority Actions**: What should be done first?
        
        Focus on contextual factors the algorithm might miss: team dynamics, business criticality, 
        technical debt accumulation, and real-world impact.
        
        Provide specific numeric values and clear reasoning.
        """
        
        try:
            llm_response = await self.agentic_explorer.query(judgment_prompt)
            
            # Parse LLM response for adjustment
            llm_judgment = self._parse_llm_judgment(llm_response, algorithmic_scores)
            
            return llm_judgment
            
        except Exception as e:
            logger.error(f"Error getting LLM risk judgment: {e}")
            return {
                'adjustment_factor': 0.0,  # No adjustment on error
                'adjusted_score': combined_score,
                'llm_confidence': 0.3,
                'business_impact_assessment': 'Unable to assess due to LLM error',
                'priority_actions': ['Manual review recommended due to analysis error'],
                'error': str(e)
            }
    
    def _format_risks_for_llm(self, risks: List[RiskFactor]) -> str:
        """Format risks for LLM analysis"""
        if not risks:
            return "No significant semantic risks detected"
            
        formatted = []
        for i, risk in enumerate(risks, 1):
            formatted.append(f"""
        {i}. [{risk.severity.upper()}] {risk.factor_type.replace('_', ' ').title()}
           Description: {risk.description[:200]}...
           Files: {', '.join(risk.affected_files[:3])}{'...' if len(risk.affected_files) > 3 else ''}
           Confidence: {risk.confidence:.2f}
        """)
        return '\n'.join(formatted)
    
    def _format_patterns_for_llm(self, patterns: List[Dict[str, Any]]) -> str:
        """Format pattern risks for LLM analysis"""
        if not patterns:
            return "No significant patterns detected"
            
        formatted = []
        for i, pattern in enumerate(patterns, 1):
            pattern_type = pattern.get('pattern_type', 'unknown')
            risk_score = pattern.get('risk_score', 0.0)
            description = pattern.get('description', 'No description available')
            
            formatted.append(f"""
        {i}. {pattern_type.replace('_', ' ').title()}
           Risk Score: {risk_score:.2f}
           Description: {description[:150]}...
        """)
        return '\n'.join(formatted)
    
    def _parse_llm_judgment(self, llm_response: str, algorithmic_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM judgment response"""
        
        llm_scores = {
            'adjustment_factor': 0.0,  # Default adjustment factor
            'adjusted_score': algorithmic_scores['overall_risk_score'],  # Default fallback
            'llm_confidence': 0.7,  # Default confidence
            'business_impact_assessment': '',
            'priority_actions': [],
            'raw_response': llm_response
        }
        
        # Simple parsing to extract scores
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip().lower()
            
            # Look for adjustment factor
            if 'adjustment factor' in line:
                import re
                factor_match = re.search(r'(\d+\.?\d*)', line)
                if factor_match:
                    try:
                        factor = float(factor_match.group(1))
                        llm_scores['adjustment_factor'] = min(max(factor, 0.0), 0.3)
                    except ValueError:
                        pass
            
            # Look for overall score
            if 'overall risk score' in line or 'overall score' in line:
                import re
                score_match = re.search(r'(\d+\.?\d*)', line)
                if score_match:
                    try:
                        score = float(score_match.group(1))
                        if score > 1.0:  # Handle percentage format
                            score = score / 100.0
                        llm_scores['adjusted_score'] = min(max(score, 0.0), 1.0)
                    except ValueError:
                        pass
            
            # Look for confidence
            if 'confidence' in line and ('0.' in line or '%' in line):
                import re
                conf_match = re.search(r'(\d+\.?\d*)', line)
                if conf_match:
                    try:
                        conf = float(conf_match.group(1))
                        if conf > 1.0:  # Handle percentage format
                            conf = conf / 100.0
                        llm_scores['llm_confidence'] = min(max(conf, 0.0), 1.0)
                    except ValueError:
                        pass
        
        return llm_scores
    
    async def _combine_scores(
        self,
        algorithmic_scores: Dict[str, Any],
        llm_scores: Dict[str, Any],
        semantic_risks: List[RiskFactor],
        pattern_risks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine algorithmic and LLM scores using LLM as final arbitrator"""
        
        # Get base algorithmic score
        base_score = algorithmic_scores.get('overall_risk_score', 0.5)
        
        # Get LLM adjustment
        adjustment_factor = llm_scores.get('adjustment_factor', 0.0)
        llm_confidence = llm_scores.get('llm_confidence', 0.7)
        algo_confidence = algorithmic_scores.get('algorithmic_confidence', 0.85)
        
        # INTUITIVE FINAL SCORING:
        # 1. Start with algorithmic base score (objective)
        # 2. Apply LLM adjustment (contextual arbitration)
        # 3. Weight adjustment by LLM confidence
        
        # Apply adjustment (can be positive or negative)
        if 'increase' in llm_scores.get('raw_response', '').lower():
            final_score = min(base_score + (adjustment_factor * llm_confidence), 1.0)
        elif 'decrease' in llm_scores.get('raw_response', '').lower():
            final_score = max(base_score - (adjustment_factor * llm_confidence), 0.0)
        else:
            # Maintain score if no clear adjustment direction
            final_score = base_score
        
        # Calculate overall confidence (weighted by component confidence)
        overall_confidence = (algo_confidence * 0.7) + (llm_confidence * 0.3)
        
        # Determine weights for transparency
        algo_weight = 1.0 - (adjustment_factor * llm_confidence)  # Decreases as LLM adjusts more
        llm_weight = adjustment_factor * llm_confidence  # Increases with adjustment and confidence
        
        return {
            'overall_risk_score': final_score,
            'overall_confidence': overall_confidence,
            'algorithmic_weight': algo_weight,
            'llm_weight': llm_weight,
            'score_breakdown': {
                'algorithmic_score': base_score,
                'llm_adjustment': adjustment_factor,
                'llm_adjusted_score': llm_scores.get('adjusted_score', base_score),
                'final_score': final_score
            },
            'component_scores': {
                'semantic_score': algorithmic_scores['semantic_component']['score'],
                'pattern_score': algorithmic_scores['pattern_component']['score'],
                'base_combined': base_score,
                'llm_arbitration': adjustment_factor
            },
            'risk_category_scores': algorithmic_scores['risk_category_scores'],
            'file_risk_scores': algorithmic_scores['file_risk_scores'],
            'severity_distribution': algorithmic_scores['severity_distribution'],
            'business_impact': llm_scores.get('business_impact_assessment', ''),
            'llm_insights': {
                'priority_actions': llm_scores.get('priority_actions', []),
                'adjustment_reasoning': self._extract_adjustment_reasoning(llm_scores.get('raw_response', ''))
            }
        }
    
    async def _generate_scoring_insights(
        self,
        final_scores: Dict[str, Any],
        semantic_risks: List[RiskFactor],
        pattern_risks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate insights about the scoring results"""
        
        insights = {
            'risk_level_assessment': self._assess_risk_level(final_scores['overall_risk_score']),
            'top_concerns': self._identify_top_concerns(final_scores, semantic_risks),
            'scoring_reliability': self._assess_scoring_reliability(final_scores),
            'recommendations': self._generate_recommendations(final_scores, semantic_risks, pattern_risks)
        }
        
        return insights
    
    def _assess_risk_level(self, overall_score: float) -> str:
        """Assess overall risk level"""
        if overall_score >= 0.8:
            return "CRITICAL - Immediate action required"
        elif overall_score >= 0.6:
            return "HIGH - Address within 1 week"
        elif overall_score >= 0.4:
            return "MEDIUM - Address within 1 month"
        else:
            return "LOW - Monitor and address as time permits"
    
    def _identify_top_concerns(self, final_scores: Dict[str, Any], semantic_risks: List[RiskFactor]) -> List[str]:
        """Identify top concerns based on scoring"""
        concerns = []
        
        # Check for critical security issues
        security_risks = [r for r in semantic_risks if r.factor_type == 'security_vulnerability' and r.severity in ['critical', 'high']]
        if security_risks:
            concerns.append(f"Security vulnerabilities detected in {len(security_risks)} locations")
        
        # Check for high complexity
        complexity_risks = [r for r in semantic_risks if r.factor_type == 'code_complexity' and r.severity in ['critical', 'high']]
        if complexity_risks:
            concerns.append(f"High code complexity in {len(complexity_risks)} areas")
        
        # Check overall confidence
        if final_scores['overall_confidence'] < 0.6:
            concerns.append("Low confidence in risk assessment - manual review recommended")
        
        return concerns[:5]  # Top 5 concerns
    
    def _assess_scoring_reliability(self, final_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Assess reliability of the scoring"""
        confidence = final_scores['overall_confidence']
        
        if confidence >= 0.8:
            reliability = "HIGH"
            note = "High confidence in assessment"
        elif confidence >= 0.6:
            reliability = "MEDIUM"
            note = "Moderate confidence - consider additional analysis"
        else:
            reliability = "LOW"
            note = "Low confidence - manual review strongly recommended"
        
        return {
            'level': reliability,
            'confidence_score': confidence,
            'note': note,
            'algorithmic_weight': final_scores.get('algorithmic_weight', 0.6),
            'llm_weight': final_scores.get('llm_weight', 0.4)
        }
    
    def _generate_recommendations(
        self,
        final_scores: Dict[str, Any],
        semantic_risks: List[RiskFactor],
        pattern_risks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        overall_score = final_scores['overall_risk_score']
        
        if overall_score >= 0.8:
            recommendations.append("🚨 URGENT: Schedule immediate code review and remediation")
            recommendations.append("🔒 Prioritize security vulnerability fixes")
        elif overall_score >= 0.6:
            recommendations.append("📋 Schedule comprehensive code review within 1 week")
            recommendations.append("🔧 Implement automated code quality checks")
        elif overall_score >= 0.4:
            recommendations.append("📊 Include risk remediation in next sprint planning")
            recommendations.append("🔍 Implement regular code quality monitoring")
        else:
            recommendations.append("✅ Maintain current code quality practices")
            recommendations.append("📈 Continue monitoring for emerging risks")
        
        return recommendations
    
    def _map_pattern_to_risk_type(self, pattern_type: str) -> str:
        """Map pattern types to risk categories"""
        mapping = {
            'file_hotspot': 'maintainability_issue',
            'code_complexity': 'code_complexity',
            'dependency_health': 'security_vulnerability',
            'velocity_spike': 'code_quality',
            'knowledge_concentration': 'maintainability_issue',
            'review_rush': 'code_quality'
        }
        
        return mapping.get(pattern_type, 'code_quality')
    
    def _extract_adjustment_reasoning(self, llm_response: str) -> str:
        """Extract the reasoning behind LLM's adjustment decision"""
        lines = llm_response.split('\n')
        reasoning_lines = []
        
        # Look for reasoning keywords
        for line in lines:
            if any(keyword in line.lower() for keyword in [
                'because', 'due to', 'reason', 'context', 'however', 'although', 
                'considering', 'given that', 'since', 'therefore'
            ]):
                reasoning_lines.append(line.strip())
        
        if reasoning_lines:
            return ' '.join(reasoning_lines[:3])  # Top 3 reasoning statements
        else:
            return "No specific reasoning provided" 