"""
Pattern detection engine for identifying bug and team patterns
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from ..data_models.prediction_models import BugPattern, TeamPattern
from .data_collector import PredictiveDataCollector

logger = logging.getLogger(__name__)

class PatternDetectionEngine:
    """Leverages existing commit_index and patch_linkage for pattern detection"""
    
    def __init__(self, data_collector: PredictiveDataCollector):
        self.data_collector = data_collector
        self.commit_manager = data_collector.commit_manager
        self.patch_builder = data_collector.patch_builder
        
        logger.info(f"Initialized PatternDetectionEngine for {data_collector.repo_owner}/{data_collector.repo_name}")
    
    async def detect_bug_patterns(self) -> List[BugPattern]:
        """Identify code patterns that historically lead to issues"""
        logger.info("Detecting bug patterns")
        patterns = []
        
        # Pattern 1: File hotspots (files with high change frequency + bug correlation)
        code_metrics = await self.data_collector.collect_code_metrics()
        hotspot_files = code_metrics.get('hotspot_files', [])
        
        if hotspot_files:
            patterns.append(BugPattern(
                pattern_id="file_hotspots",
                pattern_type="file_hotspot",
                description=f"Detected {len(hotspot_files)} files with high change frequency and bug correlation",
                confidence=0.8,
                historical_occurrences=len(hotspot_files),
                files_affected=hotspot_files,
                risk_score=self._calculate_hotspot_risk(hotspot_files),
                prevention_strategies=[
                    "Increase code review focus on hotspot files",
                    "Add comprehensive tests for frequently changed files",
                    "Consider refactoring complex hotspot files"
                ]
            ))
        
        # Pattern 2: Complexity growth patterns
        complexity_patterns = await self._detect_complexity_patterns()
        patterns.extend(complexity_patterns)
        
        # Pattern 3: Dependency update correlation with issues
        dependency_patterns = await self._detect_dependency_patterns()
        patterns.extend(dependency_patterns)
        
        # Pattern 4: File hotspot patterns using existing data
        hotspot_patterns = await self.detect_file_hotspot_patterns()
        patterns.extend(hotspot_patterns)
        
        logger.info(f"Detected {len(patterns)} bug patterns")
        return patterns
    
    async def detect_team_patterns(self) -> List[TeamPattern]:
        """Identify team behavior patterns that correlate with issues"""
        logger.info("Detecting team patterns")
        patterns = []
        
        team_metrics = await self.data_collector.collect_team_metrics()
        author_stats = team_metrics.get('author_statistics', {})
        
        # Pattern 1: Velocity spikes (sudden increase in commits/changes)
        for author, stats in author_stats.items():
            velocity_trend = stats.get('velocity_trend', {})
            if velocity_trend.get('recent_spike', False):
                patterns.append(TeamPattern(
                    pattern_id=f"velocity_spike_{author}",
                    team_members=[author],
                    pattern_type="velocity_spike",
                    time_period="last_2_weeks",
                    correlation_strength=0.7,
                    issue_count=velocity_trend.get('correlated_issues', 0),
                    prevention_recommendations=[
                        f"Monitor {author}'s workload for potential burnout",
                        "Ensure adequate code review for high-velocity periods",
                        "Consider pair programming for complex changes"
                    ]
                ))
        
        # Pattern 2: Knowledge concentration (single person touching many critical files)
        knowledge_patterns = self._detect_knowledge_concentration(author_stats)
        patterns.extend(knowledge_patterns)
        
        # Pattern 3: Review rush patterns
        review_patterns = await self._detect_review_rush_patterns(team_metrics)
        patterns.extend(review_patterns)
        
        logger.info(f"Detected {len(patterns)} team patterns")
        return patterns

    async def detect_file_hotspot_patterns(self) -> List[BugPattern]:
        """Detect file hotspot patterns using existing data"""
        hotspots = await self.data_collector.collect_hotspot_data()
        patterns = []
        
        # Pattern 1: High-touch files with recent issues
        high_risk_files = [
            file_path for file_path, data in hotspots.items() 
            if data['risk_score'] > 0.7
        ]
        
        if high_risk_files:
            patterns.append(BugPattern(
                pattern_id="high_risk_hotspots",
                pattern_type="file_hotspot",
                description=f"Detected {len(high_risk_files)} high-risk hotspot files",
                confidence=0.8,
                historical_occurrences=len(high_risk_files),
                files_affected=high_risk_files,
                risk_score=sum(hotspots[f]['risk_score'] for f in high_risk_files) / len(high_risk_files),
                prevention_strategies=[
                    f"Add comprehensive tests for {', '.join(high_risk_files[:3])}{'...' if len(high_risk_files) > 3 else ''}",
                    "Implement stricter code review for hotspot files",
                    "Consider refactoring most complex hotspot files"
                ]
            ))
        
        return patterns
    
    def _calculate_hotspot_risk(self, hotspot_files: List[str]) -> float:
        """Calculate overall risk score for hotspot files"""
        if not hotspot_files:
            return 0.0
        
        # Simple risk calculation based on number of hotspots
        base_risk = min(len(hotspot_files) / 20.0, 1.0)  # Normalize to 0-1
        return base_risk
    
    async def _detect_complexity_patterns(self) -> List[BugPattern]:
        """Detect patterns related to code complexity growth"""
        patterns = []
        
        code_metrics = await self.data_collector.collect_code_metrics()
        file_stats = code_metrics.get('file_statistics', {})
        
        # Find files with high complexity scores
        complex_files = [
            file_path for file_path, stats in file_stats.items()
            if stats['complexity_trend']['complexity_score'] > 0.8
        ]
        
        if complex_files:
            patterns.append(BugPattern(
                pattern_id="high_complexity_files",
                pattern_type="code_complexity",
                description=f"Detected {len(complex_files)} files with high complexity scores",
                confidence=0.7,
                historical_occurrences=len(complex_files),
                files_affected=complex_files,
                risk_score=0.8,
                prevention_strategies=[
                    "Refactor complex files to reduce complexity",
                    "Add unit tests for complex functions",
                    "Implement code complexity monitoring"
                ]
            ))
        
        return patterns
    
    async def _detect_dependency_patterns(self) -> List[BugPattern]:
        """Detect patterns related to dependency issues"""
        patterns = []
        
        code_metrics = await self.data_collector.collect_code_metrics()
        dependency_health = code_metrics.get('dependency_health', {})
        
        # Check for outdated dependencies
        outdated_deps = dependency_health.get('outdated_dependencies', [])
        security_vulns = dependency_health.get('security_vulnerabilities', 0)
        
        if outdated_deps or security_vulns > 0:
            patterns.append(BugPattern(
                pattern_id="dependency_issues",
                pattern_type="dependency_health",
                description=f"Detected {len(outdated_deps)} outdated dependencies and {security_vulns} security vulnerabilities",
                confidence=0.9,
                historical_occurrences=len(outdated_deps) + security_vulns,
                files_affected=["package.json", "requirements.txt", "Cargo.toml"],
                risk_score=min((len(outdated_deps) + security_vulns * 2) / 10.0, 1.0),
                prevention_strategies=[
                    "Update outdated dependencies regularly",
                    "Implement automated security scanning",
                    "Monitor dependency health in CI/CD pipeline"
                ]
            ))
        
        return patterns
    
    def _detect_knowledge_concentration(self, author_stats: Dict) -> List[TeamPattern]:
        """Detect knowledge concentration patterns"""
        patterns = []
        
        # Find authors who touch many files (potential knowledge bottlenecks)
        high_touch_authors = [
            author for author, stats in author_stats.items()
            if stats['files_touched'] > 50  # Configurable threshold
        ]
        
        if high_touch_authors:
            patterns.append(TeamPattern(
                pattern_id="knowledge_concentration",
                team_members=high_touch_authors,
                pattern_type="knowledge_gap",
                time_period="recent_activity",
                correlation_strength=0.6,
                issue_count=0,
                prevention_recommendations=[
                    "Implement knowledge sharing sessions",
                    "Encourage code reviews across team members",
                    "Document critical system knowledge",
                    "Rotate responsibilities to distribute knowledge"
                ]
            ))
        
        return patterns
    
    async def _detect_review_rush_patterns(self, team_metrics: Dict) -> List[TeamPattern]:
        """Detect patterns of rushed code reviews"""
        patterns = []
        
        review_patterns = team_metrics.get('review_patterns', {})
        avg_review_time = review_patterns.get('average_review_time', 24.0)
        
        # If average review time is very short, it might indicate rushed reviews
        if avg_review_time < 2.0:  # Less than 2 hours
            patterns.append(TeamPattern(
                pattern_id="rushed_reviews",
                team_members=["team"],
                pattern_type="review_rush",
                time_period="recent_reviews",
                correlation_strength=0.8,
                issue_count=0,
                prevention_recommendations=[
                    "Establish minimum review time guidelines",
                    "Implement review checklists",
                    "Encourage thorough code review practices",
                    "Monitor review quality metrics"
                ]
            ))
        
        return patterns