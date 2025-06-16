"""
Data collector for predictive analysis that leverages existing Triage.Flow infrastructure
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Import existing Triage.Flow components
from ...commit_index import CommitIndexManager
from ...patch_linkage import PatchLinkageBuilder
from ...github_client import GitHubIssueClient

logger = logging.getLogger(__name__)

class PredictiveDataCollector:
    """Extends existing data collection with prediction-focused metrics"""
    
    def __init__(self, repo_path: str, repo_owner: str, repo_name: str):
        self.repo_path = repo_path
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        
        # Leverage existing systems
        self.commit_manager = CommitIndexManager(repo_path, repo_owner, repo_name)
        self.patch_builder = PatchLinkageBuilder(repo_owner, repo_name)
        self.github_client = GitHubIssueClient()
        
        logger.info(f"Initialized PredictiveDataCollector for {repo_owner}/{repo_name}")
    
    async def collect_code_metrics(self) -> Dict[str, Any]:
        """Collect code complexity, test coverage, dependency health metrics"""
        logger.info("Collecting code metrics for prediction analysis")
        
        # Initialize commit manager if needed
        if not self.commit_manager.is_initialized():
            logger.warning("Commit manager not initialized, initializing now...")
            await self.commit_manager.initialize()
        
        # Get recent commits for analysis (use direct method instead of empty search)
        recent_commits = self.commit_manager.get_recent_commits(limit=1000)
        
        # Analyze file statistics
        file_stats = {}
        for commit in recent_commits:
            for file_path in commit.files_changed:
                if file_path not in file_stats:
                    file_stats[file_path] = {
                        'touch_frequency': 0,
                        'unique_authors': set(),
                        'recent_commits': 0,
                        'complexity_trend': self._calculate_complexity_trend(file_path)
                    }
                
                file_stats[file_path]['touch_frequency'] += 1
                file_stats[file_path]['unique_authors'].add(commit.author_name)
                file_stats[file_path]['recent_commits'] += 1
        
        # Convert sets to counts for JSON serialization
        for stats in file_stats.values():
            stats['unique_authors'] = len(stats['unique_authors'])
        
        # Collect metrics
        code_data = {
            'file_statistics': file_stats,
            'hotspot_files': self._identify_hotspot_files(file_stats),
            'dependency_health': await self._analyze_dependencies(),
            'test_coverage': self._estimate_test_coverage(),
            'complexity_trends': {
                file: stats['complexity_trend'] 
                for file, stats in file_stats.items()
            }
        }
        
        return code_data
    
    async def collect_team_metrics(self) -> Dict[str, Any]:
        """Collect team velocity, review patterns, collaboration metrics"""
        logger.info("Collecting team metrics for prediction analysis")
        
        # Use existing commit data for team analysis
        team_data = {}
        if self.commit_manager.is_initialized():
            # Get recent commits directly instead of using empty search
            recent_commits = self.commit_manager.get_recent_commits(limit=1000)
            
            author_stats = {}
            for commit in recent_commits:
                author = commit.author_name
                if author not in author_stats:
                    author_stats[author] = {
                        'commits': 0,
                        'files_touched': set(),
                        'recent_activity': []
                    }
                
                author_stats[author]['commits'] += 1
                author_stats[author]['files_touched'].update(commit.files_changed)
                author_stats[author]['recent_activity'].append({
                    'date': commit.commit_date,
                    'files_count': len(commit.files_changed),
                    'insertions': commit.insertions,
                    'deletions': commit.deletions
                })
            
            team_data = {
                'author_statistics': {
                    author: {
                        **stats,
                        'files_touched': len(stats['files_touched']),
                        'velocity_trend': self._calculate_velocity_trend(stats['recent_activity'])
                    }
                    for author, stats in author_stats.items()
                },
                'collaboration_patterns': self._analyze_collaboration_patterns(author_stats),
                'review_patterns': await self._analyze_review_patterns()
            }
        
        return team_data
    
    async def collect_deployment_metrics(self) -> Dict[str, Any]:
        """Collect deployment frequency, failure rates, rollback patterns"""
        logger.info("Collecting deployment metrics for prediction analysis")
        
        # Analyze merge commits and tags for deployment patterns
        deployment_data = {
            'merge_frequency': await self._analyze_merge_frequency(),
            'rollback_indicators': await self._detect_rollback_patterns(),
            'deployment_size_trends': await self._analyze_deployment_sizes(),
            'failure_correlation': await self._correlate_deployments_with_issues()
        }
        
        return deployment_data

    async def collect_hotspot_data(self) -> Dict[str, Any]:
        """Collect file hotspot data using existing commit index"""
        if not self.commit_manager.is_initialized():
            await self.commit_manager.initialize()
        
        hotspots = {}
        for file_path, stats in self.commit_manager.indexer.file_touch_stats.items():
            # Handle different data structures gracefully
            touch_count = stats.get('touches', stats.get('touch_count', 0))
            if touch_count > 10:  # Configurable threshold
                hotspots[file_path] = {
                    'touch_count': touch_count,
                    'unique_authors': len(stats.get('authors', [])),
                    'recent_activity': stats.get('recent_commits', [])[-30:],  # Last 30 commits
                    'risk_score': self._calculate_file_risk_score(stats)
                }
        
        return hotspots
    
    def _calculate_file_risk_score(self, stats: Dict) -> float:
        """Calculate risk score based on file statistics"""
        touch_count = stats.get('touches', stats.get('touch_count', 0))
        touch_weight = min(touch_count / 50.0, 1.0)  # Normalize to 0-1
        author_diversity = 1.0 / max(len(stats.get('authors', [1])), 1)  # Higher diversity = lower risk
        recent_activity = len(stats.get('recent_commits', [])) / 30.0  # Recent activity factor
        
        return (touch_weight * 0.5) + (author_diversity * 0.3) + (recent_activity * 0.2)
    
    def _calculate_complexity_trend(self, file_path: str) -> Dict[str, Any]:
        """Calculate complexity trend for a file"""
        # This would integrate with static analysis tools
        # For now, return a placeholder based on file extension and size
        try:
            import os
            if os.path.exists(os.path.join(self.repo_path, file_path)):
                file_size = os.path.getsize(os.path.join(self.repo_path, file_path))
                lines_estimate = file_size // 50  # Rough estimate
                
                return {
                    'estimated_lines': lines_estimate,
                    'complexity_score': min(lines_estimate / 1000.0, 1.0),  # Normalize
                    'trend': 'increasing' if lines_estimate > 500 else 'stable'
                }
        except Exception as e:
            logger.warning(f"Could not calculate complexity for {file_path}: {e}")
        
        return {'estimated_lines': 0, 'complexity_score': 0.0, 'trend': 'unknown'}
    
    def _identify_hotspot_files(self, file_stats: Dict[str, Any]) -> List[str]:
        """Identify hotspot files based on statistics"""
        hotspots = []
        
        for file_path, stats in file_stats.items():
            # Consider a file a hotspot if it has high touch frequency and multiple authors
            if (stats['touch_frequency'] > 15 and 
                stats['unique_authors'] > 3 and 
                stats['recent_commits'] > 5):
                hotspots.append(file_path)
        
        # Sort by touch frequency and return top 20
        hotspots.sort(key=lambda f: file_stats[f]['touch_frequency'], reverse=True)
        return hotspots[:20]
    
    async def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency health"""
        # This would integrate with dependency analysis tools
        # For now, return a placeholder
        return {
            'outdated_dependencies': [],
            'security_vulnerabilities': 0,
            'dependency_freshness_score': 0.8
        }
    
    def _estimate_test_coverage(self) -> Dict[str, Any]:
        """Estimate test coverage"""
        # This would integrate with coverage tools
        # For now, return a placeholder based on test file presence
        try:
            import os
            test_files = []
            for root, dirs, files in os.walk(self.repo_path):
                for file in files:
                    if 'test' in file.lower() or file.startswith('test_'):
                        test_files.append(file)
            
            return {
                'test_files_count': len(test_files),
                'estimated_coverage': min(len(test_files) * 0.1, 1.0),  # Rough estimate
                'coverage_trend': 'stable'
            }
        except Exception as e:
            logger.warning(f"Could not estimate test coverage: {e}")
            return {'test_files_count': 0, 'estimated_coverage': 0.0, 'coverage_trend': 'unknown'}
    
    def _calculate_velocity_trend(self, recent_activity: List[Dict]) -> Dict[str, Any]:
        """Calculate velocity trend for an author"""
        if not recent_activity:
            return {'recent_spike': False, 'average_daily_commits': 0.0}
        
        # Group by date
        daily_commits = {}
        for activity in recent_activity:
            date = activity['date'][:10]  # Extract date part
            if date not in daily_commits:
                daily_commits[date] = 0
            daily_commits[date] += 1
        
        # Calculate average and detect spikes
        commit_counts = list(daily_commits.values())
        avg_commits = sum(commit_counts) / len(commit_counts) if commit_counts else 0
        max_commits = max(commit_counts) if commit_counts else 0
        
        # Detect spike (more than 3x average)
        recent_spike = max_commits > (avg_commits * 3) if avg_commits > 0 else False
        
        return {
            'recent_spike': recent_spike,
            'average_daily_commits': avg_commits,
            'max_daily_commits': max_commits,
            'correlated_issues': 0  # Would be calculated based on issue correlation
        }
    
    def _analyze_collaboration_patterns(self, author_stats: Dict) -> Dict[str, Any]:
        """Analyze collaboration patterns between team members"""
        # This would analyze file co-editing patterns
        # For now, return basic statistics
        total_authors = len(author_stats)
        
        return {
            'total_active_authors': total_authors,
            'collaboration_score': min(total_authors / 10.0, 1.0),  # Normalize
            'knowledge_distribution': 'distributed' if total_authors > 5 else 'concentrated'
        }
    
    async def _analyze_review_patterns(self) -> Dict[str, Any]:
        """Analyze code review patterns"""
        # This would integrate with PR review data
        # For now, return placeholder
        return {
            'average_review_time': 24.0,  # hours
            'review_thoroughness_score': 0.7,
            'reviewer_distribution': 'balanced'
        }
    
    async def _analyze_merge_frequency(self) -> Dict[str, Any]:
        """Analyze merge commit frequency"""
        if not self.commit_manager.is_initialized():
            return {'merge_frequency': 0.0, 'trend': 'unknown'}
        
        # Search for merge commits
        merge_commits = await self.commit_manager.search_commits("Merge", k=100)
        
        # Group by date to calculate frequency
        daily_merges = {}
        for commit_result in merge_commits:
            date = commit_result.commit.commit_date[:10]
            if date not in daily_merges:
                daily_merges[date] = 0
            daily_merges[date] += 1
        
        avg_daily_merges = sum(daily_merges.values()) / max(len(daily_merges), 1)
        
        return {
            'merge_frequency': avg_daily_merges,
            'total_merges': len(merge_commits),
            'trend': 'increasing' if avg_daily_merges > 1 else 'stable'
        }
    
    async def _detect_rollback_patterns(self) -> Dict[str, Any]:
        """Detect rollback patterns in commits"""
        if not self.commit_manager.is_initialized():
            return {'rollback_count': 0, 'rollback_frequency': 0.0}
        
        # Search for rollback-related commits
        rollback_commits = await self.commit_manager.search_commits("revert", k=50)
        
        return {
            'rollback_count': len(rollback_commits),
            'rollback_frequency': len(rollback_commits) / 30.0,  # per month estimate
            'recent_rollbacks': [c.commit.subject for c in rollback_commits[:5]]
        }
    
    async def _analyze_deployment_sizes(self) -> Dict[str, Any]:
        """Analyze deployment size trends"""
        # This would analyze the size of deployments
        # For now, return placeholder based on recent commits
        if not self.commit_manager.is_initialized():
            return {'average_deployment_size': 0, 'size_trend': 'unknown'}
        
        # Get recent commits directly instead of using empty search
        recent_commits = self.commit_manager.get_recent_commits(limit=50)
        
        total_changes = sum(
            commit.insertions + commit.deletions
            for commit in recent_commits
        )
        
        avg_change_size = total_changes / max(len(recent_commits), 1)
        
        return {
            'average_deployment_size': avg_change_size,
            'size_trend': 'increasing' if avg_change_size > 100 else 'stable',
            'large_deployments': len([c for c in recent_commits if (c.insertions + c.deletions) > 500])
        }
    
    async def _correlate_deployments_with_issues(self) -> Dict[str, Any]:
        """Correlate deployment patterns with issue occurrence"""
        # This would analyze the correlation between deployments and issues
        # For now, return placeholder
        return {
            'correlation_strength': 0.3,
            'high_risk_deployment_patterns': [],
            'issue_spike_after_deployment': False
        } 