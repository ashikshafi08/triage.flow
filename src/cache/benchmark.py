"""
Cache Performance Benchmark Tool
Compare memory-only vs Redis-enhanced caching performance
"""
import asyncio
import time
import json
import statistics
from typing import Dict, List, Any
import logging

from .redis_cache_manager import EnhancedCacheManager, CacheLayer, RedisManager
from ..config import settings

logger = logging.getLogger(__name__)

class CacheBenchmark:
    """Benchmark cache performance for AI/RAG workloads"""
    
    def __init__(self):
        self.redis_manager = RedisManager()
        self.results = {}
    
    async def setup(self):
        """Initialize benchmark environment"""
        await self.redis_manager.initialize()
    
    def generate_test_data(self, size_kb: int = 10) -> Dict[str, Any]:
        """Generate test data of specified size"""
        # Simulate RAG context data
        content = "Sample content " * (size_kb * 100)  # Rough KB sizing
        return {
            "query": "What is the architecture of this project?",
            "context": content,
            "metadata": {
                "files": ["file1.py", "file2.py", "file3.py"],
                "timestamp": time.time(),
                "tokens": len(content.split())
            },
            "embedding": [0.1] * 1536  # Simulate embedding vector
        }
    
    async def benchmark_cache_operations(
        self, 
        cache_manager: EnhancedCacheManager,
        test_name: str,
        operations: int = 1000,
        data_size_kb: int = 10
    ) -> Dict[str, float]:
        """Benchmark cache operations"""
        print(f"🧪 Running benchmark: {test_name}")
        
        # Generate test data
        test_data = [self.generate_test_data(data_size_kb) for _ in range(operations)]
        
        # Benchmark SET operations
        set_times = []
        for i, data in enumerate(test_data):
            start = time.perf_counter()
            await cache_manager.set(f"test_key_{i}", data, ttl=3600)
            set_times.append(time.perf_counter() - start)
        
        # Benchmark GET operations
        get_times = []
        hit_count = 0
        for i in range(operations):
            start = time.perf_counter()
            result = await cache_manager.get(f"test_key_{i}")
            get_times.append(time.perf_counter() - start)
            if result is not None:
                hit_count += 1
        
        # Calculate statistics
        results = {
            "operations": operations,
            "data_size_kb": data_size_kb,
            "avg_set_time_ms": statistics.mean(set_times) * 1000,
            "avg_get_time_ms": statistics.mean(get_times) * 1000,
            "p95_set_time_ms": statistics.quantiles(set_times, n=20)[18] * 1000,
            "p95_get_time_ms": statistics.quantiles(get_times, n=20)[18] * 1000,
            "hit_rate": hit_count / operations,
            "ops_per_second": operations / (sum(set_times) + sum(get_times)),
            "total_time_seconds": sum(set_times) + sum(get_times)
        }
        
        # Get cache stats
        stats = cache_manager.get_stats()
        results.update({
            "memory_usage_mb": stats.get("memory_usage_mb", 0),
            "cache_efficiency": stats.get("hit_rate", 0)
        })
        
        return results
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive cache performance comparison"""
        print("🚀 Starting comprehensive cache benchmark...")
        
        # Test configurations
        test_configs = [
            {"operations": 500, "data_size_kb": 5, "name": "small_data"},
            {"operations": 200, "data_size_kb": 50, "name": "medium_data"},
            {"operations": 50, "data_size_kb": 200, "name": "large_data"}
        ]
        
        results = {}
        
        for config in test_configs:
            # Memory-only cache
            memory_cache = EnhancedCacheManager(
                namespace=f"bench_memory_{config['name']}",
                max_memory_size=500 * 1024 * 1024,  # 500MB
                redis_manager=None
            )
            
            memory_results = await self.benchmark_cache_operations(
                memory_cache,
                f"Memory-only ({config['name']})",
                config["operations"],
                config["data_size_kb"]
            )
            
            # Redis-enhanced cache
            redis_cache = EnhancedCacheManager(
                namespace=f"bench_redis_{config['name']}",
                max_memory_size=200 * 1024 * 1024,  # 200MB L1
                redis_manager=self.redis_manager
            )
            
            redis_results = await self.benchmark_cache_operations(
                redis_cache,
                f"Redis-enhanced ({config['name']})",
                config["operations"],
                config["data_size_kb"]
            )
            
            # Calculate improvement
            improvement = {
                "get_speed_improvement": (
                    memory_results["avg_get_time_ms"] / redis_results["avg_get_time_ms"]
                    if redis_results["avg_get_time_ms"] > 0 else 1
                ),
                "throughput_improvement": (
                    redis_results["ops_per_second"] / memory_results["ops_per_second"]
                    if memory_results["ops_per_second"] > 0 else 1
                )
            }
            
            results[config["name"]] = {
                "memory_only": memory_results,
                "redis_enhanced": redis_results,
                "improvement": improvement
            }
        
        return results
    
    async def simulate_rag_workload(self) -> Dict[str, Any]:
        """Simulate realistic RAG workload patterns"""
        print("🔬 Simulating realistic RAG workload...")
        
        # Create Redis-enhanced cache
        rag_cache = EnhancedCacheManager(
            namespace="rag_simulation",
            max_memory_size=300 * 1024 * 1024,  # 300MB L1
            redis_manager=self.redis_manager
        )
        
        # Simulate common queries (some repeated for cache hits)
        queries = [
            "What is the main architecture?",
            "How does authentication work?",
            "Show me the API endpoints",
            "What are the database models?",
            "How to run tests?",
            "What is the main architecture?",  # Repeat for cache hit
            "Deployment instructions",
            "How does authentication work?",  # Repeat for cache hit
            "Error handling patterns",
            "What are the database models?",  # Repeat for cache hit
        ]
        
        start_time = time.time()
        cache_hits = 0
        
        for i, query in enumerate(queries):
            # Check cache first
            cache_key = f"query_{hash(query)}"
            cached_result = await rag_cache.get(cache_key)
            
            if cached_result:
                cache_hits += 1
                # Simulate cache hit processing time
                await asyncio.sleep(0.01)
            else:
                # Simulate expensive RAG computation
                await asyncio.sleep(0.5)  # 500ms for expensive LLM call
                
                # Generate and cache result
                result = {
                    "query": query,
                    "response": f"Generated response for: {query}",
                    "context": self.generate_test_data(100),  # 100KB context
                    "metadata": {"model": "gpt-4", "tokens": 2000}
                }
                
                await rag_cache.set(cache_key, result, ttl=1800)
        
        total_time = time.time() - start_time
        
        return {
            "total_queries": len(queries),
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / len(queries),
            "total_time_seconds": total_time,
            "avg_time_per_query": total_time / len(queries),
            "estimated_time_without_cache": len(queries) * 0.5,
            "time_saved_seconds": (len(queries) * 0.5) - total_time,
            "cache_stats": rag_cache.get_stats()
        }
    
    def print_benchmark_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results"""
        print("\n" + "="*60)
        print("🎯 CACHE PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        if "total_queries" in results:  # RAG workload simulation
            print(f"\n📊 RAG Workload Simulation:")
            print(f"   Total Queries: {results['total_queries']}")
            print(f"   Cache Hits: {results['cache_hits']}")
            print(f"   Hit Rate: {results['cache_hit_rate']:.1%}")
            print(f"   Total Time: {results['total_time_seconds']:.2f}s")
            print(f"   Time Saved: {results['time_saved_seconds']:.2f}s")
            print(f"   ⚡ {results['time_saved_seconds']/results['estimated_time_without_cache']:.1%} faster!")
            
        else:  # Comprehensive benchmark
            for test_name, test_results in results.items():
                print(f"\n📊 {test_name.upper()} DATA TEST:")
                
                memory = test_results["memory_only"]
                redis = test_results["redis_enhanced"]
                improvement = test_results["improvement"]
                
                print(f"   Memory-only:")
                print(f"     • Avg GET: {memory['avg_get_time_ms']:.2f}ms")
                print(f"     • Throughput: {memory['ops_per_second']:.1f} ops/sec")
                
                print(f"   Redis-enhanced:")
                print(f"     • Avg GET: {redis['avg_get_time_ms']:.2f}ms")
                print(f"     • Throughput: {redis['ops_per_second']:.1f} ops/sec")
                
                print(f"   🚀 Improvements:")
                print(f"     • {improvement['get_speed_improvement']:.1f}x faster reads")
                print(f"     • {improvement['throughput_improvement']:.1f}x higher throughput")

async def run_benchmark():
    """Run the cache benchmark"""
    benchmark = CacheBenchmark()
    await benchmark.setup()
    
    # Run comprehensive benchmark
    comp_results = await benchmark.run_comprehensive_benchmark()
    benchmark.print_benchmark_results(comp_results)
    
    # Run RAG workload simulation
    rag_results = await benchmark.simulate_rag_workload()
    benchmark.print_benchmark_results(rag_results)
    
    await benchmark.redis_manager.close()

if __name__ == "__main__":
    asyncio.run(run_benchmark()) 