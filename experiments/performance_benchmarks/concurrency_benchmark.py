#!/usr/bin/env python3
"""
Concurrency Benchmark for PII Masking Services.

Tests the async/sync architecture performance with sequential vs concurrent requests.
"""

import asyncio
import time
import logging
import json
import statistics
import psutil
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

from services_setup import create_services_for_benchmark, get_test_texts

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system performance during benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.measurements = []
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.measurements = []
        
    def record_measurement(self, label: str = ""):
        """Record a performance measurement."""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            measurement = {
                'timestamp': time.time(),
                'label': label,
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'system_cpu': psutil.cpu_percent(),
                'system_memory': psutil.virtual_memory().percent
            }
            self.measurements.append(measurement)
            
        except Exception as e:
            logger.warning(f"Failed to record measurement: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.measurements:
            return {}
            
        cpu_values = [m['cpu_percent'] for m in self.measurements if m['cpu_percent'] > 0]
        memory_values = [m['memory_mb'] for m in self.measurements]
        
        return {
            'avg_cpu_percent': statistics.mean(cpu_values) if cpu_values else 0,
            'max_cpu_percent': max(cpu_values) if cpu_values else 0,
            'avg_memory_mb': statistics.mean(memory_values),
            'max_memory_mb': max(memory_values),
            'num_measurements': len(self.measurements)
        }

class ConcurrencyBenchmark:
    """Main benchmark class for testing concurrency performance."""
    
    def __init__(self):
        self.services = {}
        self.test_texts = {}
        self.results = []
        self.monitor = PerformanceMonitor()
        
    async def initialize(self):
        """Initialize services and test data."""
        logger.info("Initializing benchmark services...")
        self.services = await create_services_for_benchmark()
        self.test_texts = get_test_texts()
        logger.info(f"Initialized {len(self.services)} services and {len(self.test_texts)} test texts")
        
    async def run_sequential_test(self, service_name: str, text: str, num_requests: int = 5) -> Dict[str, Any]:
        """Run sequential requests (naive approach)."""
        logger.info(f"Running sequential test: {service_name}, {num_requests} requests")
        
        service = self.services[service_name]
        times = []
        
        self.monitor.start_monitoring()
        self.monitor.record_measurement(f"sequential_{service_name}_start")
        
        start_time = time.time()
        
        for i in range(num_requests):
            request_start = time.time()
            
            try:
                result = await service.predict(text)
                request_time = time.time() - request_start
                times.append(request_time)
                
                logger.debug(f"Sequential request {i+1}: {request_time:.2f}s, {len(result.entities)} entities")
                self.monitor.record_measurement(f"sequential_{service_name}_req_{i+1}")
                
            except Exception as e:
                logger.error(f"Sequential request {i+1} failed: {e}")
                times.append(float('inf'))
        
        total_time = time.time() - start_time
        self.monitor.record_measurement(f"sequential_{service_name}_end")
        
        return {
            'method': 'sequential',
            'service': service_name,
            'num_requests': num_requests,
            'total_time': total_time,
            'individual_times': times,
            'avg_time_per_request': statistics.mean([t for t in times if t != float('inf')]),
            'throughput_requests_per_second': num_requests / total_time,
            'performance_stats': self.monitor.get_stats()
        }
    
    async def run_concurrent_test(self, service_name: str, text: str, num_requests: int = 5) -> Dict[str, Any]:
        """Run concurrent requests (similar to async/sync architecture)."""
        logger.info(f"Running concurrent test: {service_name}, {num_requests} requests")
        
        service = self.services[service_name]
        
        self.monitor.start_monitoring()
        self.monitor.record_measurement(f"concurrent_{service_name}_start")
        
        start_time = time.time()
        
        tasks = []
        for i in range(num_requests):
            task = service.predict(text)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            success_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Concurrent request {i+1} failed: {result}")
                else:
                    success_count += 1
                    logger.debug(f"Concurrent request {i+1}: success, {len(result.entities)} entities")
            
            self.monitor.record_measurement(f"concurrent_{service_name}_end")
            
            return {
                'method': 'concurrent',
                'service': service_name,
                'num_requests': num_requests,
                'successful_requests': success_count,
                'total_time': total_time,
                'avg_time_per_request': total_time,  # All requests run in parallel
                'throughput_requests_per_second': success_count / total_time,
                'performance_stats': self.monitor.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Concurrent test failed: {e}")
            return {
                'method': 'concurrent',
                'service': service_name,
                'num_requests': num_requests,
                'successful_requests': 0,
                'total_time': time.time() - start_time,
                'error': str(e),
                'performance_stats': self.monitor.get_stats()
            }
    
    async def run_scalability_test(self, service_name: str, text: str, request_counts: List[int] = [1, 2, 5, 10]) -> List[Dict[str, Any]]:
        """Test scalability with different numbers of concurrent requests."""
        logger.info(f"Running scalability test: {service_name}")
        
        results = []
        
        for num_requests in request_counts:
            logger.info(f"Testing {num_requests} concurrent requests...")
            
            concurrent_result = await self.run_concurrent_test(service_name, text, num_requests)
            concurrent_result['test_type'] = 'scalability'
            results.append(concurrent_result)
            
            await asyncio.sleep(1)
        
        return results
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark comparing all services and approaches."""
        logger.info("Starting comprehensive benchmark...")
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'test_environment': {
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'platform': os.name
            },
            'services_tested': list(self.services.keys()),
            'results': []
        }
        
        test_text = self.test_texts['medium']
        text_length = len(test_text)
        
        logger.info(f"Using test text of {text_length} characters")
        
        for service_name in self.services.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING SERVICE: {service_name.upper()}")
            logger.info(f"{'='*60}")
            
            service_results = {
                'service_name': service_name,
                'text_length': text_length,
                'tests': []
            }
            
            try:
                logger.info("\n--- Test 1: Sequential vs Concurrent (5 requests) ---")
                
                sequential_result = await self.run_sequential_test(service_name, test_text, 5)
                sequential_result['test_type'] = 'comparison'
                service_results['tests'].append(sequential_result)
                
                await asyncio.sleep(2)
                
                concurrent_result = await self.run_concurrent_test(service_name, test_text, 5)
                concurrent_result['test_type'] = 'comparison'
                service_results['tests'].append(concurrent_result)
                
                if sequential_result['total_time'] > 0:
                    improvement = (sequential_result['total_time'] - concurrent_result['total_time']) / sequential_result['total_time']
                    service_results['concurrent_improvement'] = improvement
                    logger.info(f"Concurrency improvement: {improvement:.1%}")
                
                await asyncio.sleep(2)
                
                logger.info("\n--- Test 2: Scalability Test ---")
                scalability_results = await self.run_scalability_test(service_name, test_text, [1, 2, 5, 8])
                service_results['tests'].extend(scalability_results)
                
            except Exception as e:
                logger.error(f"Failed to test service {service_name}: {e}")
                service_results['error'] = str(e)
            
            benchmark_results['results'].append(service_results)
            
            await asyncio.sleep(3)
        
        return benchmark_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"concurrency_benchmark_{timestamp}.json"
        
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of benchmark results."""
        print("\n" + "="*80)
        print("CONCURRENCY BENCHMARK SUMMARY")
        print("="*80)
        
        print(f"\nTest Environment:")
        env = results['test_environment']
        print(f"  - Python: {env['python_version']}")
        print(f"  - CPU cores: {env['cpu_count']}")
        print(f"  - Memory: {env['memory_gb']} GB")
        
        print(f"\nServices Tested: {', '.join(results['services_tested'])}")
        
        for service_result in results['results']:
            service_name = service_result['service_name']
            print(f"\n{'-'*60}")
            print(f"SERVICE: {service_name.upper()}")
            print(f"{'-'*60}")
            
            if 'error' in service_result:
                print(f"  ERROR: {service_result['error']}")
                continue
            
            # Find sequential vs concurrent comparison
            sequential_test = None
            concurrent_test = None
            
            for test in service_result['tests']:
                if test.get('test_type') == 'comparison':
                    if test['method'] == 'sequential':
                        sequential_test = test
                    elif test['method'] == 'concurrent':
                        concurrent_test = test
            
            if sequential_test and concurrent_test:
                print(f"  Sequential (5 requests): {sequential_test['total_time']:.2f}s total")
                print(f"  Concurrent (5 requests): {concurrent_test['total_time']:.2f}s total")
                
                improvement = service_result.get('concurrent_improvement', 0)
                if improvement > 0:
                    speedup = sequential_test['total_time'] / concurrent_test['total_time']
                    print(f"  Speedup: {speedup:.1f}x faster ({improvement:.1%} improvement)")
                else:
                    print(f"  No significant improvement")
                
                seq_throughput = sequential_test.get('throughput_requests_per_second', 0)
                conc_throughput = concurrent_test.get('throughput_requests_per_second', 0)
                print(f"  Throughput - Sequential: {seq_throughput:.1f} req/s")
                print(f"  Throughput - Concurrent: {conc_throughput:.1f} req/s")
            
            # Scalability results
            scalability_tests = [t for t in service_result['tests'] if t.get('test_type') == 'scalability']
            if scalability_tests:
                print(f"  \n  Scalability Results:")
                for test in scalability_tests:
                    num_req = test['num_requests']
                    total_time = test['total_time']
                    throughput = test.get('throughput_requests_per_second', 0)
                    print(f"    {num_req} requests: {total_time:.2f}s ({throughput:.1f} req/s)")

async def main():
    """Main benchmark execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    benchmark = ConcurrencyBenchmark()
    
    try:
        await benchmark.initialize()
        
        results = await benchmark.run_comprehensive_benchmark()
        
        output_file = benchmark.save_results(results)
        
        benchmark.print_summary(results)
        
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETED!")
        print(f"Results saved to: {output_file}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 