#!/usr/bin/env python3
"""
Speed Benchmark for PII Masking Services.

Tests inference speed with different text lengths.

Throughput = nombre de caractères traités par seconde (chars/s)
- Plus le throughput est élevé, plus le service traite de texte rapidement
- Utile pour comparer l'efficacité des services sur différentes tailles de texte
- Exemple: 1000 chars/s = peut traiter 1000 caractères en 1 seconde
"""

import asyncio
import time
import logging
import json
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

from services_setup import create_services_for_benchmark, get_test_texts

logger = logging.getLogger(__name__)

class SpeedBenchmark:
    """Benchmark inference speed across different services and text lengths."""
    
    def __init__(self):
        self.services = {}
        self.test_texts = {}
        self.results = []
        
    async def initialize(self):
        """Initialize services and test data."""
        logger.info("Initializing speed benchmark...")
        self.services = await create_services_for_benchmark()
        self.test_texts = get_test_texts()
        logger.info(f"Initialized {len(self.services)} services")
        

    
    async def measure_single_inference(self, service_name: str, text: str, warmup: bool = True) -> Dict[str, Any]:
        """Measure inference time for a single text."""
        service = self.services[service_name]
        
        if warmup:
            try:
                await service.predict("Warmup text with John Smith at john@example.com")
            except Exception:
                pass 
        
        start_time = time.time()
        try:
            result = await service.predict(text)
            inference_time = time.time() - start_time
            
            return {
                'service_name': service_name,
                'text_length': len(text),
                'inference_time_seconds': inference_time,
                'inference_time_ms': inference_time * 1000,
                'entities_found': len(result.spans),
                'entity_types': len(result.entities),
                'characters_per_second': len(text) / inference_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"Inference failed for {service_name}: {e}")
            
            return {
                'service_name': service_name,
                'text_length': len(text),
                'inference_time_seconds': inference_time,
                'inference_time_ms': inference_time * 1000,
                'entities_found': 0,
                'entity_types': 0,
                'characters_per_second': 0,
                'success': False,
                'error': str(e)
            }
    
    async def measure_multiple_inferences(self, service_name: str, text: str, num_runs: int = 5) -> Dict[str, Any]:
        """Measure inference time across multiple runs for statistical accuracy."""
        logger.info(f"Measuring {service_name} with {num_runs} runs (text length: {len(text)})")
        
        measurements = []
        
        first_result = await self.measure_single_inference(service_name, text, warmup=True)
        measurements.append(first_result)
        
        for i in range(num_runs - 1):
            result = await self.measure_single_inference(service_name, text, warmup=False)
            measurements.append(result)
            
            await asyncio.sleep(0.5)
        
        successful_measurements = [m for m in measurements if m['success']]
        
        if successful_measurements:
            times = [m['inference_time_seconds'] for m in successful_measurements]
            
            return {
                'service_name': service_name,
                'text_length': len(text),
                'num_runs': num_runs,
                'successful_runs': len(successful_measurements),
                'avg_inference_time_seconds': statistics.mean(times),
                'median_inference_time_seconds': statistics.median(times),
                'min_inference_time_seconds': min(times),
                'max_inference_time_seconds': max(times),
                'std_inference_time_seconds': statistics.stdev(times) if len(times) > 1 else 0,
                'avg_inference_time_ms': statistics.mean(times) * 1000,
                'avg_characters_per_second': len(text) / statistics.mean(times),
                'avg_entities_found': statistics.mean([m['entities_found'] for m in successful_measurements]),
                'all_measurements': measurements
            }
        else:
            return {
                'service_name': service_name,
                'text_length': len(text),
                'num_runs': num_runs,
                'successful_runs': 0,
                'error': 'All measurements failed',
                'all_measurements': measurements
            }
    
    async def run_comprehensive_speed_test(self) -> Dict[str, Any]:
        """Run comprehensive speed test across all services and text lengths."""
        logger.info("Starting comprehensive speed benchmark...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'services_tested': list(self.services.keys()),
            'text_categories': list(self.test_texts.keys()),
            'measurements': []
        }
        
        for service_name in self.services.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING SERVICE: {service_name.upper()}")
            logger.info(f"{'='*60}")
            
            for text_category, text in self.test_texts.items():
                logger.info(f"\nTesting with {text_category} text ({len(text)} chars)...")
                
                try:
                    measurement = await self.measure_multiple_inferences(service_name, text, num_runs=3)
                    measurement['text_category'] = text_category
                    results['measurements'].append(measurement)
                    
                    if measurement.get('successful_runs', 0) > 0:
                        avg_time = measurement['avg_inference_time_seconds']
                        logger.info(f"Average time: {avg_time:.3f}s ({avg_time*1000:.0f}ms)")
                        logger.info(f"Throughput: {measurement['avg_characters_per_second']:.0f} chars/s")
                        logger.info(f"Entities found: {measurement['avg_entities_found']:.1f}")
                    else:
                        logger.error(f"All runs failed")
                    
                except Exception as e:
                    logger.error(f"Failed to test {service_name} with {text_category}: {e}")
                    results['measurements'].append({
                        'service_name': service_name,
                        'text_category': text_category,
                        'text_length': len(text),
                        'error': str(e),
                        'successful_runs': 0
                    })
                
                await asyncio.sleep(1)
            
            await asyncio.sleep(2)
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"speed_benchmark_{timestamp}.json"
        
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of speed benchmark results."""
        print("\n" + "="*80)
        print("SPEED BENCHMARK SUMMARY")
        print("="*80)
        
        print(f"\nServices: {', '.join(results['services_tested'])}")
        print(f"Text Categories: {', '.join(results['text_categories'])}")
        
        # Create summary table
        print(f"\n{'Service':<20} {'Text Size':<12} {'Avg Time (s)':<12} {'Throughput (chars/s)':<20}")
        print("-" * 70)
        
        for measurement in results['measurements']:
            if measurement.get('successful_runs', 0) > 0:
                service = measurement['service_name'][:18]
                category = measurement['text_category'][:10]
                avg_time = measurement['avg_inference_time_seconds']
                throughput = measurement['avg_characters_per_second']
                
                print(f"{service:<20} {category:<12} {avg_time:<12.3f} {throughput:<20.0f}")
        
        # Performance comparison
        print(f"\n" + "="*80)
        print("PERFORMANCE COMPARISON BY TEXT LENGTH")
        print("="*80)
        
        for text_category in results['text_categories']:
            print(f"\n📄 {text_category.upper()} TEXT:")
            category_measurements = [m for m in results['measurements'] 
                                   if m.get('text_category') == text_category and m.get('successful_runs', 0) > 0]
            
            if category_measurements:
                # Sort by speed (fastest first)
                category_measurements.sort(key=lambda x: x['avg_inference_time_seconds'])
                
                for i, m in enumerate(category_measurements, 1):
                    service = m['service_name']
                    time_s = m['avg_inference_time_seconds']
                    time_ms = time_s * 1000
                    entities = m['avg_entities_found']
                    
                    # Speed relative to fastest
                    if i == 1:
                        speed_factor = "🥇 FASTEST"
                    else:
                        fastest_time = category_measurements[0]['avg_inference_time_seconds']
                        factor = time_s / fastest_time
                        speed_factor = f"{factor:.1f}x slower"
                    
                    print(f"  {i}. {service:<18} {time_ms:>6.0f}ms  ({entities:.1f} entities)  {speed_factor}")

async def main():
    """Main speed benchmark execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    benchmark = SpeedBenchmark()
    
    try:
        await benchmark.initialize()
        
        results = await benchmark.run_comprehensive_speed_test()
        
        output_file = benchmark.save_results(results)
        
        benchmark.print_summary(results)
        
        print(f"\n{'='*80}")
        print("SPEED BENCHMARK COMPLETED!")
        print(f"Results saved to: {output_file}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Speed benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 