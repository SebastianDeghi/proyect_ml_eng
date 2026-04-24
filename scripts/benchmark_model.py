"""
benchmark_model.py - Benchmark the sentiment analysis model

Measures:
- Inference time per prediction
- Memory usage
- Throughput

Usage:
    python scripts/benchmark_model.py
"""

import sys
import time
import psutil
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.predict import load_model_and_vectorizer, predict_sentiment


def measure_inference_time(model, vectorizer, texts: list, num_iterations: int = 100):
    """Measure average inference time."""
    print("\n📊 Inference Time Benchmark")
    print("-" * 40)
    
    # Warm-up
    for _ in range(10):
        predict_sentiment("Warm-up text", model, vectorizer)
    
    # Measure
    times = []
    for i in range(num_iterations):
        text = texts[i % len(texts)]
        start = time.perf_counter()
        predict_sentiment(text, model, vectorizer)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)
    
    print(f"  Samples: {num_iterations}")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  P95: {p95_time:.2f} ms")
    print(f"  P99: {p99_time:.2f} ms")
    print(f"  Throughput: {1000 / avg_time:.1f} req/sec")
    
    return times


def measure_memory_usage():
    """Measure memory usage of the model."""
    print("\n💾 Memory Usage Benchmark")
    print("-" * 40)
    
    process = psutil.Process()
    
    # Memory before loading
    mem_before = process.memory_info().rss / 1024 / 1024
    
    # Load model
    model, vectorizer = load_model_and_vectorizer()
    
    # Memory after loading
    mem_after = process.memory_info().rss / 1024 / 1024
    
    print(f"  Memory before loading: {mem_before:.2f} MB")
    print(f"  Memory after loading: {mem_after:.2f} MB")
    print(f"  Model memory footprint: {mem_after - mem_before:.2f} MB")
    
    return model, vectorizer


def measure_batch_throughput(model, vectorizer, batch_sizes: list = [1, 10, 50, 100]):
    """Measure throughput for different batch sizes."""
    print("\n📦 Batch Throughput Benchmark")
    print("-" * 40)
    
    from ml_pipeline.predict import predict_batch
    
    test_text = "This is a test review for benchmarking purposes."
    texts = [test_text] * max(batch_sizes)
    
    for batch_size in batch_sizes:
        batch_texts = texts[:batch_size]
        
        start = time.perf_counter()
        predict_batch(batch_texts, model, vectorizer)
        end = time.perf_counter()
        
        total_time = end - start
        per_request_time = (total_time / batch_size) * 1000
        
        print(f"  Batch size {batch_size:3d}: {total_time*1000:.2f} ms total, {per_request_time:.2f} ms/request")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("  Model Benchmark")
    print("=" * 60)
    
    # Test texts
    test_texts = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Waste of time and money.",
        "The acting was good but the plot was boring.",
        "Best movie I've seen all year! Highly recommended!",
        "Not great, not terrible. Just average."
    ]
    
    # Measure memory and load model
    model, vectorizer = measure_memory_usage()
    
    # Measure inference time
    times = measure_inference_time(model, vectorizer, test_texts, 200)
    
    # Measure batch throughput
    measure_batch_throughput(model, vectorizer)
    
    print("\n" + "=" * 60)
    print("  Benchmark Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()