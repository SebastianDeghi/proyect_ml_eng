"""
api_examples.py - Examples of how to use the sentiment analysis API

This script demonstrates:
1. Single prediction requests
2. Batch prediction requests
3. Error handling
4. Async requests

Usage:
    python examples/api_examples.py
"""

import requests
import json
import time
from typing import List, Dict
import asyncio
import aiohttp

# =============================================
# Configuration
# =============================================

API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict"
BATCH_ENDPOINT = f"{API_URL}/batch"
HEALTH_ENDPOINT = f"{API_URL}/health"


# =============================================
# Helper Functions
# =============================================

def print_section(title: str):
    """Print a formatted section title."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(result: Dict, index: int = None):
    """Print a formatted prediction result."""
    prefix = f"[{index}] " if index is not None else ""
    print(f"{prefix}Sentiment: {result['sentiment'].upper()}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Text length: {result['text_length']} characters")


# =============================================
# Health Check
# =============================================

def check_health():
    """Check if the API is healthy."""
    print_section("Health Check")
    
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ API is healthy")
        print(f"   Status: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Version: {data['version']}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ API not reachable at {API_URL}")
        print("   Make sure the API is running: make api")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# =============================================
# Single Prediction Examples
# =============================================

def single_prediction_examples():
    """Demonstrate single prediction requests."""
    print_section("Single Prediction Examples")
    
    test_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Waste of time and money. Worst movie ever.",
        "The acting was good but the plot was boring and predictable.",
        "I don't know what to say. It was okay I guess. Nothing special.",
        "Best movie I've seen all year! Highly recommended! ★★★★★",
        "What a disaster. Poor directing, bad acting, awful script.",
        "An emotional rollercoaster from start to finish. Beautifully made.",
        "Not great, not terrible. Just average.",
        "Masterpiece of modern cinema. A must-watch for everyone.",
        "I fell asleep twice. So boring and unoriginal."
    ]
    
    for i, review in enumerate(test_reviews, 1):
        print(f"\n📝 Review {i}: {review[:80]}...")
        
        response = requests.post(
            PREDICT_ENDPOINT,
            json={"text": review},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print_result(result)
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)


# =============================================
# Batch Prediction Examples
# =============================================

def batch_prediction_examples():
    """Demonstrate batch prediction requests."""
    print_section("Batch Prediction Examples")
    
    reviews = [
        "I loved this movie!",
        "Terrible film, don't watch it.",
        "The special effects were amazing!",
        "Boring and too long.",
        "Great acting, weak story.",
        "A masterpiece!",
        "Waste of time.",
        "Absolutely fantastic!",
        "Could have been better.",
        "One of the best films I've ever seen."
    ]
    
    print(f"\n📦 Sending {len(reviews)} reviews in batch...")
    
    response = requests.post(
        BATCH_ENDPOINT,
        json={"texts": reviews},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Batch processed: {data['total_count']} reviews")
        
        for i, result in enumerate(data['results'], 1):
            print(f"\n[{i}] {reviews[i-1][:50]}...")
            print_result(result)
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")


# =============================================
# Performance Benchmark
# =============================================

def performance_benchmark(num_requests: int = 100):
    """Benchmark API performance."""
    print_section(f"Performance Benchmark ({num_requests} requests)")
    
    test_review = "This movie is absolutely fantastic! I highly recommend it."
    
    # Warm-up
    for _ in range(5):
        requests.post(PREDICT_ENDPOINT, json={"text": test_review})
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_requests):
        response = requests.post(PREDICT_ENDPOINT, json={"text": test_review})
        response.raise_for_status()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = (total_time / num_requests) * 1000  # milliseconds
    
    print(f"✅ {num_requests} requests completed")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Average time: {avg_time:.2f} ms per request")
    print(f"   Throughput: {num_requests / total_time:.1f} requests/second")


# =============================================
# Async Examples (for high throughput)
# =============================================

async def async_predict(session, text: str, semaphore: asyncio.Semaphore):
    """Make an async prediction request."""
    async with semaphore:
        try:
            async with session.post(PREDICT_ENDPOINT, json={"text": text}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}


async def async_batch_example(num_requests: int = 50):
    """Demonstrate async requests for high throughput."""
    print_section(f"Async Batch Example ({num_requests} concurrent requests)")
    
    test_review = "This movie is amazing!"
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    
    async with aiohttp.ClientSession() as session:
        tasks = [async_predict(session, test_review, semaphore) for _ in range(num_requests)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        successful = sum(1 for r in results if "error" not in r)
        total_time = end_time - start_time
        
        print(f"✅ {successful}/{num_requests} requests successful")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Throughput: {num_requests / total_time:.1f} requests/second")


def run_async_example():
    """Wrapper to run async example."""
    asyncio.run(async_batch_example(50))


# =============================================
# Error Handling Examples
# =============================================

def error_handling_examples():
    """Demonstrate error handling scenarios."""
    print_section("Error Handling Examples")
    
    test_cases = [
        ("Empty text", {"text": ""}),
        ("Missing field", {}),
        ("Invalid field", {"review": "text"}),
        ("Very long text", {"text": "a" * 20000}),
        ("Special characters", {"text": "🔥 🎬 🍿" * 1000})
    ]
    
    for name, payload in test_cases:
        print(f"\n🔍 Test: {name}")
        print(f"   Payload: {payload}")
        
        response = requests.post(PREDICT_ENDPOINT, json=payload)
        
        if response.status_code == 200:
            print(f"   ✅ Success: {response.json()}")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text[:100]}")


# =============================================
# Model Info
# =============================================

def get_model_info():
    """Get model information from the API."""
    print_section("Model Information")
    
    response = requests.get(f"{API_URL}/info")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Model Type: {data['model_type']}")
        print(f"Representation: {data['representation']}")
        print(f"Vocabulary Size: {data['vocabulary_size']}")
        print(f"\nPerformance Metrics:")
        for metric, value in data['metrics'].items():
            print(f"  {metric.capitalize()}: {value:.4f}")
    else:
        print(f"❌ Error: {response.status_code}")


# =============================================
# Main
# =============================================

def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("  IMDB Sentiment Analysis API - Examples")
    print("=" * 60)
    
    # Check if API is running
    if not check_health():
        print("\n⚠️ Please start the API first:")
        print("   make api")
        print("   or")
        print("   cd ml_pipeline/api && uvicorn app:app --reload")
        return
    
    # Run examples
    single_prediction_examples()
    batch_prediction_examples()
    performance_benchmark(50)
    error_handling_examples()
    get_model_info()
    
    # Uncomment for async example (requires aiohttp)
    # run_async_example()
    
    print("\n" + "=" * 60)
    print("  All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()