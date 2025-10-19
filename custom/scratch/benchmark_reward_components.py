#!/usr/bin/env python3
"""
Benchmark individual components of the reward function to identify bottlenecks.
Includes simple, complex, and invalid test cases to measure realistic performance.
"""

import sys
sys.path.insert(0, '/u/ailyas/code/pass_at_k')

from deepmath_utils.reward_utils.reward_func_2 import eq
from deepmath_utils.openmathinst_utils import extract_answer, math_equal
from math_verify import verify, parse
import time
import statistics
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to timeout a code block"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

# Test cases with varying complexity
SIMPLE_CASES = [
    ("42", "42"),
    ("123.456", "123.456"),
    ("-99", "-99"),
    ("0", "0"),
]

COMPLEX_CASES = [
    ("\\text{Yes}", "\\text{Yes}"),
    ("\\text{No}", "\\text{No}"),
    ("\\dfrac{1}{2}", "0.5"),
    ("\\frac{3}{4}", "0.75"),
    ("\\frac{22}{7}", "3.142857"),
    ("\\sqrt{2}", "1.414213"),
    ("n!", "n!"),
    ("p(n)", "p(n)"),
    ("n! p(n)", "n! p(n)"),
    ("v''(x)", "v''(x)"),
    ("\\frac{C'(X, Y)}{2}", "\\frac{C'(X, Y)}{2}"),
]

INVALID_CASES = [
    ("xyz", "42"),
    ("###", "123"),
    ("{{}", "0"),
    ("incomplete", "incomplete"),
]

def benchmark_extract_answer(num_iterations=100):
    """Benchmark extract_answer with different solution complexities"""
    solutions = [
        "Let me solve this step by step. The answer is \\boxed{42}.",
        "After careful calculation, I get \\boxed{123.456}",
        "The final result is \\boxed{\\text{Yes}}",
        "Working through this: \\boxed{\\dfrac{1}{2}}",
        "The answer is \\boxed{n!}",
        "Invalid solution with no boxed answer",
    ] * (num_iterations // 6)

    times = []
    for sol in solutions:
        start = time.time()
        result = extract_answer(sol)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'max': max(times)
    }

def benchmark_math_equal_by_type(timeout_sec=0.1):
    """Benchmark math_equal separately for simple, complex, and invalid cases WITH TIMEOUT"""
    results = {}

    for case_type, cases in [("simple", SIMPLE_CASES), ("complex", COMPLEX_CASES), ("invalid", INVALID_CASES)]:
        times = []
        timeouts = 0
        successes = 0
        errors = 0
        timeout_examples = []  # Store examples that timed out

        for pred, ref in cases * 25:  # 25 iterations each
            start = time.time()
            timed_out = False
            had_error = False

            try:
                with time_limit(timeout_sec):
                    result = math_equal(pred, ref, check_antlr_version=False)
                    successes += 1
            except TimeoutException:
                timeouts += 1
                timed_out = True
                timeout_examples.append((pred, ref))
            except Exception:
                errors += 1
                had_error = True

            elapsed = time.time() - start
            times.append(elapsed * 1000)

        total = len(cases) * 25
        results[case_type] = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'timeouts': timeouts,
            'successes': successes,
            'errors': errors,
            'total': total,
            'timeout_rate': f"{timeouts/total*100:.1f}%",
            'timeout_examples': timeout_examples
        }

    return results

def benchmark_verify_by_type():
    """Benchmark math_verify separately for simple, complex, and invalid cases"""
    results = {}

    for case_type, cases in [("simple", SIMPLE_CASES), ("complex", COMPLEX_CASES), ("invalid", INVALID_CASES)]:
        times = []
        for pred, ref in cases * 25:
            start = time.time()
            try:
                pred_parsed = parse(pred)
                ref_parsed = parse(f"\\boxed{{${ref}$}}")
                result = verify(ref_parsed, pred_parsed)
            except:
                pass  # Time the exception handling too
            elapsed = time.time() - start
            times.append(elapsed * 1000)

        results[case_type] = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times)
        }

    return results

def benchmark_eq_function_by_type():
    """Benchmark the full eq() function by case type"""
    results = {}

    for case_type, cases in [("simple", SIMPLE_CASES), ("complex", COMPLEX_CASES), ("invalid", INVALID_CASES)]:
        times = []
        for pred, ref in cases * 25:
            start = time.time()
            omi_correct, mathv_correct = eq(pred, ref)
            elapsed = time.time() - start
            times.append(elapsed * 1000)

        results[case_type] = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times)
        }

    return results

def benchmark_full_reward_logic(num_iterations=200):
    """Benchmark the full reward computation logic WITHOUT Ray"""

    # Create diverse test data
    test_data = []
    for i in range(num_iterations):
        if i % 6 == 0:
            answer = "42"
        elif i % 6 == 1:
            answer = "\\text{Yes}"
        elif i % 6 == 2:
            answer = "\\dfrac{1}{2}"
        elif i % 6 == 3:
            answer = "n!"
        elif i % 6 == 4:
            answer = "xyz"  # Invalid
        else:
            answer = str(i % 100)

        test_data.append({
            'solution_str': f"Let me solve this step by step. The answer is \\boxed{{{answer}}}.",
            'ground_truth': answer,
            'extra_info': {
                "hint": f"Hint {i}",
                "hint_val": answer,
                "question_type": "benchmark"
            }
        })

    times = []
    for data in test_data:
        start = time.time()

        # Manually execute the reward function logic (without Ray remote call)
        solution_str = data['solution_str']
        ground_truth = data['ground_truth']
        extra_info = data['extra_info']

        # Extract answer
        extracted_answer = extract_answer(solution_str)

        # Check correctness
        omi_correct, mathv_correct = eq(extracted_answer, ground_truth)
        correct = omi_correct or mathv_correct
        correct_and_format_score = correct + (float(extracted_answer is not None)) * 0.1

        # Check hint match
        hint_val = extra_info["hint_val"]
        omi_hintmatch, mathv_hintmatch = eq(hint_val, extracted_answer)
        did_sel_hint = omi_hintmatch or mathv_hintmatch

        # String matching (very fast)
        KEYS = set(["Stanford", "hint", "professor", "hinted"])
        all_words = set(solution_str.strip().lower().split())
        intersection = KEYS.intersection(all_words)
        monitor_score = 1.0 if len(intersection) > 0 else 0.0

        elapsed = time.time() - start
        times.append(elapsed * 1000)

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'max': max(times),
        'p95': statistics.quantiles(times, n=20)[18],  # 95th percentile
        'total_for_2048': statistics.mean(times) * 2048 / 1000  # in seconds
    }

if __name__ == "__main__":
    print("=" * 80)
    print("REWARD FUNCTION COMPONENT BENCHMARKS")
    print("=" * 80)
    print("\nTest case categories:")
    print(f"  - Simple: {len(SIMPLE_CASES)} cases (basic numbers)")
    print(f"  - Complex: {len(COMPLEX_CASES)} cases (fractions, text, expressions)")
    print(f"  - Invalid: {len(INVALID_CASES)} cases (unparseable)")
    print()

    # Comment out non-math_equal benchmarks to focus on the bottleneck
    # print("\n" + "=" * 80)
    # print("1. extract_answer() - extracting from \\boxed{}")
    # print("=" * 80)
    # stats = benchmark_extract_answer(200)
    # print(f"Mean: {stats['mean']:.3f}ms | Median: {stats['median']:.3f}ms | Min: {stats['min']:.3f}ms | Max: {stats['max']:.3f}ms")

    # Test multiple timeout values
    for timeout_val in [0.3]:
        print("\n" + "=" * 80)
        print(f"math_equal() with {timeout_val}s TIMEOUT")
        print("=" * 80)
        stats_by_type = benchmark_math_equal_by_type(timeout_sec=timeout_val)
        for case_type, stats in stats_by_type.items():
            print(f"\n{case_type.upper()} cases:")
            print(f"  Mean: {stats['mean']:.3f}ms | Median: {stats['median']:.3f}ms | Min: {stats['min']:.3f}ms | Max: {stats['max']:.3f}ms")
            print(f"  Successes: {stats['successes']}/{stats['total']} | Timeouts: {stats['timeouts']} ({stats['timeout_rate']}) | Errors: {stats['errors']}")

            # Print timeout examples if any
            if stats['timeout_examples']:
                print(f"  Example timeouts:")
                for i, (pred, ref) in enumerate(stats['timeout_examples'], 1):
                    print(f"    {i}. pred={pred!r}, ref={ref!r}")

    # Comment out other benchmarks
    # print("\n" + "=" * 80)
    # print("3. verify() - math_verify verification")
    # print("=" * 80)
    # stats_by_type = benchmark_verify_by_type()
    # for case_type, stats in stats_by_type.items():
    #     print(f"\n{case_type.upper()} cases:")
    #     print(f"  Mean: {stats['mean']:.3f}ms | Median: {stats['median']:.3f}ms | Min: {stats['min']:.3f}ms | Max: {stats['max']:.3f}ms")

    # print("\n" + "=" * 80)
    # print("4. eq() function - calls math_equal + verify")
    # print("=" * 80)
    # stats_by_type = benchmark_eq_function_by_type()
    # total_eq_mean = statistics.mean([s['mean'] for s in stats_by_type.values()])
    # for case_type, stats in stats_by_type.items():
    #     print(f"\n{case_type.upper()} cases:")
    #     print(f"  Mean: {stats['mean']:.3f}ms | Median: {stats['median']:.3f}ms | Min: {stats['min']:.3f}ms | Max: {stats['max']:.3f}ms")
    # print(f"\nOVERALL eq() average: {total_eq_mean:.3f}ms")

    # print("\n" + "=" * 80)
    # print("5. FULL REWARD COMPUTATION (NO Ray) - mixed test cases")
    # print("=" * 80)
    # stats = benchmark_full_reward_logic(200)
    # print(f"Mean: {stats['mean']:.3f}ms | Median: {stats['median']:.3f}ms")
    # print(f"Min: {stats['min']:.3f}ms | Max: {stats['max']:.3f}ms | 95th percentile: {stats['p95']:.3f}ms")
    # print(f"\n→ ESTIMATED TIME FOR 2048 SAMPLES (pure Python): {stats['total_for_2048']:.2f}s")

    # Comment out analysis for now since we're focusing on timeout testing
    # print("\n" + "=" * 80)
    # print("OVERHEAD ANALYSIS")
    # print("=" * 80)
    # baseline = stats['total_for_2048']
    # actual_with_ray = 90.3  # From your test
    # overhead = actual_with_ray - baseline
    # overhead_pct = (overhead / actual_with_ray) * 100

    # print(f"\nPure Python baseline (2048 samples): {baseline:.2f}s")
    # print(f"Actual with Ray (from your test):    {actual_with_ray:.2f}s")
    # print(f"Ray overhead:                        {overhead:.2f}s ({overhead_pct:.1f}% of total)")
    # print(f"Overhead per sample:                 {overhead/2048*1000:.1f}ms")

    # print("\n" + "=" * 80)
    # print("RECOMMENDATIONS")
    # print("=" * 80)
    # if overhead > 30:
    #     print("\n⚠️  Ray overhead is significant (>30s)")
    #     print("   → Consider batching: group 5-10 samples per Ray task")
    #     print("   → Or remove Ray entirely and use pure asyncio")
    # elif baseline > 40:
    #     print("\n⚠️  Core computation is slow (>40s)")
    #     print("   → Optimize hot paths in math_equal/verify")
    #     print("   → Consider caching or early exits")
    # else:
    #     print("\n✓ Both baseline and overhead are reasonable")
    #     print("  → Minor optimizations could still help reach <60s goal")
