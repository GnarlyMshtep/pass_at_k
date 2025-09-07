#!/usr/bin/env python3
"""
Script to analyze the diversity of attempts in JSONL files using math_verify.
For questions with partial scores (0 < score < 1), extracts attempts and counts unique ones.
"""

import json
import re
from collections import defaultdict, Counter
from typing import List, Set
import argparse
from pathlib import Path

def math_verify(expr1: str, expr2: str, tolerance: float = 1e-9) -> bool:
    """
    Check if two mathematical expressions are equivalent.
    Uses a safe, fast approach that avoids expensive symbolic computation.
    
    Args:
        expr1, expr2: Mathematical expressions as strings
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if expressions are mathematically equivalent
    """
    expr1 = expr1.strip()
    expr2 = expr2.strip()
    
    # Quick string comparison first
    if expr1 == expr2:
        return True
    
    try:
        # Try to evaluate as numbers first (most common case)
        try:
            val1 = float(expr1)
            val2 = float(expr2)
            return abs(val1 - val2) < tolerance
        except ValueError:
            pass
        
        # For non-numeric expressions, try limited symbolic comparison
        try:
            import sympy as sp
            
            # Skip symbolic computation for potentially expensive expressions
            # Check length and complexity
            if len(expr1) > 50 or len(expr2) > 50:
                return expr1 == expr2
            
            # Check for potentially problematic patterns that could be slow
            problematic_patterns = ['**', 'factorial', '!']
            has_large_numbers = False
            
            # Check for large numbers that might cause issues
            import re
            numbers = re.findall(r'\d+', expr1 + expr2)
            for num in numbers:
                if len(num) > 10:  # Numbers with more than 10 digits
                    has_large_numbers = True
                    break
            
            if (any(pattern in expr1 or pattern in expr2 for pattern in problematic_patterns) or 
                has_large_numbers):
                # For potentially expensive operations, just do string comparison
                return expr1 == expr2
            
            # Safe symbolic comparison for simple expressions
            sym1 = sp.sympify(expr1, evaluate=False)  # Don't evaluate immediately
            sym2 = sp.sympify(expr2, evaluate=False)
            
            # If both are numbers after sympify, compare numerically
            try:
                if sym1.is_number and sym2.is_number:
                    val1 = float(sym1.evalf())
                    val2 = float(sym2.evalf())
                    return abs(val1 - val2) < tolerance
            except:
                pass
            
            # For simple algebraic expressions, try basic simplification
            try:
                diff = sym1 - sym2
                # Only try simplification if the expression is reasonably simple
                if len(str(diff)) < 100:
                    simplified = sp.simplify(diff)
                    return simplified == 0 or simplified.equals(0)
                else:
                    return expr1 == expr2
            except:
                return expr1 == expr2
                
        except ImportError:
            # SymPy not available, fall back to string comparison
            return expr1 == expr2
        except Exception:
            # Any other sympy error, fall back to string comparison
            return expr1 == expr2
            
    except Exception:
        # If all else fails, do string comparison
        return expr1 == expr2


def extract_attempts(output_text: str) -> List[str]:
    """
    Extract all attempts from the output text using regex.
    
    Args:
        output_text: The full output text containing attempt tags
        
    Returns:
        List of attempt values as strings
    """
    # Pattern to match <attempt-i>value</attempt-i>
    pattern = r'<attempt-\d+>(.*?)</attempt-\d+>'
    matches = re.findall(pattern, output_text, re.DOTALL)
    
    # Clean up the attempts (strip whitespace)
    attempts = [match.strip() for match in matches]
    return attempts


def count_unique_attempts(attempts: List[str]) -> int:
    """
    Count unique attempts using math_verify for equivalence checking.
    
    Args:
        attempts: List of attempt strings
        
    Returns:
        Number of unique attempts
    """
    if not attempts:
        return 0
    
    unique_attempts = []
    
    for attempt in attempts:
        is_unique = True
        for unique_attempt in unique_attempts:
            if math_verify(attempt, unique_attempt):
                is_unique = False
                break
        
        if is_unique:
            unique_attempts.append(attempt)
    
    return len(unique_attempts)


def analyze_jsonl_file(file_path: str) -> dict:
    """
    Analyze a JSONL file for attempt diversity.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Dictionary with analysis results
    """
    partial_score_questions = []
    high_score_questions = []
    partial_unique_attempt_counts = []
    high_unique_attempt_counts = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                score = data.get('score', 0)
                output_text = data.get('output', '')
                attempts = extract_attempts(output_text)
                
                if attempts:  # Only process if we found attempts
                    unique_count = count_unique_attempts(attempts)
                    
                    question_data = {
                        'line_num': line_num,
                        'score': score,
                        'total_attempts': len(attempts),
                        'unique_attempts': unique_count,
                        'attempts': attempts,
                        'input_preview': data.get('input', '')[:100] + '...' if len(data.get('input', '')) > 100 else data.get('input', '')
                    }
                    
                    # Filter for partial scores (0 < score < 1)
                    if 0 < score < 1:
                        partial_score_questions.append(question_data)
                        partial_unique_attempt_counts.append(unique_count)
                    
                    # Filter for high scores (score >= 1)
                    elif score >= 1:
                        high_score_questions.append(question_data)
                        high_unique_attempt_counts.append(unique_count)
            
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    # Calculate statistics for partial scores
    if partial_unique_attempt_counts:
        partial_count_distribution = Counter(partial_unique_attempt_counts)
        partial_avg_unique = sum(partial_unique_attempt_counts) / len(partial_unique_attempt_counts)
    else:
        partial_count_distribution = Counter()
        partial_avg_unique = 0
    
    # Calculate statistics for high scores
    if high_unique_attempt_counts:
        high_count_distribution = Counter(high_unique_attempt_counts)
        high_avg_unique = sum(high_unique_attempt_counts) / len(high_unique_attempt_counts)
    else:
        high_count_distribution = Counter()
        high_avg_unique = 0
    
    results = {
        'partial_scores': {
            'total_questions': len(partial_score_questions),
            'unique_attempt_count_distribution': dict(partial_count_distribution),
            'average_unique_attempts': partial_avg_unique,
            'questions_with_details': partial_score_questions[:10],  # Show first 10 for inspection
            'all_unique_counts': partial_unique_attempt_counts
        },
        'high_scores': {
            'total_questions': len(high_score_questions),
            'unique_attempt_count_distribution': dict(high_count_distribution),
            'average_unique_attempts': high_avg_unique,
            'questions_with_details': high_score_questions[:10],  # Show first 10 for inspection
            'all_unique_counts': high_unique_attempt_counts
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze attempt diversity in JSONL files')
    parser.add_argument('jsonl_file', help='Path to the JSONL file to analyze')
    parser.add_argument('--output', '-o', help='Output file for detailed results (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    if not Path(args.jsonl_file).exists():
        print(f"Error: File {args.jsonl_file} does not exist")
        return
    
    print(f"Analyzing {args.jsonl_file}...")
    results = analyze_jsonl_file(args.jsonl_file)
    
    # Print summary
    print("\n" + "="*50)
    print("ATTEMPT DIVERSITY ANALYSIS")
    print("="*50)
    
    # Partial scores analysis
    print(f"\nPARTIAL SCORES (0 < score < 1):")
    print(f"Total questions: {results['partial_scores']['total_questions']}")
    if results['partial_scores']['total_questions'] > 0:
        print(f"Average unique attempts per question: {results['partial_scores']['average_unique_attempts']:.2f}")
        print("\nDistribution of unique attempt counts:")
        for count, frequency in sorted(results['partial_scores']['unique_attempt_count_distribution'].items()):
            percentage = frequency / results['partial_scores']['total_questions'] * 100
            print(f"  {count} unique attempts: {frequency} questions ({percentage:.1f}%)")
    
    # High scores analysis
    print(f"\nHIGH SCORES (score >= 1):")
    print(f"Total questions: {results['high_scores']['total_questions']}")
    if results['high_scores']['total_questions'] > 0:
        print(f"Average unique attempts per question: {results['high_scores']['average_unique_attempts']:.2f}")
        print("\nDistribution of unique attempt counts:")
        for count, frequency in sorted(results['high_scores']['unique_attempt_count_distribution'].items()):
            percentage = frequency / results['high_scores']['total_questions'] * 100
            print(f"  {count} unique attempts: {frequency} questions ({percentage:.1f}%)")
    
    if args.verbose:
        if results['partial_scores']['questions_with_details']:
            print("\nFirst 10 PARTIAL SCORE questions (sample):")
            print("-" * 50)
            for i, q in enumerate(results['partial_scores']['questions_with_details'], 1):
                print(f"{i}. Line {q['line_num']}: Score={q['score']:.3f}")
                print(f"   Total attempts: {q['total_attempts']}, Unique: {q['unique_attempts']}")
                print(f"   Attempts: {q['attempts']}")
                print(f"   Input preview: {q['input_preview']}")
                print()
        
        if results['high_scores']['questions_with_details']:
            print("\nFirst 10 HIGH SCORE questions (sample):")
            print("-" * 50)
            for i, q in enumerate(results['high_scores']['questions_with_details'], 1):
                print(f"{i}. Line {q['line_num']}: Score={q['score']:.3f}")
                print(f"   Total attempts: {q['total_attempts']}, Unique: {q['unique_attempts']}")
                print(f"   Attempts: {q['attempts']}")
                print(f"   Input preview: {q['input_preview']}")
                print()
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
