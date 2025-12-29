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


def extract_attempts(output_text: str, debug=False) -> List[str]:
    """
    Extract all attempts from the output text using simple string parsing.
    
    Args:
        output_text: The full output text containing attempt tags
        debug: If True, print debug information
        
    Returns:
        List of attempt values as strings
    """
    attempts = []
    
    # Look for attempt tags by finding all occurrences
    text = output_text
    while True:
        # Find the next attempt opening tag
        attempt_start = None
        attempt_num = None
        
        # Look for <attempt-1>, <attempt-2>, <attempt-3>, etc.
        for i in range(1, 10):  # Check up to attempt-9 (should be max 3 but being safe)
            tag_start = f"<attempt-{i}>"
            tag_end = f"</attempt-{i}>"
            
            start_pos = text.find(tag_start)
            if start_pos != -1:
                end_pos = text.find(tag_end, start_pos + len(tag_start))
                if end_pos != -1:
                    # Extract the content between tags
                    content = text[start_pos + len(tag_start):end_pos].strip()
                    attempts.append(content)
                    
                    if debug:
                        print(f"Found attempt-{i}: '{content}'")
                    
                    # Remove this attempt from text to continue searching
                    text = text[:start_pos] + text[end_pos + len(tag_end):]
                    break
        else:
            # No more attempts found
            break
    
    # Sort attempts to ensure they're in order (attempt-1, attempt-2, etc.)
    # We'll re-extract them in order to be sure
    ordered_attempts = []
    for i in range(1, 10):
        tag_start = f"<attempt-{i}>"
        tag_end = f"</attempt-{i}>"
        
        start_pos = output_text.find(tag_start)
        if start_pos != -1:
            end_pos = output_text.find(tag_end, start_pos + len(tag_start))
            if end_pos != -1:
                content = output_text[start_pos + len(tag_start):end_pos].strip()
                ordered_attempts.append(content)
    
    if debug and len(ordered_attempts) > 3:
        print(f"DEBUG: Found {len(ordered_attempts)} attempts:")
        for i, attempt in enumerate(ordered_attempts):
            print(f"  attempt-{i+1}: '{attempt}'")
        print("Full output preview:")
        print(output_text[:500] + "..." if len(output_text) > 500 else output_text)
        print("---")
    
    return ordered_attempts


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
    malformed_examples = []  # Track cases with >3 attempts
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                score = data.get('score', 0)
                output_text = data.get('output', '')
                attempts = extract_attempts(output_text, debug=(len(extract_attempts(output_text)) > 3))
                
                if attempts:  # Only process if we found attempts
                    unique_count = count_unique_attempts(attempts)
                    
                    # Track malformed cases with >3 attempts
                    if len(attempts) > 3:
                        malformed_examples.append({
                            'line_num': line_num,
                            'score': score,
                            'total_attempts': len(attempts),
                            'attempts': attempts,
                            'output_preview': output_text[:500] + '...' if len(output_text) > 500 else output_text
                        })
                    
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

    # Create results dictionary
    if partial_unique_attempt_counts or high_unique_attempt_counts:
        results = {
            'partial_score_questions': len(partial_score_questions),
            'high_score_questions': len(high_score_questions),
            'partial_unique_attempt_count_distribution': dict(partial_count_distribution),
            'high_unique_attempt_count_distribution': dict(high_count_distribution),
            'partial_average_unique_attempts': partial_avg_unique,
            'high_average_unique_attempts': high_avg_unique,
            'partial_questions_with_details': partial_score_questions[:10],
            'high_questions_with_details': high_score_questions[:10],
            'malformed_examples': malformed_examples[:10],  # First 10 malformed cases
            'total_malformed': len(malformed_examples),
            'partial_all_unique_counts': partial_unique_attempt_counts,
            'high_all_unique_counts': high_unique_attempt_counts
        }
    else:
        results = {
            'partial_score_questions': 0,
            'high_score_questions': 0,
            'partial_unique_attempt_count_distribution': {},
            'high_unique_attempt_count_distribution': {},
            'partial_average_unique_attempts': 0,
            'high_average_unique_attempts': 0,
            'partial_questions_with_details': [],
            'high_questions_with_details': [],
            'malformed_examples': [],
            'total_malformed': 0,
            'partial_all_unique_counts': [],
            'high_all_unique_counts': []
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze attempt diversity in JSONL files')
    parser.add_argument('jsonl_file', help='Path to the JSONL file to analyze')
    parser.add_argument('--output', '-o', help='Output file for detailed results (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--debug-malformed', action='store_true', help='Show examples of malformed outputs with >3 attempts')
    parser.add_argument('--save-malformed', help='Save malformed examples to a JSON file for detailed inspection')
    
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
    print(f"Total questions: {results['partial_score_questions']}")
    if results['partial_score_questions'] > 0:
        print(f"Average unique attempts per question: {results['partial_average_unique_attempts']:.2f}")
        print("\nDistribution of unique attempt counts:")
        for count, frequency in sorted(results['partial_unique_attempt_count_distribution'].items()):
            percentage = frequency / results['partial_score_questions'] * 100
            print(f"  {count} unique attempts: {frequency} questions ({percentage:.1f}%)")
    
    # High scores analysis
    print(f"\nHIGH SCORES (score >= 1):")
    print(f"Total questions: {results['high_score_questions']}")
    if results['high_score_questions'] > 0:
        print(f"Average unique attempts per question: {results['high_average_unique_attempts']:.2f}")
        print("\nDistribution of unique attempt counts:")
        for count, frequency in sorted(results['high_unique_attempt_count_distribution'].items()):
            percentage = frequency / results['high_score_questions'] * 100
            print(f"  {count} unique attempts: {frequency} questions ({percentage:.1f}%)")
    
    # Show malformed examples if any
    if results['total_malformed'] > 0:
        print(f"\nMALFORMED CASES (>3 attempts): {results['total_malformed']} total")
        if args.debug_malformed and results['malformed_examples']:
            print("\nFirst few malformed examples:")
            for i, example in enumerate(results['malformed_examples'][:3], 1):
                print(f"\n{i}. Line {example['line_num']}: Score={example['score']:.3f}")
                print(f"   Found {example['total_attempts']} attempts: {example['attempts']}")
                
                # Show the full attempt structure
                output_text = example['output_preview']
                print(f"   Attempt tags found:")
                for j in range(1, 10):
                    tag_start = f"<attempt-{j}>"
                    tag_end = f"</attempt-{j}>"
                    if tag_start in output_text and tag_end in output_text:
                        start_pos = output_text.find(tag_start)
                        end_pos = output_text.find(tag_end, start_pos)
                        if start_pos != -1 and end_pos != -1:
                            content = output_text[start_pos + len(tag_start):end_pos].strip()
                            print(f"     <attempt-{j}>{content}</attempt-{j}>")
                
                print(f"   Output preview: {example['output_preview'][:300]}...")
                print("   " + "-"*50)
    
    if args.verbose:
        if results['partial_questions_with_details']:
            print("\nFirst 10 PARTIAL SCORE questions (sample):")
            print("-" * 50)
            for i, q in enumerate(results['partial_questions_with_details'], 1):
                print(f"{i}. Line {q['line_num']}: Score={q['score']:.3f}")
                print(f"   Total attempts: {q['total_attempts']}, Unique: {q['unique_attempts']}")
                print(f"   Attempts: {q['attempts']}")
                print(f"   Input preview: {q['input_preview']}")
                print()
        
        if results['high_questions_with_details']:
            print("\nFirst 10 HIGH SCORE questions (sample):")
            print("-" * 50)
            for i, q in enumerate(results['high_questions_with_details'], 1):
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
    
    # Save malformed examples if requested
    if args.save_malformed and results['total_malformed'] > 0:
        malformed_data = {
            'total_malformed': results['total_malformed'],
            'examples': results['malformed_examples']
        }
        with open(args.save_malformed, 'w') as f:
            json.dump(malformed_data, f, indent=2)
        print(f"Malformed examples saved to {args.save_malformed}")


if __name__ == "__main__":
    main()
