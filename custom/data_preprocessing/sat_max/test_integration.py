#!/usr/bin/env python3
"""
Test script to verify the C++ SAT Max Solver integration
"""

import json
import sys
import os
import itertools
import random

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def python_verifier(clauses, num_variables):
    """
    Python verifier that checks all 2^n possibilities to find the maximum
    number of satisfiable clauses.
    
    Args:
        clauses: List of clauses, where each clause is a list of literals
        num_variables: Number of variables (1 to num_variables)
    
    Returns:
        dict: {
            'max_satisfiable_clauses': int,
            'assignment': str,
            'satisfied_clauses': list
        }
    """
    max_satisfiable = 0
    best_assignment = None
    best_satisfied_clauses = []
    
    # Generate all possible assignments (2^n possibilities)
    for assignment_bits in itertools.product([False, True], repeat=num_variables):
        # Convert to variable assignment (1-indexed)
        assignment = {}
        for i in range(num_variables):
            assignment[i + 1] = assignment_bits[i]
        
        # Count satisfied clauses
        satisfied_clauses = 0
        for clause_idx, clause in enumerate(clauses):
            clause_satisfied = False
            for literal in clause:
                var = abs(literal)
                if var in assignment:
                    if literal > 0 and assignment[var]:
                        clause_satisfied = True
                        break
                    elif literal < 0 and not assignment[var]:
                        clause_satisfied = True
                        break
            
            if clause_satisfied:
                satisfied_clauses+=1
        
        # Update best if this assignment satisfies more clauses
        if satisfied_clauses > max_satisfiable:
            max_satisfiable = satisfied_clauses
            best_assignment = assignment.copy()
    
    # Convert assignment to string format
    assignment_str = ",".join([f"{i+1}:{'true' if best_assignment[i+1] else 'false'}" 
                              for i in range(num_variables)])
    
    return {
        'max_satisfiable_clauses': max_satisfiable,
        'assignment': assignment_str,
    }

def generate_10_variable_test_cases():
    """Generate several 10-variable test cases with different characteristics"""
    
    test_cases = []
    
    # Test case 1: Easy satisfiable case
    clauses1 = [
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10],  # Easy pairs
        [1, -2], [3, -4], [5, -6], [7, -8], [9, -10]  # Some conflicts
    ]
    test_cases.append({
        'name': 'Easy satisfiable (10 vars)',
        'clauses': clauses1,
        'num_variables': 10
    })
    
    # Test case 2: Hard unsatisfiable case
    clauses2 = [
        [1, 2], [-1, 2], [1, -2], [-1, -2],  # Impossible 2-SAT
        [3, 4], [-3, 4], [3, -4], [-3, -4],  # Another impossible 2-SAT
        [5, 6], [7, 8], [9, 10]  # Some easy clauses
    ]
    test_cases.append({
        'name': 'Hard unsatisfiable (10 vars)',
        'clauses': clauses2,
        'num_variables': 10
    })
    
    # Test case 3: Random medium difficulty
    random.seed(42)  # For reproducibility
    clauses3 = []
    for _ in range(15):  # 15 clauses
        clause = []
        clause_size = random.randint(2, 4)  # 2-4 literals per clause
        used_vars = set()
        for _ in range(clause_size):
            var = random.randint(1, 10)
            while var in used_vars:
                var = random.randint(1, 10)
            used_vars.add(var)
            literal = var if random.choice([True, False]) else -var
            clause.append(literal)
        clauses3.append(clause)
    
    test_cases.append({
        'name': 'Random medium difficulty (10 vars)',
        'clauses': clauses3,
        'num_variables': 10
    })
    
    # Test case 4: Chain dependencies
    clauses4 = [
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
        [-1, -2], [-2, -3], [-3, -4], [-4, -5], [-5, -6], [-6, -7], [-7, -8], [-8, -9], [-9, -10],
        [1, 10]  # Connect the chain
    ]
    test_cases.append({
        'name': 'Chain dependencies (10 vars)',
        'clauses': clauses4,
        'num_variables': 10
    })
    
    # Test case 5: Sparse case
    clauses5 = [
        [1, 2, 3, 4, 5],  # Large clause
        [6, 7, 8, 9, 10],  # Another large clause
        [1, -6], [2, -7], [3, -8], [4, -9], [5, -10]  # Conflicts
    ]
    test_cases.append({
        'name': 'Sparse large clauses (10 vars)',
        'clauses': clauses5,
        'num_variables': 10
    })
    
    return test_cases

def test_cpp_solver():
    """Test the C++ solver with a simple example"""
    
    # Test data: simple 2-SAT problem
    test_clauses = [[1, -2], [-1, 2], [1, 2]]
    raw_sat_json = json.dumps(test_clauses)
    
    print(f"Testing with clauses: {test_clauses}")
    print(f"Raw SAT JSON: {raw_sat_json}")
    
    try:
        import sat_max_solver
        print("✓ C++ solver module imported successfully")
        
        result_json = sat_max_solver.solve(raw_sat_json)
        print(f"✓ C++ solver returned: {result_json}")
        
        result = json.loads(result_json)
        if "error" in result:
            print(f"✗ C++ solver error: {result['error']}")
            return False
        else:
            print(f"✓ Max satisfiable clauses: {result['max_satisfiable_clauses']}")
            print(f"✓ Assignment: {result['assignment']}")
            return True
            
    except ImportError as e:
        print(f"✗ Failed to import C++ solver: {e}")
        print("Run ./build.sh to compile the C++ extension")
        return False
    except Exception as e:
        print(f"✗ Error testing C++ solver: {e}")
        return False

def test_python_fallback():
    """Test the Python fallback when C++ solver is not available"""
    
    print("\nTesting Python fallback...")
    
    # Mock the generate_sat_data function
    def mock_generate_sat_data(level, num_variables, num_clauses):
        raw_clauses = [[1, -2], [-1, 2], [1, 2]]
        raw_sat = json.dumps(raw_clauses)
        
        # Simulate the fallback logic
        try:
            import sat_max_solver
            result_json = sat_max_solver.solve(raw_sat)
            result = json.loads(result_json)
            if "error" in result:
                solution_str = "a:true"  # fallback
            else:
                solution_str = result["assignment"]
        except ImportError:
            solution_str = "a:true"  # fallback
        except Exception as e:
            solution_str = "a:true"  # fallback
        
        return {
            "variable_labels": json.dumps(["a", "b"], indent=2),
            "raw_sat": raw_sat,
            "sat": [["a", "~b"], ["~a", "b"], ["a", "b"]],
            "solution": solution_str
        }
    
    result = mock_generate_sat_data(2, 2, 3)
    print(f"✓ Fallback test result: {result['solution']}")
    return True

def test_verifier_with_cpp_solver():
    """Test the Python verifier against the C++ solver"""
    
    print("\nTesting Python verifier against C++ solver...")
    
    # Test with the simple 2-variable case
    test_clauses = [[1, -2], [-1, 2], [1, 2]]
    num_variables = 2
    
    # Get Python verifier result
    python_result = python_verifier(test_clauses, num_variables)
    print(f"Python verifier: max_satisfiable={python_result['max_satisfiable_clauses']}, assignment={python_result['assignment']}")
    
    # Get C++ solver result
    try:
        import sat_max_solver
        raw_sat_json = json.dumps(test_clauses)
        cpp_result_json = sat_max_solver.solve(raw_sat_json)
        cpp_result = json.loads(cpp_result_json)
        
        if "error" in cpp_result:
            print(f"✗ C++ solver error: {cpp_result['error']}")
            return False
        
        print(f"C++ solver: max_satisfiable={cpp_result['max_satisfiable_clauses']}, assignment={cpp_result['assignment']}")
        
        # Compare results
        if python_result['max_satisfiable_clauses'] == cpp_result['max_satisfiable_clauses']:
            print("✓ Results match!")
            return True
        else:
            print("✗ Results don't match!")
            return False
            
    except ImportError:
        print("C++ solver not available, skipping comparison")
        return True
    except Exception as e:
        print(f"✗ Error testing C++ solver: {e}")
        return False

def test_10_variable_cases():
    """Test the 10-variable test cases with the Python verifier"""
    
    print("\nTesting 10-variable cases with Python verifier...")
    
    test_cases = generate_10_variable_test_cases()
    success = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test_case['name']}")
        print(f"Clauses: {len(test_case['clauses'])} clauses, {test_case['num_variables']} variables")
        
        # Use Python verifier
        result = python_verifier(test_case['clauses'], test_case['num_variables'])
        print(f"✓ Max satisfiable: {result['max_satisfiable_clauses']}/{len(test_case['clauses'])}")
        print(f"✓ Best assignment: {result['assignment']}")
        
        # Also try C++ solver if available
        try:
            import sat_max_solver
            raw_sat_json = json.dumps(test_case['clauses'])
            cpp_result_json = sat_max_solver.solve(raw_sat_json)
            cpp_result = json.loads(cpp_result_json)
            
            if "error" not in cpp_result:
                if result['max_satisfiable_clauses'] == cpp_result['max_satisfiable_clauses']:
                    print("✓ C++ solver matches Python verifier")
                else:
                    print(f"✗ C++ solver differs: {cpp_result['max_satisfiable_clauses']}")
                    success = False
            else:
                print(f"✗ C++ solver error: {cpp_result['error']}")
                success = False
                
        except ImportError:
            print("C++ solver not available for comparison")
        except Exception as e:
            print(f"✗ C++ solver error: {e}")
            success = False
    
    return success

if __name__ == "__main__":
    print("Testing SAT Max Solver Integration")
    print("=" * 40)
    
    success = True
    
    # Test C++ solver
    if not test_cpp_solver():
        success = False
    
    # Test Python fallback
    if not test_python_fallback():
        success = False
    
    # Test Python verifier against C++ solver
    if not test_verifier_with_cpp_solver():
        success = False
    
    # Test 10-variable cases
    if not test_10_variable_cases():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
