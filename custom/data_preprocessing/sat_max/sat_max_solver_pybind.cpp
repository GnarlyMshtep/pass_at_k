#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <cstdint>

class SATMaxSolver {
public:
    struct Clause {
        std::vector<int> literals;  // positive for variable, negative for negation
    };
    
    struct OptimizedClause {
        uint64_t variable_mask;  // bitmask of involved variables (1-based indexing)
        uint64_t xor_pattern;    // bitmask for negative literals (1-based indexing)
    };
    
    struct Solution {
        int max_satisfiable_clauses;
        std::vector<bool> assignment;  // true for variable i means variable i+1 is true
        std::string assignment_str;    // formatted as "a:true,b:false,..."
    };
    
    // Parse input from Python
    static std::vector<Clause> parseClauses(const std::string& raw_clauses_json) {
        std::vector<Clause> clauses;
        
        // Simple JSON parsing for the raw_clauses format
        // Expected format: [[1, -2, 3], [-1, 2, -3], ...]
        std::string input = raw_clauses_json;
        
        // Remove outer brackets
        if (input.front() == '[' && input.back() == ']') {
            input = input.substr(1, input.length() - 2);
        }
        
        size_t pos = 0;
        while (pos < input.length()) {
            // Find the next clause start '['
            size_t clause_start = input.find('[', pos);
            if (clause_start == std::string::npos) break;
            
            // Find the matching ']'
            size_t bracket_count = 0;
            size_t clause_end = clause_start;
            for (size_t i = clause_start; i < input.length(); ++i) {
                if (input[i] == '[') {
                    bracket_count++;
                } else if (input[i] == ']') {
                    bracket_count--;
                    if (bracket_count == 0) {
                        clause_end = i;
                        break;
                    }
                }
            }
            
            if (clause_end == clause_start) {
                pos = clause_start + 1;
                continue;
            }
            
            // Extract clause content (between the brackets)
            std::string clause_content = input.substr(clause_start + 1, clause_end - clause_start - 1);
            
            // Parse literals from clause content
            Clause clause;
            std::stringstream clause_ss(clause_content);
            std::string literal_str;
            
            while (std::getline(clause_ss, literal_str, ',')) {
                // Remove whitespace
                literal_str.erase(std::remove_if(literal_str.begin(), literal_str.end(), ::isspace), literal_str.end());
                
                if (!literal_str.empty()) {
                    try {
                        int literal = std::stoi(literal_str);
                        clause.literals.push_back(literal);
                    } catch (const std::exception& e) {
                        // Skip invalid literals
                    }
                }
            }
            
            if (!clause.literals.empty()) {
                clauses.push_back(clause);
            }
            
            // Move to after this clause
            pos = clause_end + 1;
            
            // Skip comma if present
            if (pos < input.length() && input[pos] == ',') {
                pos++;
            }
        }
        
        return clauses;
    }
    
    // Convert clauses to optimized bitmask representation
    static std::vector<OptimizedClause> optimizeClauses(const std::vector<Clause>& clauses) {
        std::vector<OptimizedClause> optimized;
        
        for (const auto& clause : clauses) {
            OptimizedClause opt_clause;
            opt_clause.variable_mask = 0;
            opt_clause.xor_pattern = 0;
            
            for (int literal : clause.literals) {
                int var = std::abs(literal);
                if (var > 0 && var <= 60) {  // Support up to 60 variables
                    // Set bit for this variable (1-based indexing)
                    opt_clause.variable_mask |= (1ULL << (var - 1));
                    
                    // If literal is negative, set the corresponding bit in xor_pattern
                    if (literal < 0) {
                        opt_clause.xor_pattern |= (1ULL << (var - 1));
                    }
                }
            }
            
            optimized.push_back(opt_clause);
        }
        
        return optimized;
    }
    
    // Check if a clause is satisfied by a given assignment
    static bool isClauseSatisfied(const OptimizedClause& clause, uint64_t assignment) {
        // Extract only the variables involved in this clause
        uint64_t relevant_vars = assignment & clause.variable_mask;
        
        // Apply XOR pattern to handle negative literals
        uint64_t result = relevant_vars ^ clause.xor_pattern;
        
        // Check if at least one literal is true (any bit set in result)
        return result != 0;
    }
    
    // Efficient brute force solver using bitmasks
    static Solution solveMaxSAT(const std::vector<Clause>& clauses) {
        Solution solution;
        
        if (clauses.empty()) {
            solution.max_satisfiable_clauses = 0;
            solution.assignment_str = "";
            return solution;
        }
        
        // Find the maximum variable number
        int max_var = 0;
        for (const auto& clause : clauses) {
            for (int literal : clause.literals) {
                max_var = std::max(max_var, std::abs(literal));
            }
        }
        
        // Check if we can handle this many variables
        if (max_var > 60) {
            // Fallback to original approach for >60 variables
            solution.max_satisfiable_clauses = 0;
            solution.assignment_str = "error:too_many_variables";
            return solution;
        }
        
        // Convert clauses to optimized bitmask representation
        std::vector<OptimizedClause> optimized_clauses = optimizeClauses(clauses);
        
        // Initialize solution
        solution.max_satisfiable_clauses = 0;
        solution.assignment.resize(max_var, false);
        uint64_t best_assignment = 0;
        
        // Try all possible assignments (2^max_var possibilities)
        uint64_t total_assignments = 1ULL << max_var;
        for (uint64_t assignment = 0; assignment < total_assignments; ++assignment) {
            int satisfied_clauses = 0;
            
            // Count satisfied clauses for this assignment
            for (size_t i = 0; i < optimized_clauses.size(); ++i) {
                if (isClauseSatisfied(optimized_clauses[i], assignment)) {
                    satisfied_clauses++;
                }
            }
            
            // Update best solution if this assignment is better
            if (satisfied_clauses > solution.max_satisfiable_clauses) {
                solution.max_satisfiable_clauses = satisfied_clauses;
                best_assignment = assignment;
            }
        }
        
        // Convert bitmask assignment to boolean vector (1-based indexing)
        for (int i = 0; i < max_var; ++i) {
            solution.assignment[i] = (best_assignment & (1ULL << i)) != 0;
        }
        
        // Format assignment string
        std::stringstream assignment_ss;
        for (int i = 0; i < max_var; ++i) {
            if (i > 0) assignment_ss << ",";
            char var_name = 'a' + i;
            assignment_ss << var_name << ":" << (solution.assignment[i] ? "true" : "false");
        }
        solution.assignment_str = assignment_ss.str();
        
        return solution;
    }
    
    // Main function to be called from Python
    static std::string solve(const std::string& raw_clauses_json) {
        try {
            std::vector<Clause> clauses = parseClauses(raw_clauses_json);
            Solution solution = solveMaxSAT(clauses);
            
            // Return result as JSON-like string
            std::stringstream result;
            result << "{\"max_satisfiable_clauses\": " << solution.max_satisfiable_clauses
                   << ", \"assignment\": \"" << solution.assignment_str << "\"}";
            
            return result.str();
        } catch (const std::exception& e) {
            return "{\"error\": \"" + std::string(e.what()) + "\"}";
        }
    }
};

// Python bindings
namespace py = pybind11;

PYBIND11_MODULE(sat_max_solver, m) {
    m.doc() = "SAT Max Solver - Find maximum number of satisfiable clauses";
    
    m.def("solve", &SATMaxSolver::solve, 
          "Solve MAX-SAT problem",
          py::arg("raw_clauses_json"));
    
    m.def("parse_clauses", [](const std::string& raw_clauses_json) {
        auto clauses = SATMaxSolver::parseClauses(raw_clauses_json);
        std::vector<std::vector<int>> result;
        for (const auto& clause : clauses) {
            result.push_back(clause.literals);
        }
        return result;
    }, "Parse clauses from JSON string");
}
