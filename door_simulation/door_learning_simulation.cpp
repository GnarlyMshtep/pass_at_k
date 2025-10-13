#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cstdlib>

class DoorLearningSimulation {
private:
    int k;
    double a1, a2, a3, a4, a5;
    double prob1, prob2, prob3, prob4, prob5;
    std::vector<double> strategy_5_probs;
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;
    std::uniform_int_distribution<> int_dis;
    
    // History tracking
    std::vector<int> iteration_history;
    std::vector<double> a1_history, a2_history, a3_history, a4_history, a5_history;
    std::string output_filename;
    
    // Strategy function pointers
    int (DoorLearningSimulation::*strategies[6])(int);
    
public:
    DoorLearningSimulation(int k_val, double a1_val, double a2_val, double a3_val, 
                          double a4_val = 0.0, double a5_val = 0.0) 
        : k(k_val), a1(a1_val), a2(a2_val), a3(a3_val), a4(a4_val), a5(a5_val),
          gen(rd()), dis(0.0, 1.0), int_dis(1, k_val) {
        
        // Normalize probabilities
        double total = a1 + a2 + a3 + a4 + a5;
        if (total > 0) {
            prob1 = a1 / total;
            prob2 = a2 / total;
            prob3 = a3 / total;
            prob4 = a4 / total;
            prob5 = a5 / total;
        } else {
            prob1 = prob2 = prob3 = prob4 = prob5 = 0.2;
        }
        
        // Initialize strategy function pointers
        strategies[1] = &DoorLearningSimulation::strategy_1;
        strategies[2] = &DoorLearningSimulation::strategy_2;
        strategies[3] = &DoorLearningSimulation::strategy_3;
        strategies[4] = &DoorLearningSimulation::strategy_4;
        strategies[5] = &DoorLearningSimulation::strategy_5;
        
        // Precompute strategy 5 probabilities
        compute_strategy_5_probs();
        
        // Generate output filename
        generate_output_filename();
    }
    
    // Strategy implementations
    int strategy_1(int attempt) { return attempt; }
    int strategy_2(int attempt) { return k + 1 - attempt; }
    int strategy_3(int attempt) { return 1; }
    int strategy_4(int attempt) { return int_dis(gen); }
    int strategy_5(int attempt) {
        std::discrete_distribution<> dist(strategy_5_probs.begin(), strategy_5_probs.end());
        return dist(gen) + 1; // +1 because doors start from 1
    }
    
    void compute_strategy_5_probs() {
        strategy_5_probs.clear();
        double total = 0.0;
        
        // Compute probabilities proportional to 1/2^i for doors 1 to k
        for (int i = 1; i <= k; i++) {
            double prob = 1.0 / (1 << i); // 1/2^i
            strategy_5_probs.push_back(prob);
            total += prob;
        }
        
        // Normalize to sum to 1
        for (double& prob : strategy_5_probs) {
            prob /= total;
        }
    }
    
    double get_reward_probability(int door) {
        return 1.0 / (1 << door); // 1/2^door
    }
    
    int choose_strategy_randomly() {
        double rand_val = dis(gen);
        double cumulative = 0;
        
        cumulative += prob1;
        if (rand_val < cumulative) return 1;
        
        cumulative += prob2;
        if (rand_val < cumulative) return 2;
        
        cumulative += prob3;
        if (rand_val < cumulative) return 3;
        
        cumulative += prob4;
        if (rand_val < cumulative) return 4;
        
        return 5;
    }
    
    double simulate_attempt(int target_strategy, int target_attempt, int num_simulations = 10000) {
        double total_reward = 0.0;
        
        for (int sim = 0; sim < num_simulations; sim++) {
            // Step 1: Randomly choose solution door according to reward probabilities
            int max_door = std::max(20, k);
            std::vector<double> door_probs;
            double total_prob = 0.0;
            
            for (int door = 1; door <= max_door; door++) {
                double prob = get_reward_probability(door);
                door_probs.push_back(prob);
                total_prob += prob;
            }
            
            // Normalize probabilities
            for (double& prob : door_probs) {
                prob /= total_prob;
            }
            
            // Choose solution door
            std::discrete_distribution<> door_dist(door_probs.begin(), door_probs.end());
            int solution_door = door_dist(gen) + 1; // +1 because doors start from 1
            
            int reward = 0;
            for (int attempt = 1; attempt <= k; attempt++) {
                // Step 2: Choose door based on strategy
                int chosen_door;
                if (attempt == target_attempt) {
                    // Use the target strategy directly
                    chosen_door = (this->*strategies[target_strategy])(attempt);
                } else {
                    // Randomly choose a strategy and sample door from that strategy
                    int chosen_strategy = choose_strategy_randomly();
                    chosen_door = (this->*strategies[chosen_strategy])(attempt);
                }
                
                // Step 3: Check if we found the solution
                if (chosen_door == solution_door) {
                    reward = 1;
                    break;
                }
            }
            total_reward += reward;
        }
        
        return total_reward / num_simulations;
    }
    
    std::map<int, double> calculate_strategy_scores(int num_simulations = 10000) {
        std::map<int, double> scores;
        
        for (int strategy_num = 1; strategy_num <= 5; strategy_num++) {
            double total_score = 0.0;
            
            for (int attempt = 1; attempt <= k; attempt++) {
                double attempt_score = simulate_attempt(strategy_num, attempt, num_simulations);
                total_score += attempt_score;
            }
            
            scores[strategy_num] = total_score;
        }
        
        return scores;
    }
    
    void update_strategy_probabilities(const std::map<int, double>& scores, double lr) {
        // Sort strategies by performance (descending)
        std::vector<std::pair<int, double>> sorted_strategies;
        for (const auto& pair : scores) {
            sorted_strategies.push_back(pair);
        }
        std::sort(sorted_strategies.begin(), sorted_strategies.end(),
                 [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                     return a.second > b.second;
                 });
        
        // Get strategy rankings
        std::map<int, int> rankings;
        for (size_t i = 0; i < sorted_strategies.size(); i++) {
            rankings[sorted_strategies[i].first] = i + 1;
        }
        
        // Update probabilities based on rankings
        std::map<int, double> updates = {{1, 2*lr}, {2, lr}, {3, 0}, {4, -lr}, {5, -2*lr}};
        
        double new_a1 = a1 + updates[rankings[1]];
        double new_a2 = a2 + updates[rankings[2]];
        double new_a3 = a3 + updates[rankings[3]];
        double new_a4 = a4 + updates[rankings[4]];
        double new_a5 = a5 + updates[rankings[5]];

        // Ensure non-negative values
        a1 = std::max(0.01, new_a1);
        a2 = std::max(0.01, new_a2);
        a3 = std::max(0.01, new_a3);
        a4 = std::max(0.01, new_a4);
        a5 = std::max(0.01, new_a5);

        // Normalize all new_a* values
        double total = a1 + a2 + a3 + a4 + a5;
        if (total > 0) {
            a1 = a1 / total;
            a2 = a2 / total;
            a3 = a3 / total;
            a4 = a4 / total;
            a5 = a5 / total;

            prob1 = a1;
            prob2 = a2;
            prob3 = a3;
            prob4 = a4;
            prob5 = a5;
        } else {
            // If total is 0, reset to uniform distribution
            a1 = a2 = a3 = a4 = a5 = 0.2;
            prob1 = prob2 = prob3 = prob4 = prob5 = 0.2;
        }
    }
    
    void generate_output_filename() {
        std::time_t now = std::time(0);
        std::stringstream ss;
        ss << "learning_history_k" << k << "_lr0_02_a1" << std::fixed << std::setprecision(3) 
           << a1 << "_a2" << a2 << "_a3" << a3 << "_a4" << a4 << "_a5" << a5 
           << "_" << now << ".txt";
        output_filename = ss.str();
    }
    
    void save_history_to_file() {
        std::ofstream file(output_filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << output_filename << " for writing." << std::endl;
            return;
        }
        
        // Write header
        file << "# Door Learning Simulation History" << std::endl;
        file << "# k=" << k << ", lr=0.02" << std::endl;
        file << "# Initial probabilities: a1=" << a1_history[0] << ", a2=" << a2_history[0] 
             << ", a3=" << a3_history[0] << ", a4=" << a4_history[0] << ", a5=" << a5_history[0] << std::endl;
        file << "# Format: iteration a1 a2 a3 a4 a5" << std::endl;
        
        // Write data
        for (size_t i = 0; i < iteration_history.size(); i++) {
            file << iteration_history[i] << " " 
                 << std::fixed << std::setprecision(6)
                 << a1_history[i] << " " << a2_history[i] << " " 
                 << a3_history[i] << " " << a4_history[i] << " " 
                 << a5_history[i] << std::endl;
        }
        
        file.close();
        std::cout << "\nHistory saved to: " << output_filename << std::endl;
    }
    
    void learn(int num_iterations, double lr, int num_simulations = 10000) {
        std::cout << "Starting learning with " << num_iterations << " iterations, lr=" << lr << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Record initial values
        iteration_history.push_back(0);
        a1_history.push_back(a1);
        a2_history.push_back(a2);
        a3_history.push_back(a3);
        a4_history.push_back(a4);
        a5_history.push_back(a5);
        
        for (int iteration = 0; iteration < num_iterations; iteration++) {
            // Calculate current strategy scores
            auto scores = calculate_strategy_scores(num_simulations);
            
            // Print iteration summary
            std::cout << "Iteration " << (iteration + 1) << ":" << std::endl;
            std::cout << "  Strategy probabilities: a1=" << std::fixed << std::setprecision(4) 
                      << a1 << ", a2=" << a2 << ", a3=" << a3 << ", a4=" << a4 << ", a5=" << a5 << std::endl;
            std::cout << "  Strategy scores: ";
            for (int i = 1; i <= 5; i++) {
                std::cout << std::fixed << std::setprecision(4) << scores[i];
                if (i < 5) std::cout << ", ";
            }
            std::cout << std::endl;
            
            // Sort and show ranking
            std::vector<std::pair<int, double>> sorted_strategies;
            for (const auto& pair : scores) {
                sorted_strategies.push_back(pair);
            }
            std::sort(sorted_strategies.begin(), sorted_strategies.end(),
                     [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                         return a.second > b.second;
                     });
            
            std::cout << "  Ranking: ";
            for (size_t i = 0; i < sorted_strategies.size(); i++) {
                std::cout << "S" << sorted_strategies[i].first << "(" 
                          << std::fixed << std::setprecision(4) << sorted_strategies[i].second << ")";
                if (i < sorted_strategies.size() - 1) std::cout << " > ";
            }
            std::cout << std::endl << std::endl;
            
            // Update probabilities
            update_strategy_probabilities(scores, lr);
            
            // Record values after update
            iteration_history.push_back(iteration + 1);
            a1_history.push_back(a1);
            a2_history.push_back(a2);
            a3_history.push_back(a3);
            a4_history.push_back(a4);
            a5_history.push_back(a5);
        }
        
        // Final summary
        auto final_scores = calculate_strategy_scores(num_simulations);
        std::cout << "Final Results:" << std::endl;
        std::cout << std::string(20, '-') << std::endl;
        std::cout << "Final strategy probabilities: a1=" << std::fixed << std::setprecision(4) 
                  << a1 << ", a2=" << a2 << ", a3=" << a3 << ", a4=" << a4 << ", a5=" << a5 << std::endl;
        std::cout << "Final strategy scores: ";
        for (int i = 1; i <= 5; i++) {
            std::cout << std::fixed << std::setprecision(4) << final_scores[i];
            if (i < 5) std::cout << ", ";
        }
        std::cout << std::endl;
        
        std::vector<std::pair<int, double>> final_sorted;
        for (const auto& pair : final_scores) {
            final_sorted.push_back(pair);
        }
        std::sort(final_sorted.begin(), final_sorted.end(),
                 [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                     return a.second > b.second;
                 });
        
        std::cout << "Final ranking: ";
        for (size_t i = 0; i < final_sorted.size(); i++) {
            std::cout << "S" << final_sorted[i].first << "(" 
                      << std::fixed << std::setprecision(4) << final_sorted[i].second << ")";
            if (i < final_sorted.size() - 1) std::cout << " > ";
        }
        std::cout << std::endl;
        
        // Save history to file
        save_history_to_file();
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --k <value>              Number of chances/attempts (default: 5)" << std::endl;
    std::cout << "  --a <a1> <a2> <a3> <a4> <a5>  Initial strategy probabilities (default: 0.5 0.4 0.05 0.02 0.03)" << std::endl;
    std::cout << "  --lr <value>             Learning rate (default: 0.02)" << std::endl;
    std::cout << "  --iterations <value>     Number of learning iterations (default: 100)" << std::endl;
    std::cout << "  --simulations <value>    Number of simulations per iteration (default: 10000)" << std::endl;
    std::cout << "  --help                   Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default values
    int k = 5;
    std::vector<double> a = {0.5, 0.4, 0.05, 0.02, 0.03};
    double learning_rate = 0.02;
    int num_iterations = 100;
    int num_simulations = 10000;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--k" && i + 1 < argc) {
            k = std::stoi(argv[++i]);
        } else if (arg == "--a" && i + 5 < argc) {
            a[0] = std::stod(argv[++i]);
            a[1] = std::stod(argv[++i]);
            a[2] = std::stod(argv[++i]);
            a[3] = std::stod(argv[++i]);
            a[4] = std::stod(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            learning_rate = std::stod(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            num_iterations = std::stoi(argv[++i]);
        } else if (arg == "--simulations" && i + 1 < argc) {
            num_simulations = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate arguments
    if (k <= 0) {
        std::cerr << "Error: k must be a positive integer" << std::endl;
        return 1;
    }
    if (a.size() != 5) {
        std::cerr << "Error: Strategy probabilities array must contain exactly 5 values" << std::endl;
        return 1;
    }
    for (double prob : a) {
        if (prob < 0) {
            std::cerr << "Error: All strategy probabilities must be non-negative" << std::endl;
            return 1;
        }
    }
    if (learning_rate <= 0) {
        std::cerr << "Error: Learning rate must be positive" << std::endl;
        return 1;
    }
    if (num_iterations <= 0) {
        std::cerr << "Error: Number of iterations must be positive" << std::endl;
        return 1;
    }
    if (num_simulations <= 0) {
        std::cerr << "Error: Number of simulations must be positive" << std::endl;
        return 1;
    }
    
    std::cout << "Door Learning Algorithm Simulation" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  k (chances): " << k << std::endl;
    std::cout << "  Initial strategy probabilities: a=[" << a[0] << ", " << a[1] << ", " 
              << a[2] << ", " << a[3] << ", " << a[4] << "]" << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Number of iterations: " << num_iterations << std::endl;
    std::cout << "  Simulations per iteration: " << num_simulations << std::endl;
    std::cout << std::endl;
    
    std::cout << "Strategy Descriptions:" << std::endl;
    std::cout << "  Strategy 1: Choose door i on attempt i" << std::endl;
    std::cout << "  Strategy 2: Choose door k+1-i on attempt i" << std::endl;
    std::cout << "  Strategy 3: Always choose door 1" << std::endl;
    std::cout << "  Strategy 4: Random door between 1 and k" << std::endl;
    std::cout << "  Strategy 5: Random door with prob proportional to 1/2^i" << std::endl;
    std::cout << std::endl;
    
    // Create simulation
    DoorLearningSimulation sim(k, a[0], a[1], a[2], a[3], a[4]);
    
    // Run learning algorithm
    sim.learn(num_iterations, learning_rate, num_simulations);
    
    return 0;
}
