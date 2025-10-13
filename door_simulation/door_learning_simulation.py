import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random
import time
import argparse

class DoorLearningSimulation:
    """
    Simulation for analyzing learning algorithms with infinite doors.
    
    Problem setup:
    - Infinite doors with reward behind door i with probability 1/2^i
    - k chances to choose doors
    - Five strategies to analyze:
      1. Choose door i on attempt i
      2. Choose door k+1-i on attempt i
      3. Always choose door 1
      4. Random door between 1 and k
      5. Random door with probability proportional to 1/2^i
    """
    
    def __init__(self, k: int, a1: float, a2: float, a3: float, a4: float = 0.0, a5: float = 0.0):
        """
        Initialize the simulation.
        
        Args:
            k: Number of chances/attempts
            a1, a2, a3, a4, a5: Current probabilities for strategies 1, 2, 3, 4, 5
        """
        self.k = k
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        
        # Normalize probabilities
        total = a1 + a2 + a3 + a4 + a5
        if total > 0:
            self.a1 = self.prob1 = a1 / total
            self.a2 = self.prob2 = a2 / total
            self.a3 = self.prob3 = a3 / total
            self.a4 = self.prob4 = a4 / total
            self.a5 = self.prob5 = a5 / total
        else:
            self.a1 = self.prob1 = self.a2 = self.prob2 = self.a3 = self.prob3 = self.a4 = self.prob4 = self.a5 = self.prob5 = 0.2
        
        # Strategy definitions
        self.strategies = {
            1: self.strategy_1,  # Choose door i on attempt i
            2: self.strategy_2,  # Choose door k+1-i on attempt i
            3: self.strategy_3,  # Always choose door 1
            4: self.strategy_4,  # Random door between 1 and k
            5: self.strategy_5   # Random door with prob proportional to 1/2^i
        }
        
        # Precompute strategy 5 probabilities
        self._compute_strategy_5_probs()
        
        # Initialize tracking for plotting
        self.iteration_history = []
        self.a1_history = []
        self.a2_history = []
        self.a3_history = []
        self.a4_history = []
        self.a5_history = []
        self.start_time = None
    
    def strategy_1(self, attempt: int) -> int:
        """Strategy 1: Choose door i on attempt i"""
        return attempt
    
    def strategy_2(self, attempt: int) -> int:
        """Strategy 2: Choose door k+1-i on attempt i"""
        return self.k + 1 - attempt
    
    def strategy_3(self, attempt: int) -> int:
        """Strategy 3: Always choose door 1"""
        return 1
    
    def strategy_4(self, attempt: int) -> int:
        """Strategy 4: Random door between 1 and k"""
        return random.randint(1, self.k)
    
    def strategy_5(self, attempt: int) -> int:
        """Strategy 5: Random door with probability proportional to 1/2^i"""
        return np.random.choice(range(1, self.k + 1), p=self.strategy_5_probs)
    
    def _compute_strategy_5_probs(self):
        """Precompute probabilities for strategy 5"""
        # Compute probabilities proportional to 1/2^i for doors 1 to k
        probs = [1 / (2 ** i) for i in range(1, self.k + 1)]
        # Normalize to sum to 1
        total = sum(probs)
        self.strategy_5_probs = [p / total for p in probs]
    
    def get_reward_probability(self, door: int) -> float:
        """Get the probability of reward behind door i"""
        return 1 / (2 ** door)
    
    def choose_strategy_randomly(self) -> int:
        """Choose a strategy randomly based on current probabilities"""
        rand = random.random()
        cumulative = 0
        
        if rand < (cumulative := cumulative + self.prob1):
            return 1
        elif rand < (cumulative := cumulative + self.prob2):
            return 2
        elif rand < (cumulative := cumulative + self.prob3):
            return 3
        elif rand < (cumulative := cumulative + self.prob4):
            return 4
        else:
            return 5
    
    def simulate_attempt(self, target_strategy: int, traget_attempt: int, num_simulations: int = 10000) -> float:
        """
        Simulate one attempt for a target strategy.
        
        For each simulation:
        1. First randomly choose a solution door according to reward probabilities
        2. For the given attempt, if it's the target strategy, use that strategy
        3. Otherwise, randomly choose a strategy and sample door from that strategy
        
        Args:
            target_strategy: The strategy we're evaluating (1, 2, 3, 4, or 5)
            traget_attempt: The traget attempt number (1 to k)
            num_simulations: Number of simulations
            
        Returns:
            Expected reward for this strategy and attempt
        """
        total_reward = 0
        
        for _ in range(num_simulations):
            # Step 1: Randomly choose solution door according to reward probabilities
            # We need to choose from all possible doors (1 to some reasonable upper bound)
            max_door = 20  # At least k, but allow higher doors
            door_probs = [self.get_reward_probability(door) for door in range(1, max_door + 1)]
            # Normalize probabilities
            total_prob = sum(door_probs)
            normalized_probs = [p / total_prob for p in door_probs]
            
            solution_door = np.random.choice(range(1, max_door + 1), p=normalized_probs)
            reward = 0
            for attempt in range(1, self.k + 1):
                # Step 2: For the given attempt, choose door based on strategy
                if attempt == traget_attempt:
                    # Use the target strategy directly
                    chosen_door = self.strategies[target_strategy](attempt)
                else:
                    # Randomly choose a strategy and sample door from that strategy
                    chosen_strategy = self.choose_strategy_randomly()
                    chosen_door = self.strategies[chosen_strategy](attempt)
                
                # Step 3: Check if we found the solution
                reward = 1 if chosen_door == solution_door else reward
                if reward == 1:
                    break
            total_reward += reward
        
        return total_reward / num_simulations
    
    def calculate_strategy_scores(self, num_simulations: int = 10000) -> Dict[int, float]:
        """
        Calculate the total expected score for each strategy.
        
        Returns:
            Dictionary mapping strategy number to total expected score
        """
        scores = {}
        
        for strategy_num in [1, 2, 3, 4, 5]:
            total_score = 0
            
            for attempt in range(1, self.k + 1):
                attempt_score = self.simulate_attempt(strategy_num, attempt, num_simulations)
                total_score += attempt_score
            
            scores[strategy_num] = total_score
        
        return scores
    
    def update_strategy_probabilities(self, scores: Dict[int, float], lr: float):
        """
        Update strategy probabilities based on performance.
        
        Args:
            scores: Dictionary mapping strategy number to score
            lr: Learning rate
        """
        # Sort strategies by performance (descending)
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get strategy rankings (1st, 2nd, 3rd, 4th, 5th best)
        rankings = {}
        for rank, (strategy, score) in enumerate(sorted_strategies, 1):
            rankings[strategy] = rank
        
        # Update probabilities based on rankings
        # 1st place: +2lr, 2nd place: +lr, 3rd place: no change, 4th place: -lr, 5th place: -2lr
        updates = {1: 2*lr, 2: lr, 3: 0, 4: -lr, 5: -2*lr}
        
        new_a1 = self.a1 + updates[rankings[1]]
        new_a2 = self.a2 + updates[rankings[2]]
        new_a3 = self.a3 + updates[rankings[3]]
        new_a4 = self.a4 + updates[rankings[4]]
        new_a5 = self.a5 + updates[rankings[5]]
        
        # Ensure probabilities stay non-negative
        new_a1 = max(0.01, new_a1)
        new_a2 = max(0.01, new_a2)
        new_a3 = max(0.01, new_a3)
        new_a4 = max(0.01, new_a4)
        new_a5 = max(0.01, new_a5)
        
        # Update the instance variables
        self.a1 = new_a1
        self.a2 = new_a2
        self.a3 = new_a3
        self.a4 = new_a4
        self.a5 = new_a5
        
        # Re-normalize probabilities
        total = self.a1 + self.a2 + self.a3 + self.a4 + self.a5
        if total > 0:
            self.a1 =self.prob1 = self.a1 / total
            self.a2 = self.prob2 = self.a2 / total
            self.a3 = self.prob3 = self.a3 / total
            self.a4 = self.prob4 = self.a4 / total
            self.a5 = self.prob5 = self.a5 / total
        else:
            # If all probabilities are 0, reset to uniform
            self.a1 = self.prob1 = self.a2 = self.prob2 = self.a3 = self.prob3 = self.a4 = self.prob4 = self.a5 = self.prob5 = 0.2
    
    def learn(self, num_iterations: int, lr: float, num_simulations: int = 10000):
        """
        Run the learning algorithm for multiple iterations.
        
        Args:
            num_iterations: Number of learning iterations
            lr: Learning rate
            num_simulations: Number of simulations per iteration
        """
        # Record start time
        self.start_time = time.time()
        
        print(f"Starting learning with {num_iterations} iterations, lr={lr}")
        print("=" * 60)
        
        # Record initial values
        self.iteration_history.append(0)
        self.a1_history.append(self.a1)
        self.a2_history.append(self.a2)
        self.a3_history.append(self.a3)
        self.a4_history.append(self.a4)
        self.a5_history.append(self.a5)
        
        for iteration in range(num_iterations):
            # Calculate current strategy scores
            scores = self.calculate_strategy_scores(num_simulations)
            
            # Print iteration summary
            print(f"Iteration {iteration + 1}:")
            print(f"  Strategy probabilities: a1={self.a1:.4f}, a2={self.a2:.4f}, a3={self.a3:.4f}, a4={self.a4:.4f}, a5={self.a5:.4f}")
            print(f"  Strategy scores: {[f'{scores[i]:.4f}' for i in [1,2,3,4,5]]}")
            
            # Sort and show ranking
            sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranking_str = " > ".join([f"S{strategy}({score:.4f})" for strategy, score in sorted_strategies])
            print(f"  Ranking: {ranking_str}")
            
            # Update probabilities
            self.update_strategy_probabilities(scores, lr)
            
            # Record values after update
            self.iteration_history.append(iteration + 1)
            self.a1_history.append(self.a1)
            self.a2_history.append(self.a2)
            self.a3_history.append(self.a3)
            self.a4_history.append(self.a4)
            self.a5_history.append(self.a5)
            print()
        
        # Final summary
        final_scores = self.calculate_strategy_scores(num_simulations)
        print("Final Results:")
        print("-" * 20)
        print(f"Final strategy probabilities: a1={self.a1:.4f}, a2={self.a2:.4f}, a3={self.a3:.4f}, a4={self.a4:.4f}, a5={self.a5:.4f}")
        print(f"Final strategy scores: {[f'{final_scores[i]:.4f}' for i in [1,2,3,4,5]]}")
        
        final_sorted = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        final_ranking = " > ".join([f"S{strategy}({score:.4f})" for strategy, score in final_sorted])
        print(f"Final ranking: {final_ranking}")
        
        # Create and save the plot
        self.plot_learning_evolution()
    
    def plot_learning_evolution(self):
        """
        Create and save a plot showing the evolution of strategy probabilities over time.
        """
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Define fixed colors for each strategy
        colors = {
            1: '#1f77b4',  # Blue
            2: '#ff7f0e',  # Orange  
            3: '#2ca02c',  # Green
            4: '#d62728',  # Red
            5: '#9467bd'   # Purple
        }
        
        # Plot each strategy's probability evolution
        plt.plot(self.iteration_history, self.a1_history, 'o-', color=colors[1], 
                label='Strategy 1 (door i on attempt i)', linewidth=2, markersize=4)
        plt.plot(self.iteration_history, self.a2_history, 's-', color=colors[2], 
                label='Strategy 2 (door k+1-i on attempt i)', linewidth=2, markersize=4)
        plt.plot(self.iteration_history, self.a3_history, '^-', color=colors[3], 
                label='Strategy 3 (always door 1)', linewidth=2, markersize=4)
        plt.plot(self.iteration_history, self.a4_history, 'v-', color=colors[4], 
                label='Strategy 4 (random door 1-k)', linewidth=2, markersize=4)
        plt.plot(self.iteration_history, self.a5_history, 'd-', color=colors[5], 
                label='Strategy 5 (prob ∝ 1/2^i)', linewidth=2, markersize=4)
        
        # Customize the plot
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Strategy Probability (a_i)', fontsize=12)
        plt.title(f'Learning Evolution of Strategy Probabilities\n'
                 f'k={self.k}, lr={0.02}, Total Time: {elapsed_time:.2f}s', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(self.iteration_history))
        plt.ylim(0, max(max(self.a1_history), max(self.a2_history), max(self.a3_history), 
                       max(self.a4_history), max(self.a5_history)) * 1.1)
        
        # Add timing information
        plt.figtext(0.02, 0.02, f'Simulation started at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))}\n'
                                f'Total simulation time: {elapsed_time:.2f} seconds', 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save the plot
        filename = f'learning_evolution_k{self.k}_lr{0.02}_a1{self.a1_history[0]:.3f}_a2{self.a2_history[0]:.3f}_a3{self.a3_history[0]:.3f}_a4{self.a4_history[0]:.3f}_a5{self.a5_history[0]:.3f}_{int(time.time())}.png'.replace('.', '_')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {filename}")


def main():
    """Main function to run the learning simulation"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Door Learning Algorithm Simulation')
    parser.add_argument('--k', type=int, default=5, help='Number of chances/attempts (default: 5)')
    parser.add_argument('--a', type=float, nargs=5, default=[0.5, 0.4, 0.05, 0.02, 0.03], 
                       help='Initial strategy probabilities [a1, a2, a3, a4, a5] (default: [0.5, 0.4, 0.05, 0.02, 0.03])')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate (default: 0.02)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of learning iterations (default: 100)')
    parser.add_argument('--simulations', type=int, default=10000, help='Number of simulations per iteration (default: 10000)')
    
    args = parser.parse_args()
    
    # Extract parameters from arguments
    k = args.k
    a1, a2, a3, a4, a5 = args.a
    learning_rate = args.lr
    num_iterations = args.iterations
    num_simulations = args.simulations
    
    # Validate arguments
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if len(args.a) != 5:
        raise ValueError("Strategy probabilities array must contain exactly 5 values")
    if any(prob < 0 for prob in [a1, a2, a3, a4, a5]):
        raise ValueError("All strategy probabilities must be non-negative")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if num_iterations <= 0:
        raise ValueError("Number of iterations must be positive")
    if num_simulations <= 0:
        raise ValueError("Number of simulations must be positive")
    
    print("Door Learning Algorithm Simulation")
    print("=" * 50)
    print(f"Parameters:")
    print(f"  k (chances): {k}")
    print(f"  Initial strategy probabilities: a={[a1, a2, a3, a4, a5]}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of iterations: {num_iterations}")
    print(f"  Simulations per iteration: {num_simulations}")
    print()
    
    print("Strategy Descriptions:")
    print("  Strategy 1: Choose door i on attempt i")
    print("  Strategy 2: Choose door k+1-i on attempt i") 
    print("  Strategy 3: Always choose door 1")
    print("  Strategy 4: Random door between 1 and k")
    print("  Strategy 5: Random door with prob proportional to 1/2^i")
    print()
    
    # Create simulation
    sim = DoorLearningSimulation(k, a1, a2, a3, a4, a5)
    
    # Run learning algorithm
    sim.learn(num_iterations, learning_rate, num_simulations=num_simulations)


if __name__ == "__main__":
    main()
