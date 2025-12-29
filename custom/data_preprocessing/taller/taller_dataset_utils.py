import random
import string
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, deque, Counter
from typing import Set, List, Tuple, Dict
import argparse
import numpy as np
import statistics


class DAG:
    def __init__(self):
        self.vertices = set()
        self.edges = defaultdict(set)  # adjacency list: vertex -> set of vertices it points to
        self.in_edges = defaultdict(set)  # reverse adjacency: vertex -> set of vertices pointing to it
    
    def add_vertex(self, vertex):
        self.vertices.add(vertex)
    
    def add_edge(self, from_vertex, to_vertex):
        if from_vertex not in self.vertices:
            self.add_vertex(from_vertex)
        if to_vertex not in self.vertices:
            self.add_vertex(to_vertex)
        
        self.edges[from_vertex].add(to_vertex)
        self.in_edges[to_vertex].add(from_vertex)
    
    def remove_vertex(self, vertex):
        if vertex not in self.vertices:
            return
        
        # Remove all edges involving this vertex
        for v in self.edges[vertex].copy():
            self.in_edges[v].discard(vertex)
        for v in self.in_edges[vertex].copy():
            self.edges[v].discard(vertex)
        
        del self.edges[vertex]
        del self.in_edges[vertex]
        self.vertices.discard(vertex)
    
    def has_edge(self, from_vertex, to_vertex):
        return to_vertex in self.edges[from_vertex]
    
    def remove_edge(self, from_vertex, to_vertex):
        if from_vertex in self.edges:
            self.edges[from_vertex].discard(to_vertex)
        if to_vertex in self.in_edges:
            self.in_edges[to_vertex].discard(from_vertex)
    
    def get_out_degree(self, vertex):
        return len(self.edges[vertex])
    
    def get_in_degree(self, vertex):
        return len(self.in_edges[vertex])
    
    def get_total_degree(self, vertex):
        return self.get_out_degree(vertex) + self.get_in_degree(vertex)


def would_create_cycle(graph: DAG, from_vertex, to_vertex) -> bool:
    # Check if adding edge from_vertex -> to_vertex would create a cycle
    # This happens if there's already a path from to_vertex to from_vertex
    if from_vertex == to_vertex:
        return True
    
    visited = set()
    stack = [to_vertex]
    
    while stack:
        current = stack.pop()
        if current == from_vertex:
            return True
        if current in visited:
            continue
        visited.add(current)
        stack.extend(graph.edges[current])
    
    return False


def count_vertices_with_path_to(graph: DAG, target_vertex) -> int:
    # Count how many vertices have a path to target_vertex
    visited = set()
    stack = list(graph.in_edges[target_vertex])
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for predecessor in graph.in_edges[current]:
            if predecessor not in visited:
                stack.append(predecessor)
    
    return len(visited)


def _return_plausibly_third_tallest(graph: DAG) -> Set[str]:
    plausible = set()
    
    for vertex in graph.vertices:
        c = count_vertices_with_path_to(graph, vertex)
        
        # If already has exactly 2 paths, it's plausible
        if c == 2:
            plausible.add(vertex)
            continue
        
        # Try adding single edges to see if we can get count = 2
        found_single_edge = False
        for u in graph.vertices:
            if u != vertex and not graph.has_edge(u, vertex) and not would_create_cycle(graph, u, vertex):
                # Temporarily add edge and check count
                graph.add_edge(u, vertex)
                new_count = count_vertices_with_path_to(graph, vertex)
                graph.remove_edge(u, vertex)  # Remove the temporary edge
                
                if new_count == 2:
                    plausible.add(vertex)
                    found_single_edge = True
                    break
        
        # If single edge didn't work, try pairs of edges
        if not found_single_edge:
            vertices_list = list(graph.vertices)
            for i in range(len(vertices_list)):
                for j in range(i + 1, len(vertices_list)):
                    u, w = vertices_list[i], vertices_list[j]
                    
                    # Check if both edges can be added without cycles
                    if (u != vertex and w != vertex and 
                        not graph.has_edge(u, vertex) and not graph.has_edge(w, vertex) and
                        not would_create_cycle(graph, u, vertex) and not would_create_cycle(graph, w, vertex)):
                        
                        # Check if adding both edges simultaneously would create cycles
                        # First add u -> vertex
                        graph.add_edge(u, vertex)
                        if not would_create_cycle(graph, w, vertex):
                            # Now add w -> vertex and check count
                            graph.add_edge(w, vertex)
                            new_count = count_vertices_with_path_to(graph, vertex)
                            graph.remove_edge(w, vertex)  # Remove second edge
                            
                            if new_count == 2:
                                graph.remove_edge(u, vertex)  # Remove first edge
                                plausible.add(vertex)
                                break
                        
                        graph.remove_edge(u, vertex)  # Remove first edge
                
                if vertex in plausible:
                    break
    
    return plausible


def generate_instance_graph(n_vertex: int, max_n_sol: int) -> Tuple[DAG, Set[str], List[Tuple[str, str]]]:
    # Sample n_same
    n_same = random.randint(1, min(n_vertex // 3, 1))
    
    # Generate vertices with random capital letters
    available_letters = list(string.ascii_uppercase)
    random.shuffle(available_letters)
    vertices = available_letters[:n_vertex]
    
    # Create graph
    graph = DAG()
    for v in vertices:
        graph.add_vertex(v)
    
    # Pick first vertex and give it outgoing edges to all others
    first_vertex = vertices[0]
    for v in vertices[1:]:
        graph.add_edge(first_vertex, v)
    
    # Continue adding edges until plausibly third tallest <= max_n_sol
    iteration = 0
    while len(_return_plausibly_third_tallest(graph)) > max_n_sol:
        iteration += 1
        
        # Get all possible edges not yet in graph
        possible_edges = []
        for u in vertices:
            for v in vertices:
                if u != v and not graph.has_edge(u, v):
                    possible_edges.append((u, v))
        
        if not possible_edges:
            break
        
        # Weight edges by inverse total degree (prefer lower degree vertices)
        weights = []
        for u, v in possible_edges:
            # Lower total degree = higher weight (2x more likely)
            total_degree_u = graph.get_total_degree(u)
            total_degree_v = graph.get_total_degree(v)
            avg_degree = (total_degree_u + total_degree_v) / 2
            weight = max(1.0, 2.0 / (avg_degree + 1))
            weights.append(weight)
        
        # Sample edge with weights
        edge_idx = random.choices(range(len(possible_edges)), weights=weights)[0]
        u, v = possible_edges[edge_idx]
        
        # Only add if it doesn't create a cycle
        if not would_create_cycle(graph, u, v):
            graph.add_edge(u, v)
        
        # Safety check to prevent infinite loops
        if iteration > 50:
            break
    
    # Store plausibly third tallest before splitting
    third_tallest_before_split = _return_plausibly_third_tallest(graph)
    
    # Split n_same random vertices
    split_vertices = []
    vertices_to_split = random.sample(vertices, n_same)
    
    for original_vertex in vertices_to_split:
        # Get available letters for new vertices
        remaining_letters = [l for l in string.ascii_uppercase if l not in graph.vertices]
        if len(remaining_letters) < 2:
            continue  # Skip if not enough letters
        
        new_vertex1 = remaining_letters[0]
        new_vertex2 = remaining_letters[1]
        
        # Get all edges involving original vertex
        incoming_edges = list(graph.in_edges[original_vertex])
        outgoing_edges = list(graph.edges[original_vertex])
        
        # Remove original vertex
        graph.remove_vertex(original_vertex)
        
        # Add new vertices
        graph.add_vertex(new_vertex1)
        graph.add_vertex(new_vertex2)
        
        # Randomly split incoming edges
        random.shuffle(incoming_edges)
        mid = len(incoming_edges) // 2
        for i, from_v in enumerate(incoming_edges):
            if i < mid:
                graph.add_edge(from_v, new_vertex1)
            else:
                graph.add_edge(from_v, new_vertex2)
        
        # Randomly split outgoing edges  
        random.shuffle(outgoing_edges)
        mid = len(outgoing_edges) // 2
        for i, to_v in enumerate(outgoing_edges):
            if i < mid:
                graph.add_edge(new_vertex1, to_v)
            else:
                graph.add_edge(new_vertex2, to_v)
        
        split_vertices.append((original_vertex, new_vertex1, new_vertex2))
    
    # Update third tallest set according to splitting
    updated_third_tallest = set()
    for vertex in third_tallest_before_split:
        # Check if this vertex was split
        was_split = False
        for orig, new1, new2 in split_vertices:
            if vertex == orig:
                updated_third_tallest.add(new1)
                updated_third_tallest.add(new2)
                was_split = True
                break
        
        if not was_split:
            updated_third_tallest.add(vertex)
    
    return graph, updated_third_tallest, split_vertices


def generate_instance_text(graph: DAG, third_tallest: Set[str], split_vertices: List[Tuple[str, str, str]], generate_viz: bool = True) -> Tuple[str, Set[str]]:
    # Create a random mapping of current vertex names to new random letters
    current_vertices = list(graph.vertices)
    available_letters = list(string.ascii_uppercase)
    random.shuffle(available_letters)
    new_letters = available_letters[:len(current_vertices)]
    
    # Create mapping from old to new vertex names
    vertex_mapping = dict(zip(current_vertices, new_letters))
    
    # Update graph vertex names
    new_graph = DAG()
    for old_vertex in current_vertices:
        new_graph.add_vertex(vertex_mapping[old_vertex])
    
    # Copy edges with new names
    for old_from in graph.vertices:
        for old_to in graph.edges[old_from]:
            new_graph.add_edge(vertex_mapping[old_from], vertex_mapping[old_to])
    
    # Update third_tallest set with new names
    new_third_tallest = {vertex_mapping[v] for v in third_tallest}
    
    # Update split_vertices with new names
    new_split_vertices = []
    for orig, new1, new2 in split_vertices:
        new_orig = vertex_mapping.get(orig, orig)
        new_new1 = vertex_mapping.get(new1, new1) if new1 in vertex_mapping else new1
        new_new2 = vertex_mapping.get(new2, new2) if new2 in vertex_mapping else new2
        new_split_vertices.append((new_orig, new_new1, new_new2))
    
    # Use the new graph and mappings for the rest of the function
    graph = new_graph
    third_tallest = new_third_tallest
    split_vertices = new_split_vertices
    
    sentences = []
    
    # Check for vertex with outgoing edges to all others
    tallest_vertex = None
    for vertex in graph.vertices:
        other_vertices = graph.vertices - {vertex}
        if len(graph.edges[vertex]) == len(other_vertices):
            # Check if it has edges to ALL other vertices
            if graph.edges[vertex] == other_vertices:
                tallest_vertex = vertex
                break
    
    if tallest_vertex:
        sentences.append(f"{tallest_vertex} is the tallest.")
        graph.remove_vertex(tallest_vertex)
    
    # Add sentences for split vertices (same height)
    if split_vertices:
        same_height_groups = {}
        for orig, new1, new2 in split_vertices:
            if orig not in same_height_groups:
                same_height_groups[orig] = []
            same_height_groups[orig].extend([new1, new2])
        
        for group in same_height_groups.values():
            if len(group) >= 2:
                sentence = " and ".join(group[:-1]) + f" and {group[-1]} are the same height."
                sentences.append(sentence)
    
    # Add sentences for edges
    sentence_templates = [
        "{from_vertex} is taller than {to_vertex}", 
        "{to_vertex} is shorter than {from_vertex}"
    ]
    
    for from_vertex in graph.vertices:
        for to_vertex in graph.edges[from_vertex]:
            template = random.choice(sentence_templates)
            sentence = template.format(from_vertex=from_vertex, to_vertex=to_vertex)
            sentences.append(sentence)
    
    # Shuffle sentences
    random.shuffle(sentences)
    
    result_text = "\n".join(sentences)
    
    # Create matplotlib visualization only if requested
    if generate_viz:
        plt.figure(figsize=(12, 10))
        
        # Position vertices in a hierarchical layout
        n_vertices = len(graph.vertices)
        vertices_list = list(graph.vertices)
        
        if n_vertices > 0:
            # Hierarchical layout based on topological ordering
            positions = {}
        
            # Calculate levels using topological sort
            in_degree = {v: graph.get_in_degree(v) for v in vertices_list}
            levels = {}
            queue = [v for v in vertices_list if in_degree[v] == 0]
            level = 0
        
        while queue:
            current_level = queue[:]
            queue = []
            for vertex in current_level:
                levels[vertex] = level
                for neighbor in graph.edges[vertex]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            level += 1
        
        # Handle any remaining vertices (in case of cycles - shouldn't happen in DAG)
        for vertex in vertices_list:
            if vertex not in levels:
                levels[vertex] = level
        
        # Position vertices
        level_counts = Counter(levels.values())
        level_positions = defaultdict(int)
        
        for vertex in vertices_list:
            v_level = levels[vertex]
            level_width = level_counts[v_level]
            
            if level_width == 1:
                x = 0.5
            else:
                x = 0.1 + 0.8 * level_positions[v_level] / (level_width - 1)
            
            y = 0.9 - 0.8 * v_level / max(levels.values()) if levels.values() else 0.5
            positions[vertex] = (x, y)
            level_positions[v_level] += 1
        
        # Draw edges with more pronounced arrows
        for from_vertex in graph.vertices:
            for to_vertex in graph.edges[from_vertex]:
                x1, y1 = positions[from_vertex]
                x2, y2 = positions[to_vertex]
                
                # Calculate arrow position to avoid overlapping with vertex circles
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx, dy = dx/length, dy/length
                    # Adjust start and end points
                    radius = 0.03
                    x1_adj = x1 + radius * dx
                    y1_adj = y1 + radius * dy
                    x2_adj = x2 - radius * dx
                    y2_adj = y2 - radius * dy
                    
                    plt.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                               arrowprops=dict(arrowstyle='->', color='darkblue', 
                                             lw=3.5, shrinkA=0, shrinkB=0,
                                             mutation_scale=30))
        
        # Draw green lines for vertices that were split from the same original vertex
        for orig, new1, new2 in split_vertices:
            if new1 in positions and new2 in positions:
                x1, y1 = positions[new1]
                x2, y2 = positions[new2]
                plt.plot([x1, x2], [y1, y2], color='green', linewidth=3, zorder=3)
        
        # Draw vertices with better styling
        for vertex in graph.vertices:
            x, y = positions[vertex]
            plt.scatter(x, y, s=1200, c='lightcyan', edgecolors='navy', linewidth=3, zorder=5)
            plt.text(x, y, vertex, ha='center', va='center', fontsize=16, fontweight='bold', zorder=6)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Height Comparison Graph')
        plt.tight_layout()
        
        # Convert plot to string representation for now
        plt.savefig('temp_graph.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        result_text = "\n".join(sentences)
        result_text += "\n\n[Graph visualization saved as temp_graph.png]"
    else:
        result_text = "\n".join(sentences)
    
    return result_text, new_third_tallest


def generate_instance(n_vertex: int, max_n_sol: int, generate_viz: bool = True) -> Tuple[str, Set[str]]:
    graph, third_tallest, split_vertices = generate_instance_graph(n_vertex, max_n_sol)
    return generate_instance_text(graph, third_tallest, split_vertices, generate_viz)


def run_statistics(n_vertex: int, max_n_sol: int, n_runs: int = 100):
    """Run statistics on plausible 3rd tallest counts across multiple instances."""
    counts = []
    
    for _ in range(n_runs):
        graph, third_tallest, split_vertices = generate_instance_graph(n_vertex, max_n_sol)
        counts.append(len(third_tallest))
    
    # Calculate statistics
    mean_val = statistics.mean(counts)
    q1 = np.percentile(counts, 25)
    q3 = np.percentile(counts, 75)
    mode_val = statistics.mode(counts) if counts else 0
    max_val = max(counts) if counts else 0
    min_val = min(counts) if counts else 0
    
    print(f"Statistics for plausible 3rd tallest counts over {n_runs} instances:")
    print(f"Mean: {mean_val:.2f}")
    print(f"Q1 (25th percentile): {q1:.2f}")
    print(f"Q3 (75th percentile): {q3:.2f}")
    print(f"Mode: {mode_val}")
    print(f"Max: {max_val}")
    print(f"Min: {min_val}")
    print(f"Full distribution: {sorted(Counter(counts).items())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ambiguous height puzzle instances')
    parser.add_argument('--statistics', action='store_true', 
                       help='Generate 100 instances and show statistics of plausible 3rd tallest counts')
    parser.add_argument('--n-vertex', type=int, default=3,
                       help='Number of vertices (default: 3)')
    parser.add_argument('--max-3rd-tallest', type=int, default=1,
                       help='Maximum number of plausible 3rd tallest solutions (default: 1)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: random)')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 999999)
    
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")
    
    if args.statistics:
        run_statistics(args.n_vertex, args.max_3rd_tallest)
    else:
        # Test with single instance
        graph, third_tallest, split_vertices = generate_instance_graph(args.n_vertex, args.max_3rd_tallest)
        result_text, shuffled_third_tallest = generate_instance_text(graph, third_tallest, split_vertices)
        
        print("Generated instance:")
        print("Plausible 3rd tallest:", sorted(list(shuffled_third_tallest)))
        print(result_text)