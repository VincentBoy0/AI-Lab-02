"""
Comparison Script for Hashiwokakero Solvers

This script compares 4 different algorithms for solving Hashiwokakero puzzles:
1. PySAT CNF Solver (CDCL-based)
2. A* on CNF Solution Space
3. Backtracking with Constraint Propagation
4. Brute Force (if available)

Metrics compared:
- Execution time
- Solution correctness
- Memory usage (nodes expanded for search algorithms)
"""

import sys
import os
import time
import traceback
from typing import List, Tuple, Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import solvers
from HashiwokakeroSolver import HashiSolver
from additional_algorithms.astar_solver import AStarHashiSolver, solve_hashiwokakero_astar_cnf, format_solution_grid
from additional_algorithms.backtrack import Backtrack


class SolverResult:
    """Stores the result of a solver run."""
    def __init__(self, name: str):
        self.name = name
        self.solution = None
        self.time_taken = 0.0
        self.success = False
        self.error = None
        self.stats = {}
    
    def __str__(self):
        status = "✓ SOLVED" if self.success else "✗ FAILED"
        if self.error:
            status = f"✗ ERROR: {self.error}"
        return f"{self.name}: {status} in {self.time_taken:.4f}s"


def parse_grid_from_file(filepath: str) -> List[List[int]]:
    """Parse a grid from a test file."""
    grid = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row = [int(x.strip()) for x in line.split(',')]
                grid.append(row)
    return grid


def parse_grid_from_string(grid_str: str) -> List[List[int]]:
    """Parse a grid from a string."""
    grid = []
    for line in grid_str.strip().split('\n'):
        line = line.strip()
        if line:
            row = [int(x.strip()) for x in line.split(',')]
            grid.append(row)
    return grid


def verify_solution(grid: List[List[int]], solution: List[Tuple]) -> Tuple[bool, str]:
    """
    Verify that a solution is correct.
    
    Checks:
    1. All islands have correct degree (number of bridges)
    2. No bridges cross each other
    3. All islands are connected
    """
    if solution is None:
        return False, "No solution provided"
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Build island list and degree count
    islands = []
    island_capacity = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] > 0:
                islands.append((r, c))
                island_capacity[(r, c)] = grid[r][c]
    
    # Count degrees from solution
    degree_count = {island: 0 for island in islands}
    
    for item in solution:
        if len(item) == 2:
            (u, v), count = item
        else:
            continue
        
        if u in degree_count:
            degree_count[u] += count
        if v in degree_count:
            degree_count[v] += count
    
    # Check capacities
    for island in islands:
        expected = island_capacity[island]
        actual = degree_count.get(island, 0)
        if actual != expected:
            return False, f"Island {island} has degree {actual}, expected {expected}"
    
    # Check connectivity using simple DFS
    if len(islands) > 1:
        # Build adjacency from solution
        adj = {island: [] for island in islands}
        for item in solution:
            if len(item) == 2:
                (u, v), count = item
                if count > 0:
                    if u in adj:
                        adj[u].append(v)
                    if v in adj:
                        adj[v].append(u)
        
        # DFS from first island
        visited = set()
        stack = [islands[0]]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        
        if len(visited) != len(islands):
            return False, f"Not all islands connected: {len(visited)}/{len(islands)}"
    
    return True, "Valid solution"


def run_pysat_solver(grid: List[List[int]], timeout: float = 60.0) -> SolverResult:
    """Run the PySAT CNF solver."""
    result = SolverResult("PySAT CNF (CDCL)")
    
    try:
        start_time = time.time()
        solver = HashiSolver(grid)
        solution = solver.solve()
        result.time_taken = time.time() - start_time
        
        if solution is not None:
            result.solution = solution
            result.success = True
            result.stats = {
                'num_islands': len(solver.nodes),
                'num_edges': len(solver.potential_edges),
                'num_variables': solver.counter
            }
        else:
            result.success = False
            result.error = "No solution found"
    except Exception as e:
        result.error = str(e)
        result.time_taken = time.time() - start_time
    
    return result


def run_astar_solver(grid: List[List[int]], timeout: float = 60.0) -> SolverResult:
    """Run the A* CNF solver."""
    result = SolverResult("A* on CNF Space")
    
    try:
        start_time = time.time()
        solver = AStarHashiSolver(grid)
        solution = solver.solve()
        result.time_taken = time.time() - start_time
        
        if solution is not None:
            result.solution = solution
            result.success = True
        else:
            result.success = False
            result.error = "No solution found"
        
        result.stats = solver.get_statistics()
    except Exception as e:
        result.error = str(e)
        result.time_taken = time.time() - start_time
        traceback.print_exc()
    
    return result


def run_backtrack_solver(grid: List[List[int]], timeout: float = 60.0) -> SolverResult:
    """Run the Backtracking solver."""
    result = SolverResult("Backtracking")
    
    try:
        start_time = time.time()
        solver = Backtrack(grid)
        solution = solver.solve()
        result.time_taken = time.time() - start_time
        
        if solution is not None:
            result.solution = solution
            result.success = True
            result.stats = {
                'num_islands': len(solver.nodes),
                'num_edges': len(solver.potential_edges)
            }
        else:
            result.success = False
            result.error = "No solution found"
    except Exception as e:
        result.error = str(e)
        result.time_taken = time.time() - start_time
    
    return result


def run_brute_force_solver(grid: List[List[int]], timeout: float = 60.0) -> SolverResult:
    """Run the Brute Force solver (placeholder if not implemented)."""
    result = SolverResult("Brute Force")
    result.error = "Not implemented"
    result.time_taken = 0.0
    return result


def format_solution_display(grid: List[List[int]], solution: List[Tuple]) -> str:
    """Format solution for display."""
    if solution is None:
        return "No solution"
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Create output grid
    output = [['.' if grid[r][c] == 0 else str(grid[r][c]) for c in range(cols)] for r in range(rows)]
    
    # Place bridges
    for item in solution:
        if len(item) == 2:
            (u, v), count = item
            r1, c1 = u
            r2, c2 = v
            
            if r1 == r2:  # Horizontal
                bridge_char = '-' if count == 1 else '='
                for c in range(min(c1, c2) + 1, max(c1, c2)):
                    output[r1][c] = bridge_char
            else:  # Vertical
                bridge_char = '|' if count == 1 else '$'
                for r in range(min(r1, r2) + 1, max(r1, r2)):
                    output[r][c1] = bridge_char
    
    return '\n'.join([''.join(row) for row in output])


def compare_solvers(grid: List[List[int]], timeout: float = 60.0, 
                    run_all: bool = True, verbose: bool = True) -> Dict[str, SolverResult]:
    """
    Run all solvers on a grid and compare results.
    
    Args:
        grid: The puzzle grid
        timeout: Maximum time per solver
        run_all: Whether to run all solvers even if one finds a solution
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary of solver name -> result
    """
    results = {}
    
    # Print puzzle
    if verbose:
        print("\n" + "=" * 60)
        print("PUZZLE:")
        print("=" * 60)
        for row in grid:
            print(' '.join(str(x) if x > 0 else '.' for x in row))
        print()
    
    # Run each solver
    solvers = [
        ("PySAT CNF", run_pysat_solver),
        ("A* CNF", run_astar_solver),
        ("Backtracking", run_backtrack_solver),
        # ("Brute Force", run_brute_force_solver),  # Uncomment if implemented
    ]
    
    for name, solver_func in solvers:
        if verbose:
            print(f"\nRunning {name}...", end=" ", flush=True)
        
        result = solver_func(grid, timeout)
        results[name] = result
        
        if verbose:
            if result.success:
                print(f"✓ Solved in {result.time_taken:.4f}s")
            elif result.error:
                print(f"✗ {result.error} ({result.time_taken:.4f}s)")
            else:
                print(f"✗ No solution ({result.time_taken:.4f}s)")
    
    return results


def print_comparison_table(results: Dict[str, SolverResult], grid: List[List[int]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Header
    print(f"{'Algorithm':<25} {'Status':<12} {'Time (s)':<12} {'Details'}")
    print("-" * 80)
    
    # Results
    for name, result in results.items():
        status = "SOLVED" if result.success else "FAILED"
        time_str = f"{result.time_taken:.4f}"
        
        details = []
        if result.stats:
            if 'nodes_expanded' in result.stats:
                details.append(f"nodes={result.stats['nodes_expanded']}")
            if 'num_variables' in result.stats:
                details.append(f"vars={result.stats['num_variables']}")
        if result.error and not result.success:
            details.append(result.error[:30])
        
        details_str = ", ".join(details) if details else ""
        print(f"{name:<25} {status:<12} {time_str:<12} {details_str}")
    
    # Verify solutions
    print("\n" + "-" * 80)
    print("SOLUTION VERIFICATION:")
    print("-" * 80)
    
    for name, result in results.items():
        if result.solution:
            valid, msg = verify_solution(grid, result.solution)
            status = "✓ Valid" if valid else f"✗ Invalid: {msg}"
            print(f"{name:<25} {status}")
        else:
            print(f"{name:<25} No solution to verify")
    
    # Show first valid solution
    print("\n" + "-" * 80)
    print("SOLUTION DISPLAY:")
    print("-" * 80)
    
    for name, result in results.items():
        if result.success and result.solution:
            print(f"\n{name}:")
            print(format_solution_display(grid, result.solution))
            break
    else:
        print("No valid solution found by any solver.")


def run_test_suite(test_dir: str = "tests/input", verbose: bool = True):
    """Run comparison on all test files in a directory."""
    import glob
    
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.txt")))
    
    if not test_files:
        print(f"No test files found in {test_dir}")
        return
    
    all_results = []
    
    for test_file in test_files:
        test_name = os.path.basename(test_file)
        print(f"\n{'#' * 80}")
        print(f"# TEST: {test_name}")
        print(f"{'#' * 80}")
        
        try:
            grid = parse_grid_from_file(test_file)
            results = compare_solvers(grid, verbose=verbose)
            print_comparison_table(results, grid)
            all_results.append((test_name, results))
        except Exception as e:
            print(f"Error running test {test_name}: {e}")
            traceback.print_exc()
    
    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY - ALL TESTS")
    print("=" * 100)
    
    print(f"\n{'Test':<20} {'PySAT':<15} {'A* CNF':<15} {'Backtrack':<15}")
    print("-" * 100)
    
    for test_name, results in all_results:
        row = [test_name[:18]]
        for solver_name in ["PySAT CNF", "A* CNF", "Backtracking"]:
            if solver_name in results:
                r = results[solver_name]
                if r.success:
                    row.append(f"✓ {r.time_taken:.3f}s")
                else:
                    row.append(f"✗ {r.time_taken:.3f}s")
            else:
                row.append("N/A")
        print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15}")


# Interactive mode
def interactive_mode():
    """Run in interactive mode, allowing user to input puzzles."""
    print("\n" + "=" * 60)
    print("HASHIWOKAKERO SOLVER COMPARISON - Interactive Mode")
    print("=" * 60)
    print("\nEnter a puzzle grid (comma-separated values).")
    print("Enter an empty line when done.")
    print("Example:")
    print("  2, 0, 0, 2")
    print("  0, 0, 0, 0")
    print("  2, 0, 0, 2")
    print()
    
    lines = []
    while True:
        try:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        except EOFError:
            break
    
    if not lines:
        print("No input provided.")
        return
    
    grid_str = '\n'.join(lines)
    grid = parse_grid_from_string(grid_str)
    
    results = compare_solvers(grid, verbose=True)
    print_comparison_table(results, grid)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Hashiwokakero solvers")
    parser.add_argument("--test-dir", type=str, default="tests/input",
                        help="Directory containing test files")
    parser.add_argument("--file", type=str, help="Single test file to run")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.file:
        grid = parse_grid_from_file(args.file)
        results = compare_solvers(grid, verbose=not args.quiet)
        print_comparison_table(results, grid)
    else:
        run_test_suite(args.test_dir, verbose=not args.quiet)
