# Hashiwokakero Puzzle Solver

A Python implementation of various algorithms to solve Hashiwokakero (Bridges) puzzles, including SAT-based, A*, Backtracking, and Brute Force approaches.

## 📖 About Hashiwokakero

Hashiwokakero (also known as Bridges or Hashi) is a logic puzzle where:
- Islands (nodes) are represented by numbers indicating how many bridges must connect to them
- Bridges connect islands horizontally or vertically
- Up to 2 bridges can connect any two adjacent islands
- Bridges cannot cross each other
- All islands must be connected into a single group

## 🛠️ Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## 📦 Installation

1. Clone or download the project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Main Solver

The main script supports different solver types. Run from the `source` directory:

```bash
# Using SAT solver (default - recommended)
python main.py sat

# Using A* solver
python main.py astar
```

### Available Solver Types

| Solver | Command | Description |
|--------|---------|-------------|
| SAT (CNF) | `sat` | Uses PySAT library with CDCL-based SAT solving |
| A* Search | `astar` | A* algorithm on the solution space |

### Comparing Solvers

To compare performance between different solving algorithms:

```bash
python compare_solvers.py
```

This will run all available solvers and display:
- Execution time
- Solution correctness
- Algorithm statistics

## 📁 Project Structure

```
source/
├── main.py                    # Main entry point
├── HashiwokakeroSolver.py     # SAT-based CNF solver
├── compare_solvers.py         # Solver comparison utility
├── requirements.txt           # Python dependencies
├── additional_algorithms/     # Alternative solving algorithms
│   ├── A_Star.py             # A* implementation (edge-based)
│   ├── astar_solver.py       # A* on CNF solution space
│   ├── backtrack.py          # Backtracking with constraint propagation
│   └── brute_force.py        # Brute force approach
├── utils/
│   ├── DisjoinUnionSet.py    # Union-Find data structure
│   └── visualize_solution.py # Solution visualization
└── tests/
    ├── input/                 # Test puzzle inputs
    │   ├── input-01.txt
    │   ├── input-02.txt
    │   └── ...
    └── output/                # Generated solutions
        ├── output-01.txt
        ├── output-02.txt
        └── ...
```

## 📝 Input Format

Input files are located in `tests/input/` and use the following format:
- Grid of numbers separated by commas
- `0` represents empty cells
- Numbers `1-8` represent islands with that many required bridges

**Example (input-01.txt):**
```
0, 2, 0, 5, 0, 0, 2
0, 0, 0, 0, 0, 0, 0
4, 0, 2, 0, 2, 0, 4
0, 0, 0, 0, 0, 0, 0
0, 1, 0, 5, 0, 2, 0
0, 0, 0, 0, 0, 0, 0
4, 0, 0, 0, 0, 0, 3
```

## 📤 Output Format

Solutions are written to `tests/output/` with the following symbols:
- Numbers: Original islands
- `-`: Single horizontal bridge
- `=`: Double horizontal bridge
- `|`: Single vertical bridge
- `$`: Double vertical bridge

## 🧪 Running Specific Test Cases

To run individual test cases, you can modify the `main.py` or use the solver classes directly:

```python
from HashiwokakeroSolver import HashiSolver

# Define a puzzle grid
grid = [
    [0, 2, 0, 5, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0],
    [4, 0, 2, 0, 2, 0, 4],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 5, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 3]
]

# Solve using SAT solver
solver = HashiSolver(grid)
solution = solver.solve()
print(solution)
```

## 📚 Algorithm Details

### SAT Solver (CNF)
- Converts the puzzle into a Boolean satisfiability problem
- Uses PySAT library for efficient CDCL-based solving
- Ensures connectivity using Disjoint Set Union (DSU)

### A* Search
- Explores solution space using heuristic-guided search
- Evaluates states based on constraint satisfaction

### Backtracking
- Systematic search with constraint propagation
- Prunes invalid branches early

### Brute Force
- Exhaustive search through all possible configurations
- Used mainly for verification on small puzzles

## 👥 Authors

HCMUS - Introduction to Artificial Intelligence Lab 02

## 📄 License

This project is for educational purposes.
