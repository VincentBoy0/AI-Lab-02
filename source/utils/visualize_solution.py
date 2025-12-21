"""
Visualize Hashiwokakero solution from various input formats.

Symbol meanings:
- Numbers (1-8): Island with that capacity
- "0" or ".": Empty cell
- "-": Single horizontal bridge
- "=": Double horizontal bridge  
- "|": Single vertical bridge
- "$": Double vertical bridge
"""

def visualize_solution(solution_grid):
    """
    Visualize the Hashiwokakero solution with colored output.
    
    Args:
        solution_grid: 2D list of strings representing the solution
    """
    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    
    # Unicode characters for better visualization
    SYMBOLS = {
        '-': '─',      # Single horizontal bridge
        '=': '═',      # Double horizontal bridge
        '|': '│',      # Single vertical bridge
        '$': '║',      # Double vertical bridge
        '0': ' ',      # Empty cell
        '.': ' ',      # Empty cell
    }
    
    print("\n" + "=" * 50)
    print("     HASHIWOKAKERO SOLUTION")
    print("=" * 50 + "\n")
    
    for row in solution_grid:
        line = ""
        for cell in row:
            cell_str = str(cell).strip('"').strip("'")
            
            if cell_str.isdigit() and cell_str != '0':
                # Island - display in bold yellow
                line += f"{BOLD}{YELLOW}({cell_str}){RESET}"
            elif cell_str in ['-', '─']:
                # Single horizontal bridge - cyan
                line += f"{CYAN}───{RESET}"
            elif cell_str in ['=', '═']:
                # Double horizontal bridge - blue
                line += f"{BLUE}═══{RESET}"
            elif cell_str in ['|', '│']:
                # Single vertical bridge - cyan
                line += f"{CYAN} │ {RESET}"
            elif cell_str in ['$', '║']:
                # Double vertical bridge - blue
                line += f"{BLUE} ║ {RESET}"
            else:
                # Empty cell
                line += "   "
        
        print(line)
    
    print("\n" + "=" * 50)
    print("Legend:")
    print(f"  {BOLD}{YELLOW}(N){RESET} = Island with capacity N")
    print(f"  {CYAN}───{RESET} = Single horizontal bridge")
    print(f"  {BLUE}═══{RESET} = Double horizontal bridge")
    print(f"  {CYAN} │ {RESET} = Single vertical bridge")
    print(f"  {BLUE} ║ {RESET} = Double vertical bridge")
    print("=" * 50 + "\n")


def visualize_simple(solution_grid):
    """
    Simple ASCII visualization without colors.
    
    Args:
        solution_grid: 2D list of strings representing the solution
    """
    # Unicode characters for better visualization
    SYMBOLS = {
        '-': '───',
        '=': '═══',
        '|': ' │ ',
        '$': ' ║ ',
        '0': '   ',
        '.': '   ',
    }
    
    print("\n" + "+" + "-" * 50 + "+")
    print("|" + " HASHIWOKAKERO SOLUTION".center(50) + "|")
    print("+" + "-" * 50 + "+")
    print()
    
    for row in solution_grid:
        line = "  "
        for cell in row:
            cell_str = str(cell).strip('"').strip("'")
            
            if cell_str.isdigit() and cell_str != '0':
                # Island
                line += f"({cell_str})"
            elif cell_str in SYMBOLS:
                line += SYMBOLS[cell_str]
            else:
                line += "   "
        
        print(line)
    
    print()
    print("+" + "-" * 50 + "+")
    print("| Legend:                                          |")
    print("|   (N) = Island with capacity N                   |")
    print("|   ─── = Single horizontal bridge                 |")
    print("|   ═══ = Double horizontal bridge                 |")
    print("|    │  = Single vertical bridge                   |")
    print("|    ║  = Double vertical bridge                   |")
    print("+" + "-" * 50 + "+")
    print()


def parse_solution_string(solution_str):
    """
    Parse solution from string format.
    
    Args:
        solution_str: String representation of the solution grid
        
    Returns:
        2D list of cell values
    """
    import ast
    
    lines = solution_str.strip().split('\n')
    grid = []
    
    for line in lines:
        line = line.strip()
        if line:
            try:
                # Try to parse as Python list
                row = ast.literal_eval(line)
                grid.append(row)
            except:
                # Parse manually
                row = []
                for char in line:
                    if char not in ' [],"\':':
                        row.append(char)
                if row:
                    grid.append(row)
    
    return grid


def visualize_from_string(solution_str, use_colors=True):
    """
    Visualize solution from string input.
    
    Args:
        solution_str: String representation of the solution
        use_colors: Whether to use ANSI colors
    """
    grid = parse_solution_string(solution_str)
    
    if use_colors:
        visualize_solution(grid)
    else:
        visualize_simple(grid)


# Example usage and testing
if __name__ == "__main__":
    # Test with the provided solution
    test_solution = '''
    ["0", "2", "0", "2", "0"]
    ["2", "8", "=", "8", "2"]
    ["0", "$", "0", "$", "0"]
    ["2", "8", "=", "8", "2"]
    ["0", "2", "0", "2", "0"]
    '''
    
    print("Testing with provided solution:")
    print("-" * 40)
    
    # Parse and visualize
    grid = parse_solution_string(test_solution)
    
    print("\nWith colors:")
    visualize_solution(grid)
    
    print("\nWithout colors (simple ASCII):")
    visualize_simple(grid)
    
    # Also show how to use directly with a grid
    print("\n" + "=" * 50)
    print("Direct grid input example:")
    print("=" * 50)