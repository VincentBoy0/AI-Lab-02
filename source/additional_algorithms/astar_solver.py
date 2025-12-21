"""
A* Solver for Hashiwokakero Puzzle on CNF Solution Space

This implementation follows the theoretical framework for designing an A* solver
that operates directly on the Conjunctive Normal Form (CNF) solution space.

Key Components:
- Boolean Variables: x_{e,1} (at least 1 bridge), x_{e,2} (exactly 2 bridges)
- CNF Clauses: Encode capacity, crossing, and implication constraints
- g(n): Cost function = number of bridge segments assigned True
- h(n): Heuristic function = (sum of remaining island capacities) / 2
- Unit Propagation: Integrated into A* expansion for efficiency
- Lazy Connectivity: DSU-based verification at goal states

The heuristic is proven to be both admissible and consistent.
"""

import heapq
from copy import deepcopy
from collections import defaultdict
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.DisjoinUnionSet import DSU


# Constants for variable assignment states
UNASSIGNED = None
TRUE = True
FALSE = False


class CNFFormula:
    """
    Represents the CNF formula for Hashiwokakero puzzle.
    
    Variables:
    - For each potential edge e between islands (u, v):
      - x_{e,1}: True if there is exactly 1 bridge on edge e
      - x_{e,2}: True if there are exactly 2 bridges on edge e
    
    Encoding:
    - x_{e,1}=False, x_{e,2}=False => 0 bridges
    - x_{e,1}=True,  x_{e,2}=False => 1 bridge
    - x_{e,1}=False, x_{e,2}=True  => 2 bridges
    - x_{e,1}=True,  x_{e,2}=True  => INVALID (mutual exclusion)
    
    Constraint: -x_{e,1} OR -x_{e,2} (at most one can be true)
    """
    
    def __init__(self):
        self.var_count = 0
        self.clauses = []  # List of clauses, each clause is a list of literals
        
        # Variable mappings
        self.edge_to_var1 = {}  # edge -> variable ID for "exactly 1 bridge"
        self.edge_to_var2 = {}  # edge -> variable ID for "exactly 2 bridges"
        self.var_to_edge = {}   # variable ID -> (edge, bridge_level)
        
        # Island data for heuristic calculation
        self.islands = []
        self.island_capacity = {}
        self.island_edges = defaultdict(list)  # island -> list of edges
        
    def create_variable(self):
        """Create a new Boolean variable and return its ID."""
        self.var_count += 1
        return self.var_count
    
    def add_clause(self, literals):
        """
        Add a clause to the formula.
        A clause is a disjunction (OR) of literals.
        A literal is a variable ID (positive) or its negation (negative).
        """
        if literals:  # Don't add empty clauses
            self.clauses.append(list(literals))
    
    def get_literal(self, var_id, positive=True):
        """Get a literal for a variable (positive or negative)."""
        return var_id if positive else -var_id


class SearchNode:
    """
    Represents a node in the A* search graph (partial truth assignment).
    
    The assignment maps variable IDs to {True, False, None (unassigned)}.
    """
    
    def __init__(self, assignment, g, h, unassigned_vars):
        self.assignment = assignment  # {var_id: True/False/None}
        self.g = g  # Cost so far (bridges built)
        self.h = h  # Heuristic (remaining capacity / 2)
        self.f = g + h
        self.unassigned_vars = unassigned_vars  # List of unassigned variable IDs
    
    def __lt__(self, other):
        """For heap comparison - prioritize by f, then by h (prefer deeper search)."""
        if self.f != other.f:
            return self.f < other.f
        return self.h < other.h
    
    def get_state_hash(self):
        """Generate a hashable state representation for closed set."""
        # Only include assigned variables in hash
        assigned = tuple(sorted((k, v) for k, v in self.assignment.items() if v is not None))
        return assigned


class AStarHashiSolver:
    """
    A* Solver for Hashiwokakero that operates on CNF solution space.
    
    The search space is the Boolean hypercube of partial assignments.
    Nodes are partial truth assignments, edges are decision literals.
    """
    
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Build the CNF formula
        self.cnf = CNFFormula()
        
        # Island and edge data
        self.islands = []
        self.island_capacity = {}
        self.potential_edges = []
        self.crossing_edges = defaultdict(list)
        
        # Parse grid and build CNF
        self._parse_grid()
        self._find_potential_edges()
        self._find_crossing_edges()
        self._create_variables()
        self._create_clauses()
        
        # Statistics
        self.nodes_expanded = 0
        self.max_open_size = 0
    
    def _parse_grid(self):
        """Extract islands from the grid."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] > 0:
                    island = (r, c)
                    self.islands.append(island)
                    self.island_capacity[island] = self.grid[r][c]
        
        self.cnf.islands = self.islands
        self.cnf.island_capacity = self.island_capacity
    
    def _find_potential_edges(self):
        """Find all valid potential bridge connections between islands."""
        for island in self.islands:
            r, c = island
            
            # Look right (horizontal)
            for nc in range(c + 1, self.cols):
                if self.grid[r][nc] > 0:
                    neighbor = (r, nc)
                    edge = (island, neighbor) if island < neighbor else (neighbor, island)
                    if edge not in self.potential_edges:
                        self.potential_edges.append(edge)
                    break
                elif self.grid[r][nc] < 0:  # Hit a blocked cell
                    break
            
            # Look down (vertical)
            for nr in range(r + 1, self.rows):
                if self.grid[nr][c] > 0:
                    neighbor = (nr, c)
                    edge = (island, neighbor) if island < neighbor else (neighbor, island)
                    if edge not in self.potential_edges:
                        self.potential_edges.append(edge)
                    break
                elif self.grid[nr][c] < 0:  # Hit a blocked cell
                    break
    
    def _check_crossing(self, edge1, edge2):
        """Check if two edges would cross if both had bridges."""
        u1, v1 = edge1
        u2, v2 = edge2
        
        is_horz1 = (u1[0] == v1[0])
        is_horz2 = (u2[0] == v2[0])
        
        # Same orientation edges cannot cross
        if is_horz1 == is_horz2:
            return False
        
        if is_horz1:  # edge1 horizontal, edge2 vertical
            r1 = u1[0]
            c1_start, c1_end = min(u1[1], v1[1]), max(u1[1], v1[1])
            c2 = u2[1]
            r2_start, r2_end = min(u2[0], v2[0]), max(u2[0], v2[0])
            return (c1_start < c2 < c1_end) and (r2_start < r1 < r2_end)
        else:  # edge1 vertical, edge2 horizontal
            c1 = u1[1]
            r1_start, r1_end = min(u1[0], v1[0]), max(u1[0], v1[0])
            r2 = u2[0]
            c2_start, c2_end = min(u2[1], v2[1]), max(u2[1], v2[1])
            return (c2_start < c1 < c2_end) and (r1_start < r2 < r1_end)
    
    def _find_crossing_edges(self):
        """Pre-compute which edges would cross each other."""
        n = len(self.potential_edges)
        for i in range(n):
            for j in range(i + 1, n):
                if self._check_crossing(self.potential_edges[i], self.potential_edges[j]):
                    self.crossing_edges[self.potential_edges[i]].append(self.potential_edges[j])
                    self.crossing_edges[self.potential_edges[j]].append(self.potential_edges[i])
    
    def _create_variables(self):
        """
        Create Boolean variables for the CNF encoding.
        
        For each edge e:
        - x_{e,1}: True if exactly 1 bridge exists on edge e
        - x_{e,2}: True if exactly 2 bridges exist on edge e
        
        Both False = 0 bridges, at most one can be True.
        """
        for edge in self.potential_edges:
            # Variable for "exactly 1 bridge"
            var1 = self.cnf.create_variable()
            self.cnf.edge_to_var1[edge] = var1
            self.cnf.var_to_edge[var1] = (edge, 1)
            
            # Variable for "exactly 2 bridges"
            var2 = self.cnf.create_variable()
            self.cnf.edge_to_var2[edge] = var2
            self.cnf.var_to_edge[var2] = (edge, 2)
            
            # Track edges per island for heuristic
            u, v = edge
            self.cnf.island_edges[u].append(edge)
            self.cnf.island_edges[v].append(edge)
    
    def _create_clauses(self):
        """
        Create CNF clauses encoding Hashiwokakero constraints.
        
        Constraints:
        1. Mutual Exclusion: NOT(x_{e,1} AND x_{e,2}) - can't be both 1 and 2 bridges
        2. Crossing: For crossing edges e1, e2: bridges on both is forbidden
        3. Capacity: Each island's degree must equal its capacity
        """
        # 1. Mutual exclusion: at most one of var1, var2 can be true
        # In CNF: NOT x_{e,1} OR NOT x_{e,2}
        for edge in self.potential_edges:
            var1 = self.cnf.edge_to_var1[edge]
            var2 = self.cnf.edge_to_var2[edge]
            self.cnf.add_clause([-var1, -var2])
        
        # 2. Crossing constraints: NOT(x_{e1,1} AND x_{e2,1})
        # In CNF: NOT x_{e1,1} OR NOT x_{e2,1}
        added_crossings = set()
        for edge1 in self.potential_edges:
            for edge2 in self.crossing_edges.get(edge1, []):
                pair = tuple(sorted([edge1, edge2]))
                if pair not in added_crossings:
                    var1_e1 = self.cnf.edge_to_var1[edge1]
                    var1_e2 = self.cnf.edge_to_var1[edge2]
                    var2_e1 = self.cnf.edge_to_var2[edge1]
                    var2_e2 = self.cnf.edge_to_var2[edge2]
                    var_e1s = [var1_e1, var2_e1]
                    var_e2s = [var1_e2, var2_e2]
                    for v1 in var_e1s:
                        for v2 in var_e2s:
                            self.cnf.add_clause([-v1, -v2])
                    
                    added_crossings.add(pair)
        self._create_capacity_clauses()
    
    def _create_capacity_clauses(self):
        """
        Create capacity constraint clauses.
        
        For an island with capacity D and edges e1, e2, ..., ek:
        - Sum of bridges = D
        - Each edge contributes: 0 (no bridge), 1 (single), or 2 (double)
        
        We encode this using auxiliary variables and at-most/at-least constraints.
        For simplicity, we use a direct encoding with bound propagation.
        """
        for island in self.islands:
            capacity = self.island_capacity[island]
            edges = self.cnf.island_edges[island]
            
            if not edges:
                continue
            
            # Upper bound: Cannot exceed capacity
            # We'll check this during search via constraint propagation
            
            # Lower bound: Must have enough potential bridges
            max_possible = 2 * len(edges)
            if max_possible < capacity:
                # Unsatisfiable - add empty clause
                self.cnf.add_clause([])
    
    def _calculate_current_degree(self, assignment, island):
        """
        Calculate the current number of bridges connected to an island
        based on the partial assignment.
        
        With "exactly" semantics:
        - var1=True => 1 bridge
        - var2=True => 2 bridges
        - Both cannot be True (mutual exclusion)
        """
        degree = 0
        for edge in self.cnf.island_edges[island]:
            var1 = self.cnf.edge_to_var1[edge]
            var2 = self.cnf.edge_to_var2[edge]
            
            # var1 and var2 are mutually exclusive
            if assignment.get(var1) == TRUE:
                degree += 1
            elif assignment.get(var2) == TRUE:
                degree += 2
        
        return degree
    
    def _calculate_heuristic(self, assignment):
        """
        Calculate h(n) = (1/2) * sum of remaining capacities.
        
        This is proven to be admissible and consistent.
        """
        total_residual = 0
        for island in self.islands:
            current_deg = self._calculate_current_degree(assignment, island)
            residual = max(0, self.island_capacity[island] - current_deg)
            total_residual += residual
        
        return total_residual / 2
    
    def _calculate_g(self, assignment):
        """
        Calculate g(n) = total bridge segments assigned True.
        
        With "exactly" semantics:
        - var1=True => 1 bridge (contributes 1)
        - var2=True => 2 bridges (contributes 2)
        """
        g = 0
        for edge in self.potential_edges:
            var1 = self.cnf.edge_to_var1[edge]
            var2 = self.cnf.edge_to_var2[edge]
            
            # var1 and var2 are mutually exclusive
            if assignment.get(var1) == TRUE:
                g += 1  # Exactly 1 bridge
            elif assignment.get(var2) == TRUE:
                g += 2  # Exactly 2 bridges
        
        return g
    
    def _evaluate_clause(self, clause, assignment):
        """
        Evaluate a clause under partial assignment.
        
        Returns:
        - TRUE if clause is satisfied
        - FALSE if clause is falsified (conflict)
        - UNASSIGNED if clause is undetermined
        - (unassigned_literal,) if unit clause (one unassigned literal)
        """
        unassigned_literals = []
        
        for literal in clause:
            var_id = abs(literal)
            is_positive = literal > 0
            
            value = assignment.get(var_id)
            
            if value is None:
                unassigned_literals.append(literal)
            elif (value == TRUE and is_positive) or (value == FALSE and not is_positive):
                return TRUE  # Clause satisfied
        
        if not unassigned_literals:
            return FALSE  # All literals false - conflict
        elif len(unassigned_literals) == 1:
            return (unassigned_literals[0],)  # Unit clause
        else:
            return UNASSIGNED  # Still undetermined
    
    def _unit_propagate(self, assignment, unassigned_vars):
        """
        Perform unit propagation on the CNF formula.
        
        Returns:
        - (new_assignment, new_unassigned, success)
        - success is False if a conflict is detected
        """
        assignment = dict(assignment)
        unassigned_vars = set(unassigned_vars)
        
        changed = True
        while changed:
            changed = False
            
            for clause in self.cnf.clauses:
                result = self._evaluate_clause(clause, assignment)
                
                if result == FALSE:
                    # Conflict detected
                    return assignment, list(unassigned_vars), False
                
                if isinstance(result, tuple) and len(result) == 1:
                    # Unit clause - force the literal
                    literal = result[0]
                    var_id = abs(literal)
                    value = TRUE if literal > 0 else FALSE
                    
                    if assignment.get(var_id) is None:
                        assignment[var_id] = value
                        unassigned_vars.discard(var_id)
                        changed = True
            
            # Additional domain-specific propagation
            # Check capacity constraints
            for island in self.islands:
                capacity = self.island_capacity[island]
                current_deg = self._calculate_current_degree(assignment, island)
                
                if current_deg > capacity:
                    # Over capacity - conflict
                    return assignment, list(unassigned_vars), False
                
                remaining = capacity - current_deg
                
                # Find available edges (unassigned and not blocked)
                available_edges = []
                max_additional = 0
                for edge in self.cnf.island_edges[island]:
                    var1 = self.cnf.edge_to_var1[edge]
                    var2 = self.cnf.edge_to_var2[edge]
                            
                    # Skip if edge is completely blocked (both var1 and var2 are False)
                    if assignment.get(var1) == FALSE and assignment.get(var2) == FALSE:
                        continue
                    
                    # Check if edge is blocked by crossing
                    blocked = False
                    for crossing_edge in self.crossing_edges.get(edge, []):
                        crossing_var1 = self.cnf.edge_to_var1[crossing_edge]
                        crossing_var2 = self.cnf.edge_to_var2[crossing_edge]
                        # Edge is blocked if crossing edge has any bridge
                        if assignment.get(crossing_var1) == TRUE or assignment.get(crossing_var2) == TRUE:
                            blocked = True
                            break
                    
                    if blocked:
                        # Force this edge to have no bridge
                        if assignment.get(var1) is None:
                            assignment[var1] = FALSE
                            unassigned_vars.discard(var1)
                            changed = True
                        if assignment.get(var2) is None:
                            assignment[var2] = FALSE
                            unassigned_vars.discard(var2)
                            changed = True
                        continue
                    if assignment.get(var2) == TRUE:
                        max_contrib = 2  # Fixed at 2 bridges
                    elif assignment.get(var1) == TRUE:
                        max_contrib = 1  # Fixed at 1 bridge (cannot upgrade!)
                    elif assignment.get(var1) == FALSE and assignment.get(var2) == FALSE:
                        max_contrib = 0  # Fixed at 0 bridges
                        continue  # Skip this edge
                    elif assignment.get(var2) == FALSE:
                        # var2 is False, var1 is unassigned: can be 0 or 1
                        max_contrib = 1
                    elif assignment.get(var1) == FALSE:
                        # var1 is False, var2 is unassigned: can be 0 or 2
                        max_contrib = 2
                    else:
                        # Both unassigned: can be 0, 1, or 2
                        max_contrib = 2
                    if assignment.get(var2) != TRUE and assignment.get(var1) != TRUE:
                        available_edges.append((edge, max_contrib))
                        max_additional += max_contrib
                
                if max_additional < remaining:
                    return assignment, list(unassigned_vars), False

                still_needed = remaining
                if remaining == 0:
                    for edge in self.cnf.island_edges[island]:
                        var1 = self.cnf.edge_to_var1[edge]
                        var2 = self.cnf.edge_to_var2[edge]
                        
                        # If edge not yet assigned as having bridge, set to no bridge
                        if assignment.get(var1) is None:
                            assignment[var1] = FALSE
                            unassigned_vars.discard(var1)
                            changed = True
                        if assignment.get(var2) is None:
                            assignment[var2] = FALSE
                            unassigned_vars.discard(var2)
                            changed = True
                        # If edge has single bridge, cannot upgrade to double
                        elif assignment.get(var1) == TRUE and assignment.get(var2) is None:
                            assignment[var2] = FALSE
                            unassigned_vars.discard(var2)
                            changed = True
                if max_additional == remaining and remaining > 0 and len(available_edges) > 0:
                    for edge, max_contrib in available_edges:
                        var1 = self.cnf.edge_to_var1[edge]
                        var2 = self.cnf.edge_to_var2[edge]
                        
                        if max_contrib == 2:
                            # Must have 2 bridges (set var2=True, var1=False)
                            if assignment.get(var2) is None:
                                assignment[var2] = TRUE
                                unassigned_vars.discard(var2)
                                changed = True
                            if assignment.get(var1) is None:
                                assignment[var1] = FALSE
                                unassigned_vars.discard(var1)
                                changed = True
                        elif max_contrib == 1:
                            # Must have 1 bridge (set var1=True, var2=False)
                            if assignment.get(var1) is None:
                                assignment[var1] = TRUE
                                unassigned_vars.discard(var1)
                                changed = True
                            if assignment.get(var2) is None:
                                assignment[var2] = FALSE
                                unassigned_vars.discard(var2)
                                changed = True
                
                # Single edge case: force specific number of bridges
                # elif still_needed == 1 and len(available_edges) > 1:
                #     for edge2, max_contrib2 in available_edges:
                #         if assignment.get(var2) is None:
                #             var2 = self.cnf.edge_to_var2[edge2]
                #             assignment[var2] = FALSE
                #             unassigned_vars.discard(var2)
                #             changed = True
                elif len(available_edges) == 1 and still_needed > 0:
                    edge, max_contrib = available_edges[0]
                    var1 = self.cnf.edge_to_var1[edge]
                    var2 = self.cnf.edge_to_var2[edge]
                    
                    if still_needed == 1:
                        # Need exactly 1 more bridge from this edge
                        if assignment.get(var1) is None:
                            assignment[var1] = TRUE
                            unassigned_vars.discard(var1)
                            changed = True
                        if assignment.get(var2) is None:
                            assignment[var2] = FALSE
                            unassigned_vars.discard(var2)
                            changed = True
                    elif still_needed == 2:
                        # Need exactly 2 more bridges from this edge
                        if max_contrib >= 2:
                            if assignment.get(var2) is None:
                                assignment[var2] = TRUE
                                unassigned_vars.discard(var2)
                                changed = True
                            if assignment.get(var1) is None:
                                assignment[var1] = FALSE
                                unassigned_vars.discard(var1)
                                changed = True
                        else:
                            # Can't satisfy - conflict
                            return assignment, list(unassigned_vars), False
                    elif still_needed > max_contrib:
                    # Can't satisfy with current available edges - conflict
                        return assignment, list(unassigned_vars), False
                
        
        return assignment, list(unassigned_vars), True
    
    def _is_edge_blocked(self, edge, assignment):
        """Check if an edge is blocked by a crossing edge with bridges."""
        for crossing_edge in self.crossing_edges.get(edge, []):
            var1 = self.cnf.edge_to_var1[crossing_edge]
            var2 = self.cnf.edge_to_var2[crossing_edge]
            if assignment.get(var1) == TRUE or assignment.get(var2) == TRUE:
                return True
        return False
    
    def _check_connectivity(self, assignment):
        """
        Check if all islands with bridges form a single connected component.
        Uses DSU for efficient connectivity checking.
        """
        if len(self.islands) <= 1:
            return True
        
        node_to_idx = {node: i for i, node in enumerate(self.islands)}
        dsu = DSU(len(self.islands))
        
        for edge in self.potential_edges:
            var1 = self.cnf.edge_to_var1[edge]
            var2 = self.cnf.edge_to_var2[edge]
            # Edge has bridge if var1=True (1 bridge) OR var2=True (2 bridges)
            if assignment.get(var1) == TRUE or assignment.get(var2) == TRUE:
                u, v = edge
                dsu.join(node_to_idx[u] + 1, node_to_idx[v] + 1)
        
        return dsu.check_connected()
    
    def _check_early_connectivity(self, assignment, unassigned_vars):
        """
        Early check if connectivity is still possible.
        If islands are disconnected and no unassigned edges can connect them, prune.
        """
        if len(self.islands) <= 1:
            return True
        
        node_to_idx = {node: i for i, node in enumerate(self.islands)}
        dsu = DSU(len(self.islands))
        
        # Add current bridges
        for edge in self.potential_edges:
            var1 = self.cnf.edge_to_var1[edge]
            var2 = self.cnf.edge_to_var2[edge]
            if assignment.get(var1) == TRUE or assignment.get(var2) == TRUE:
                u, v = edge
                dsu.join(node_to_idx[u] + 1, node_to_idx[v] + 1)
        
        # Add potential bridges from unassigned edges (not blocked)
        for edge in self.potential_edges:
            var1 = self.cnf.edge_to_var1[edge]
            var2 = self.cnf.edge_to_var2[edge]
            # If edge is unassigned and not blocked
            if assignment.get(var1) is None and assignment.get(var2) is None:
                if not self._is_edge_blocked(edge, assignment):
                    u, v = edge
                    dsu.join(node_to_idx[u] + 1, node_to_idx[v] + 1)
        
        return dsu.check_connected()
    
    def _is_goal_state(self, assignment):
        """Check if all islands have their capacity exactly satisfied."""
        for island in self.islands:
            current_deg = self._calculate_current_degree(assignment, island)
            if current_deg != self.island_capacity[island]:
                return False
        return True
    
    def _select_next_variable(self, assignment, unassigned_vars):
        """
        Select the next variable to branch on using MRV/saturation heuristic.
        
        Prioritizes variables connected to islands with lowest remaining capacity
        (highest saturation), analogous to DSatur heuristic in graph coloring.
        """
        best_var = None
        best_score = float('inf')
        
        for var_id in unassigned_vars:
            if var_id not in self.cnf.var_to_edge:
                continue
            
            edge, level = self.cnf.var_to_edge[var_id]
            var1 = self.cnf.edge_to_var1[edge]
            
            # # Skip var2 if var1 is not yet True (maintain implication)
            # if level == 2 and assignment.get(var1) != TRUE:
            #     continue
            
            # Skip if edge is blocked by crossing
            blocked = False
            for crossing_edge in self.crossing_edges.get(edge, []):
                crossing_var1 = self.cnf.edge_to_var1[crossing_edge]
                crossing_var2 = self.cnf.edge_to_var2[crossing_edge]
                # Edge is blocked if crossing edge has any bridge (var1=True OR var2=True)
                if assignment.get(crossing_var1) == TRUE or assignment.get(crossing_var2) == TRUE:
                    blocked = True
                    break
            
            if blocked:
                continue
            
            u, v = edge
            u_remaining = self.island_capacity[u] - self._calculate_current_degree(assignment, u)
            v_remaining = self.island_capacity[v] - self._calculate_current_degree(assignment, v)
            
            # Lower remaining = higher saturation = more constrained
            score = min(u_remaining, v_remaining)
            
            # Prefer var1 over var2 for the same edge
            if level == 2:
                score += 0.5
            
            if score < best_score:
                best_score = score
                best_var = var_id
        
        return best_var
    
    def solve(self):
        """
        Main A* search algorithm on CNF solution space.
        
        Returns:
        - List of ((u, v), count) tuples representing the solution bridges,
          or None if no solution exists.
        """
        self.start_time = time.time()
        
        # Initial state: all variables unassigned
        initial_assignment = {var_id: None for var_id in range(1, self.cnf.var_count + 1)}
        initial_unassigned = list(range(1, self.cnf.var_count + 1))
        
        # Apply initial unit propagation
        initial_assignment, initial_unassigned, success = self._unit_propagate(
            initial_assignment, initial_unassigned
        )
        
        if not success:
            return None
        
        # Check early connectivity
        if not self._check_early_connectivity(initial_assignment, initial_unassigned):
            return None
        
        initial_g = self._calculate_g(initial_assignment)
        initial_h = self._calculate_heuristic(initial_assignment)
        
        start_node = SearchNode(
            assignment=initial_assignment,
            g=initial_g,
            h=initial_h,
            unassigned_vars=initial_unassigned
        )
        
        # A* data structures
        open_list = [start_node]
        closed_set = set()
        
        while open_list:
            self.max_open_size = max(self.max_open_size, len(open_list))
            
            # Pop node with lowest f value
            current = heapq.heappop(open_list)
            
            # Skip if already visited
            state_hash = current.get_state_hash()
            if state_hash in closed_set:
                continue
            closed_set.add(state_hash)
            
            self.nodes_expanded += 1
            
            # Progress indicator for large puzzles
            if self.nodes_expanded % 10000 == 0:
                elapsed = time.time() - self.start_time
                print(f"Nodes: {self.nodes_expanded}, open: {len(open_list)}, h: {current.h:.1f}, time: {elapsed:.1f}s")
            
            # Check if goal state (h = 0 means all capacities satisfied)
            if current.h == 0 or self._is_goal_state(current.assignment):
                # Verify connectivity (lazy constraint)
                if self._check_connectivity(current.assignment):
                    return self._extract_solution(current.assignment)
                else:
                    # Connectivity failed, continue search
                    continue
            
            # No more variables to assign
            if not current.unassigned_vars:
                continue
            
            # Select next variable using MRV heuristic
            next_var = self._select_next_variable(current.assignment, current.unassigned_vars)
            
            if next_var is None:
                # No valid variable to branch on
                continue
            
            # Generate children: assign True or False to the variable
            for value in [TRUE, FALSE]:
                new_assignment = dict(current.assignment)
                new_assignment[next_var] = value
                new_unassigned = [v for v in current.unassigned_vars if v != next_var]
                
                # Apply unit propagation
                new_assignment, new_unassigned, success = self._unit_propagate(
                    new_assignment, new_unassigned
                )
                
                if not success:
                    # Conflict detected, prune this branch
                    continue
                
                # Early connectivity check
                if not self._check_early_connectivity(new_assignment, new_unassigned):
                    continue
                
                # Calculate g and h for new state
                new_g = self._calculate_g(new_assignment)
                new_h = self._calculate_heuristic(new_assignment)
                
                # Early pruning: check if goal is reachable
                if new_h == float('inf'):
                    continue
                
                child_node = SearchNode(
                    assignment=new_assignment,
                    g=new_g,
                    h=new_h,
                    unassigned_vars=new_unassigned
                )
                
                # Skip if already in closed set
                child_hash = child_node.get_state_hash()
                if child_hash in closed_set:
                    continue
                
                heapq.heappush(open_list, child_node)
        
        return None
    
    def _extract_solution(self, assignment):
        """Extract bridge configuration from the satisfying assignment."""
        solution = []
        
        for edge in self.potential_edges:
            var1 = self.cnf.edge_to_var1[edge]
            var2 = self.cnf.edge_to_var2[edge]
            
            if assignment.get(var2) == TRUE:
                solution.append((edge, 2))
            elif assignment.get(var1) == TRUE:
                solution.append((edge, 1))
        
        return solution
    
    def get_statistics(self):
        """Return search statistics."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'max_open_size': self.max_open_size,
            'num_islands': len(self.islands),
            'num_potential_edges': len(self.potential_edges),
            'num_variables': self.cnf.var_count,
            'num_clauses': len(self.cnf.clauses)
        }


def solve_hashiwokakero_astar_cnf(grid):
    """
    Convenience function to solve a Hashiwokakero puzzle using A* on CNF.
    
    Args:
        grid: 2D list where positive integers are island capacities, 0 is empty
        
    Returns:
        (solution, statistics) where solution is a list of ((u, v), count) tuples,
        or (None, statistics) if unsolvable
    """
    solver = AStarHashiSolver(grid)
    solution = solver.solve()
    return solution, solver.get_statistics()


def format_solution_grid(grid, solution):
    """
    Format the solution as a visual grid.
    
    Args:
        grid: Original puzzle grid
        solution: List of ((u, v), count) tuples
        
    Returns:
        2D list representing the solved puzzle
    """
    if solution is None:
        return None
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Create output grid with string cells
    output = [[' ' for _ in range(cols)] for _ in range(rows)]
    
    # Place islands
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] > 0:
                output[r][c] = str(grid[r][c])
    
    # Place bridges
    for (u, v), count in solution:
        r1, c1 = u
        r2, c2 = v
        
        if r1 == r2:  # Horizontal bridge
            bridge_char = '-' if count == 1 else '='
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                output[r1][c] = bridge_char
        else:  # Vertical bridge
            bridge_char = '|' if count == 1 else '"'
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                output[r][c1] = bridge_char
    
    return output


def print_solution(grid, solution):
    """Print the solution grid to console."""
    output = format_solution_grid(grid, solution)
    if output is None:
        print("No solution found")
        return
    
    for row in output:
        print(''.join(row))


# Example usage and testing
if __name__ == "__main__":
    # Simple test puzzle
    test_grid = [
        [2, 0, 0, 2],
        [0, 0, 0, 0],
        [2, 0, 0, 2]
    ]
    
    print("Test Puzzle:")
    for row in test_grid:
        print(row)
    print()
    
    solution, stats = solve_hashiwokakero_astar_cnf(test_grid)
    
    print("Solution:")
    print_solution(test_grid, solution)
    print()
    
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
