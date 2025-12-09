from pysat.solvers import Solver
import itertools
from utils.DisjoinUnionSet import DSU
from collections import defaultdict

# Idea
# Variable Bridge(u, v, cnt): there are cnt edges between node u and v (0 <= cnt <= 2)
# assume u < v and u, v in the same row or the same column
# Constraints:
#       -(1) For every adjacent node u, v: only one case is true: Bridge(u, v, 0), Bridge(u, v, 1), Bridge(u, v, 2)
#           ==> {At least one is true: Bridge(u, v, 0) v Bridge(u, v, 1) v Bridge(u, v, 2)
#               {At most one is true: (-Bridge(u, v, 0) v -Bridge(u, v, 1)) ^ (-Bridge(u, v, 1) v -Bridge(u, v, 2)) ^ (-Bridge(u, v, 0) v -Bridge(u, v, 2))

#       -(2) Two edges do not cross each other (assume 2 edges are A-B and C-D):
#           ==> -(Bridge(A, B, 1) v Bridge(A, B, 2)) v -(Bridge(C, D, 1) v Bridge(C, D, 2)) (it means if we have bridge between A-B (1 or 2 also valid), we do not have edge between C, D, and the reverse condition is also right)
#           ==> { -Bridge(A, B, 1) v -Bridge(C, D, 1)  
#               { -Bridge(A, B, 1) v -Bridge(C, D, 2)  
#               { -Bridge(A, B, 2) v -Bridge(C, D, 1)  
#               { -Bridge(A, B, 2) v -Bridge(C, D, 2)  

#       -(3) Sum all edges of a node equal with the value of that node
#       I call Z(u) = {z1(u), z2(u), z3(u), ...} is the set of combinations of edge number's neighbors which equal with Value(u)
#       For example: if Value(u) = 5, and there are 3 neigbors of u (v1, v2, v3)
#       So, we have the combinations like this: (2, 2, 1), (2, 1, 2), (1, 2, 2) 
#       Therefore, I call z_i(u) = {x1, x2, ..., xk} (k is the number of neighbors of u) is also a variable, z_i(u) is true if:
#           - z_i(u) = Bridge(u, v1, x1) ^ Bridge(u, v2, x2) ^ ... ^ Bridge(u, vk, xk)
#           ==> For neighbor v1: 
#               + if x1 == 1: (only one case is true )
#                   ==> { -z_1(u) v Bridge(u, v1, 1)
#                       { -z_1(u) v -Bridge(u, v1, 2)
#               + if x1 == 2: 
#                   ==> { -z_1(u) v -Bridge(u, v1, 1)  
#                       { -z_1(u) v Bridge(u, v1, 2)  
#               + if x1 == 0: 
#                   ==> { -z_1(u) v -Bridge(u, v1, 1)  
#                       { -z_1(u) v -Bridge(u, v1, 2)  
#               Do it for all neighbor's u
#           Finally: z_1(u) v z_2(u) v ... v z_k(u) (at least one combination is valid)

#       - (4): All egdes must connect into one component:
#           After finding a solution, use DSU to connect and check if all nodes are connected together or not
#           If not, add constraints to avoid this combonation and solve again
class HashiSolver: 
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.nodes = [] 
        self.neighbors_map = {} 
        self.potential_edges = [] 
        
        # PySAT variables
        self.solver = Solver(name='glucose3')
        self.var_map = {} # (u, v, count) -> int_id
        self.rev_map = {} # int_id -> (u, v, count)
        self.counter = 0

        self._parse_grid()
        self._find_potential_edges()

    def _get_id_var(self, u, v, cnt):
        if u > v: u, v = v, u 
        key = (u, v, cnt)
        if key not in self.var_map:
            self.counter += 1
            self.var_map[key] = self.counter
            self.rev_map[self.counter] = key
        return self.var_map[key]

    def _parse_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] > 0:
                    self.nodes.append((r, c))

    def _find_potential_edges(self):
        for i, u in enumerate(self.nodes):
            r, c = u
            if u not in self.neighbors_map:
                self.neighbors_map[u] = []
            
            for nc in range(c + 1, self.cols):
                if self.grid[r][nc] > 0: # Gặp đảo
                    v = (r, nc)
                    self.neighbors_map[u].append(v)
                    if v not in self.neighbors_map: self.neighbors_map[v] = []
                    self.neighbors_map[v].append(u)
                    
                    if u < v: self.potential_edges.append((u, v))
                    else: self.potential_edges.append((v, u))
                    break
            
            for nr in range(r + 1, self.rows):
                if self.grid[nr][c] > 0:
                    v = (nr, c)
                    self.neighbors_map[u].append(v)
                    if v not in self.neighbors_map: self.neighbors_map[v] = []
                    self.neighbors_map[v].append(u)
                    
                    if u < v: self.potential_edges.append((u, v))
                    else: self.potential_edges.append((v, u))
                    break

    def _check_crossing(self, edge1, edge2):
        u1, v1 = edge1
        u2, v2 = edge2
        
        if u1 > v1: u1, v1 = v1, u1
        if u2 > v2: u2, v2 = v2, u2

        is_horz1 = (u1[0] == v1[0])
        is_horz2 = (u2[0] == v2[0])

        if is_horz1 == is_horz2: return False

        if is_horz1: # 1 Hor, 2 Ver
            r1, c1_start, c1_end = u1[0], u1[1], v1[1]
            c2, r2_start, r2_end = u2[1], u2[0], v2[0]
            return (c1_start < c2 < c1_end) and (r2_start < r1 < r2_end)
        else: # 1 Ver, 2 Hor
            c1, r1_start, r1_end = u1[1], u1[0], v1[0]
            r2, c2_start, c2_end = u2[0], u2[1], v2[1]
            return (c2_start < c1 < c2_end) and (r1_start < r2 < r1_end)

    def build_constraints(self):
        for u, v in self.potential_edges:
            b1 = self._get_id_var(u, v, 1)
            b2 = self._get_id_var(u, v, 2)
            self.solver.add_clause([-b1, -b2])

        n = len(self.potential_edges)
        for i in range(n):
            for j in range(i + 1, n):
                edge1 = self.potential_edges[i]
                edge2 = self.potential_edges[j]
                
                if self._check_crossing(edge1, edge2):

                    u1, v1 = edge1
                    u2, v2 = edge2
                    
                    vars1 = [self._get_id_var(u1, v1, 1), self._get_id_var(u1, v1, 2)]
                    vars2 = [self._get_id_var(u2, v2, 1), self._get_id_var(u2, v2, 2)]
                    
                    for v_ab in vars1:
                        for v_cd in vars2:
                            self.solver.add_clause([-v_ab, -v_cd])

        for node in self.nodes:
            target = self.grid[node[0]][node[1]]
            neighbors = self.neighbors_map.get(node, [])
            
            valid_combos = []
            for p in itertools.product([0, 1, 2], repeat=len(neighbors)):
                if sum(p) == target:
                    valid_combos.append(p)
            
            if not valid_combos:
                self.solver.add_clause([])
                continue

            z_vars = []
            for combo in valid_combos:
                self.counter += 1
                z = self.counter
                z_vars.append(z)
                
                for i, val in enumerate(combo):
                    neighbor = neighbors[i]
                    b1 = self._get_id_var(node, neighbor, 1)
                    b2 = self._get_id_var(node, neighbor, 2)
                    
                    if val == 1:
                        self.solver.add_clause([-z, b1])  # Phải là 1
                        self.solver.add_clause([-z, -b2]) # Cấm là 2
                    elif val == 2:
                        self.solver.add_clause([-z, -b1]) # Cấm là 1
                        self.solver.add_clause([-z, b2])  # Phải là 2
                    else:
                        self.solver.add_clause([-z, -b1]) # Cấm 1
                        self.solver.add_clause([-z, -b2]) # Cấm 2
            
            self.solver.add_clause(z_vars)

    def solve(self):
        # print(len(self.nodes), len(self.potential_edges))
        self.build_constraints()
        for _ in range(50000):
            if not self.solver.solve():
                return None
            model = self.solver.get_model()

            # Take valid edges
            result_edges = []
            for val in model:
                if val > 0 and val in self.rev_map:
                    u, v, cnt = self.rev_map[val]
                    result_edges.append(((u, v), cnt))

            # Check connected components
            nodes = []
            nodes_id = defaultdict()
            for ((u, v), cnt) in result_edges:
                if u not in nodes:
                    nodes.append(u)
                    nodes_id[u] = len(nodes)
                if v not in nodes:
                    nodes.append(v)
                    nodes_id[v] = len(nodes)

            # Use DSU to join all nodes that have edge to connect
            dsu = DSU(len(nodes))
            for ((u, v), cnt) in result_edges:
                dsu.join(nodes_id[u], nodes_id[v])
            
            if dsu.check_connected():
                return result_edges
            
            # remove the case if there is more than 1 connected components
            remove_all_edges = []
            for ((u, v), cnt) in result_edges:
                node_id = self._get_id_var(u, v, cnt)
                remove_all_edges.append(-node_id)

            self.solver.add_clause(remove_all_edges)
    
        return None