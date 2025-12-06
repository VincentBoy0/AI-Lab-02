# from pysat.solvers import Solver

# class HashiwokakeroSolver:
#     def __init__(self):
#         self.variable_map = {}
#         self.reverse_map = {}
#         self.counter = 0
#         self.solver = Solver(name="glucose3")
#         self.MAX_EDGE = 2
    
#     def get_variable_id(self, u, v, cnt):
#         if u > v:
#             u, v = v, u
#         key = (u, v, cnt)
#         if key not in self.variable_map:
#             self.counter += 1
#             self.variable_map[key] = self.counter
#             self.reverse_map[self.counter] = key
#         return self.variable_map[key]

#     def add_constraint_two_nodes_adjacent(self, u, v):
#         # with 2 adjacent node u, v, only one true case:
#         # bridge(u, v, 0) v bridge(u, v, 1) v bridge(u, v, 2)
#         # [~bridge(u, v, 0) v ~bridge(u, v, 1)] ^ [~bridge(u, v, 1) v ~bridge(u, v, 2)] ^ [~bridge(u, v, 0) v ~bridge(u, v, 2)]
#         uv_0 = self.get_variable_id(u, v, 0)
#         uv_1 = self.get_variable_id(u, v, 1)
#         uv_2 = self.get_variable_id(u, v, 2)
#         self.solver.add_clause([uv_0, uv_1, uv_2])

#         self.solver.add_clause([-uv_0, -uv_1])
#         self.solver.add_clause([-uv_0, -uv_2])
#         self.solver.add_clause([-uv_1, -uv_2])
    
#     def add_constraint_two_edge_cross(self, u1, v1, u2, v2):
#         for i in range(1, self.MAX_EDGE + 1):
#             edge_ab = self.get_variable_id(u1, v1, i)
#             for j in range(1, self.MAX_EDGE + 1):
#                 edge_cd = self.get_variable_id(u2, v2, j)
#                 self.solver.add_clause([-edge_ab, -edge_cd])
    
#     def find_all_valid_combinations(self, target_sum, num_neighbors):
#         combos = []
#         def backtrack(cur_sum, neighbor_id, res):
#             if cur_sum > target_sum:
#                 return
#             if neighbor_id == num_neighbors:
#                 if cur_sum == target_sum:
#                     combos.append(res.copy())
#                 return
#             for i in range(self.MAX_EDGE + 1):
#                 res.append(i)
#                 backtrack(cur_sum + i, neighbor_id + 1, res)
#                 res.pop()
#         backtrack(0, 0, [])
#         return combos

#     def add_sum_constraints(self, u, value_u, neigbors):
#         combos = self.find_all_valid_combinations(value_u, len(neigbors))
#         z_vars = []
#         for combo in combos:
#             self.counter += 1
#             z = self.counter
#             z_vars.append(z)
#             for i in range(len(neigbors)):
#                 wanted_val = combo[i]
#                 uv_wanted = self.get_variable_id(u, neigbors[i], wanted_val)
#                 self.solver.add_clause([-z, wanted_val])
#                 for j in range(self.MAX_EDGE + 1):
#                     if j == wanted_val:
#                         continue
#                     uv_other = self.get_variable_id(u, neigbors[i], j)
#                     self.solver.add_clause([-z, -uv_other])
        
#         if z_vars:
#             self.solver.add_clause(z_vars)
#         else:
#             self.solver.add_clause([])
        
from pysat.solvers import Solver
import itertools

class HashiSolver:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.nodes = [] # List các tọa độ (r, c) có đảo
        self.neighbors_map = {} # Map: (r,c) -> list of neighbor_coords
        self.potential_edges = [] # List các cặp cạnh khả thi (u, v)
        
        # PySAT variables
        self.solver = Solver(name='glucose3')
        self.var_map = {} # (u, v, count) -> int_id
        self.rev_map = {} # int_id -> (u, v, count)
        self.counter = 0

        # Phân tích bản đồ ngay khi khởi tạo
        self._parse_grid()
        self._find_potential_edges()

    def _get_var(self, u, v, cnt):
        """Lấy ID biến cho cặp u, v với số lượng cầu cnt (1 hoặc 2)"""
        if u > v: u, v = v, u # Chuẩn hóa thứ tự
        key = (u, v, cnt)
        if key not in self.var_map:
            self.counter += 1
            self.var_map[key] = self.counter
            self.rev_map[self.counter] = key
        return self.var_map[key]

    def _parse_grid(self):
        """Tìm vị trí các đảo"""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] > 0:
                    self.nodes.append((r, c))

    def _find_potential_edges(self):
        """Tìm các hàng xóm có thể nối cầu (không bị chắn)"""
        for i, u in enumerate(self.nodes):
            r, c = u
            if u not in self.neighbors_map:
                self.neighbors_map[u] = []
            
            # Tìm 4 hướng: Phải (East) và Dưới (South) 
            # (Chỉ cần tìm 2 hướng dương rồi add 2 chiều để tránh lặp)
            
            # 1. Tìm hướng Đông (East)
            for nc in range(c + 1, self.cols):
                if self.grid[r][nc] > 0: # Gặp đảo
                    v = (r, nc)
                    self.neighbors_map[u].append(v)
                    if v not in self.neighbors_map: self.neighbors_map[v] = []
                    self.neighbors_map[v].append(u)
                    
                    # Lưu cạnh khả thi (luôn lưu u < v)
                    if u < v: self.potential_edges.append((u, v))
                    else: self.potential_edges.append((v, u))
                    break
                # (Nếu gặp logic chắn đường khác thì break ở đây, nhưng Hashi đơn giản chỉ quan tâm đảo)
            
            # 2. Tìm hướng Nam (South)
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
        """Kiểm tra 2 cạnh có cắt nhau không"""
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
        # 1. Ràng buộc cơ bản trên mỗi cạnh (At most one logic)
        for u, v in self.potential_edges:
            b1 = self._get_var(u, v, 1)
            b2 = self._get_var(u, v, 2)
            # Không thể vừa 1 vừa 2 cầu
            self.solver.add_clause([-b1, -b2])

        # 2. Ràng buộc cắt nhau (Crossing)
        # Duyệt qua mọi cặp cạnh khả thi
        n = len(self.potential_edges)
        for i in range(n):
            for j in range(i + 1, n):
                edge1 = self.potential_edges[i]
                edge2 = self.potential_edges[j]
                
                if self._check_crossing(edge1, edge2):
                    # Nếu cắt nhau, cấm cả 2 cùng tồn tại
                    # Cần xét cả trường hợp 1 cầu và 2 cầu
                    # (HasBridge_1 OR HasBridge_2)
                    
                    # Logic: Edge1(any) -> NOT Edge2(any)
                    # Triển khai thành 4 mệnh đề
                    u1, v1 = edge1
                    u2, v2 = edge2
                    
                    vars1 = [self._get_var(u1, v1, 1), self._get_var(u1, v1, 2)]
                    vars2 = [self._get_var(u2, v2, 1), self._get_var(u2, v2, 2)]
                    
                    for v_ab in vars1:
                        for v_cd in vars2:
                            self.solver.add_clause([-v_ab, -v_cd])

        # 3. Ràng buộc TỔNG tại mỗi đảo (Sum Constraints)
        for node in self.nodes:
            target = self.grid[node[0]][node[1]]
            neighbors = self.neighbors_map.get(node, [])
            
            # Sinh tổ hợp hợp lệ
            valid_combos = []
            # Mỗi neighbor có thể nhận 0, 1, 2 cầu. Dùng itertools product
            for p in itertools.product([0, 1, 2], repeat=len(neighbors)):
                if sum(p) == target:
                    valid_combos.append(p)
            
            if not valid_combos:
                print(f"Lỗi: Đảo tại {node} giá trị {target} không có tổ hợp thỏa mãn!")
                print(len(neighbors))
                self.solver.add_clause([]) # UNSAT ngay lập tức
                continue

            # Tạo biến phụ Z cho mỗi combo
            z_vars = []
            for combo in valid_combos:
                self.counter += 1
                z = self.counter
                z_vars.append(z)
                
                # Z -> (Neighbor_i có số cầu đúng như combo)
                for i, val in enumerate(combo):
                    neighbor = neighbors[i]
                    b1 = self._get_var(node, neighbor, 1)
                    b2 = self._get_var(node, neighbor, 2)
                    
                    if val == 1:
                        self.solver.add_clause([-z, b1])  # Phải là 1
                        self.solver.add_clause([-z, -b2]) # Cấm là 2
                    elif val == 2:
                        self.solver.add_clause([-z, -b1]) # Cấm là 1
                        self.solver.add_clause([-z, b2])  # Phải là 2
                    else: # val == 0
                        self.solver.add_clause([-z, -b1]) # Cấm 1
                        self.solver.add_clause([-z, -b2]) # Cấm 2
            
            # Ít nhất 1 combo phải đúng
            self.solver.add_clause(z_vars)

    def solve(self):
        self.build_constraints()
        if self.solver.solve():
            model = self.solver.get_model()
            result_edges = []
            for val in model:
                if val > 0 and val in self.rev_map:
                    u, v, cnt = self.rev_map[val]
                    result_edges.append(((u, v), cnt))
            return result_edges
        else:
            return None