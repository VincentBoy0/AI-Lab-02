from utils.DisjoinUnionSet import DSU

class Backtrack:
    def __init__(self, grid):
        self.grid = grid
        self.nodes = []
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.value = [0]

        # self.adj = {} 
        self.var_map = {}
        self.rev_map = {}
        self.ban_edges = {}
        self.counter = 0
        self.potential_edges = []

        self._parse_grid()
        self._find_potential_edges()

        # for edge in self.potential_edges:
        #     print(edge, self.rev_map[edge[0]], self.rev_map[edge[1]])
        

    def _get_id_var(self, x, y):
        key = (x, y)
        if key not in self.var_map:
            self.counter += 1
            self.var_map[key] = self.counter
            self.rev_map[self.counter] = key
            self.value.append(0)
        return self.var_map[key]
    
    def _parse_grid(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] > 0:
                    self.nodes.append((i, j))
                    self.value[self._get_id_var(i, j)] = self.grid[i][j]

    # def _add_edge(self, u, v):
    #     if u not in self.adj: self.adj[u] = []
    #     self.adj[u].append(v)
    #     if v not in self.adj: self.adj[v] = []
    #     self.adj[v].append(u)
        
    def _find_potential_edges(self):
        for i in range(self.rows):
            for j in range(self.cols):

                if self.grid[i][j] == 0:
                    continue

                u = self._get_id_var(i, j)

                for ni in range(i + 1, self.rows):
                    if self.grid[ni][j] > 0:
                        v = self._get_id_var(ni, j)
                        self.potential_edges.append((u, v))
                        break
                
                for nj in range(j + 1, self.cols):
                    if self.grid[i][nj] > 0:
                        v = self._get_id_var(i, nj)
                        self.potential_edges.append((u, v))
                        break

        for i in range(len(self.potential_edges)):
            edge1 = self.potential_edges[i]
            for j in range(i + 1, len(self.potential_edges)):
                edge2 = self.potential_edges[j]
                if self._check_crossing(edge1, edge2):
                    if edge1 not in self.ban_edges: self.ban_edges[edge1] = []
                    if edge2 not in self.ban_edges: self.ban_edges[edge2] = []
                    self.ban_edges[edge1].append(edge2)
                    self.ban_edges[edge2].append(edge1)
            

    def _check_crossing(self, edge1, edge2):
        # edge1 = (self.rev_map[edge1[0]], self.rev_map[edge1[1]])
        # edge2 = (self.rev_map[edge2[0]], self.rev_map[edge2[1]])
        u1, v1 = self.rev_map[edge1[0]], self.rev_map[edge1[1]]
        u2, v2 = self.rev_map[edge2[0]], self.rev_map[edge2[1]]

        
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

    def _backtrack(self, edge_id, ban_lists = [], result_edges = []):
        if edge_id == len(self.potential_edges):
            sum_remain_val = sum(self.value)
            if sum_remain_val != 0:
                return None
            
            dsu = DSU(len(self.nodes))
            for ((u, v), cnt) in result_edges:
                if cnt > 0:
                    dsu.join(u, v)
            if dsu.check_connected():
                return result_edges
            return None
            
        edge = self.potential_edges[edge_id]
        u_id = edge[0]
        v_id = edge[1]

        
        res = self._backtrack(edge_id + 1, ban_lists, result_edges)
        if res != None:
            return res
        
        if edge in ban_lists:
            return None

        for c in range(1, 3):
            if self.value[u_id] >= c and self.value[v_id] >= c:
                self.value[u_id] -= c
                self.value[v_id] -= c

                if edge in self.ban_edges:
                    for ban_edge in self.ban_edges[edge]:
                        ban_lists.append(ban_edge)

                result_edges.append(((u_id, v_id), c))
                res = self._backtrack(edge_id + 1, ban_lists, result_edges)

                if res != None:
                    return res

                result_edges.remove(((u_id, v_id), c))
                self.value[u_id] += c
                self.value[v_id] += c

                if edge in self.ban_edges:
                    for ban_edge in self.ban_edges[edge]:
                        ban_lists.remove(ban_edge)
        
        return None

    def solve(self):
        tmp = sum(self.value)
        if tmp % 2 == 1:
            return None

        result_edges = self._backtrack(0, [], [])
        if result_edges != None:
            change_edges = []
            for ((u, v), cnt) in result_edges:
                if cnt == 0:
                    continue
                change_edges.append(((self.rev_map[u], self.rev_map[v]), cnt))
            return change_edges
        return None
