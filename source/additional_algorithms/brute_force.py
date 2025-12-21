import sys

class BruteForce:
    def __init__(self, grid):
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.original_grid = grid
        self.islands = []
        self.edges = [] 

        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] > 0:
                    self.islands.append({'r': r, 'c': c, 'val': grid[r][c]})

        self._find_potential_edges()

    def _find_potential_edges(self):
        for i, u in enumerate(self.islands):
            r_u, c_u = u['r'], u['c']

            for j in range(i + 1, len(self.islands)):
                v = self.islands[j]
                if v['r'] == r_u and v['c'] > c_u:
                    is_blocked = False
                    for c_k in range(c_u + 1, v['c']):
                        if self.original_grid[r_u][c_k] > 0:
                            is_blocked = True
                            break
                    
                    if not is_blocked:
                        self.edges.append({'u': i, 'v': j, 'type': 'H'})
                    break 

            closest_v_idx = -1
            min_dist = float('inf')
            
            for j in range(len(self.islands)):
                if i == j: continue
                v = self.islands[j]
                if v['c'] == c_u and v['r'] > r_u:
                    if v['r'] < min_dist:
                        min_dist = v['r']
                        closest_v_idx = j
            
            if closest_v_idx != -1:
                v = self.islands[closest_v_idx]
                is_blocked = False
                for r_k in range(r_u + 1, v['r']):
                    if self.original_grid[r_k][c_u] > 0:
                        is_blocked = True
                        break
                
                if not is_blocked:
                    self.edges.append({'u': i, 'v': closest_v_idx, 'type': 'V'})

    def solve(self):
        num_edges = len(self.edges)
        total_combinations = 3 ** num_edges
        
        for k in range(total_combinations):
            config = []
            temp = k
            for _ in range(num_edges):
                config.append(temp % 3)
                temp //= 3
            
            if self._is_valid(config):

                result_list = []
                for idx, count in enumerate(config):
                    if count > 0:
                        edge = self.edges[idx]
                        island_u = self.islands[edge['u']]
                        island_v = self.islands[edge['v']]
                        
                        pos_pair = ((island_u['r'], island_u['c']), (island_v['r'], island_v['c']))
                        
                        result_list.append((pos_pair, count))
                
                return result_list

        return [] 

    def _is_valid(self, config):
        island_bridges = [0] * len(self.islands)
        

        h_segments = []
        v_segments = []

        for idx, count in enumerate(config):
            if count > 0:
                edge = self.edges[idx]
                island_bridges[edge['u']] += count
                island_bridges[edge['v']] += count
                
                u_node = self.islands[edge['u']]
                v_node = self.islands[edge['v']]
                
                if edge['type'] == 'H':
                    h_segments.append((u_node['r'], u_node['c'], v_node['c']))
                else:
                    v_segments.append((u_node['c'], u_node['r'], v_node['r']))

        for i, val in enumerate(island_bridges):
            if val != self.islands[i]['val']:
                return False
        

        for h in h_segments: 
            for v in v_segments: 

                if (v[1] < h[0] < v[2]) and (h[1] < v[0] < h[2]):
                    return False


        adj = {i: [] for i in range(len(self.islands))}
        for idx, count in enumerate(config):
            if count > 0:
                u, v = self.edges[idx]['u'], self.edges[idx]['v']
                adj[u].append(v)
                adj[v].append(u)
        
        visited = set()
        stack = [0]
        visited.add(0)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        
        if len(visited) != len(self.islands):
            return False

        return True
