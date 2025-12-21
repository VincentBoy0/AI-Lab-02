import heapq
import sys

class AStar:
    def __init__(self, grid):
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        self.islands = []
        self.edges = [] 
        
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] > 0:
                    self.islands.append({'r': r, 'c': c, 'val': grid[r][c]})
        
        self._find_potential_edges()
        
        self.conflict_map = {i: [] for i in range(len(self.edges))}
        self._build_conflict_map()

    def _find_potential_edges(self):
        for i, u in enumerate(self.islands):
            r_u, c_u = u['r'], u['c']

            for j in range(i + 1, len(self.islands)):
                v = self.islands[j]
                if v['r'] == r_u and v['c'] > c_u:
                    is_blocked = False
                    for c_k in range(c_u + 1, v['c']):
                        if self.grid[r_u][c_k] > 0:
                            is_blocked = True; break
                    if not is_blocked:
                        self.edges.append({'u': i, 'v': j, 'type': 'H', 
                                           'r': r_u, 'c1': c_u, 'c2': v['c']})
                    break 

            closest_v = -1; min_dist = float('inf')
            for j in range(len(self.islands)):
                if i == j: continue
                v = self.islands[j]
                if v['c'] == c_u and v['r'] > r_u:
                    if v['r'] < min_dist:
                        min_dist = v['r']; closest_v = j
            
            if closest_v != -1:
                v = self.islands[closest_v]
                is_blocked = False
                for r_k in range(r_u + 1, v['r']):
                    if self.grid[r_k][c_u] > 0:
                        is_blocked = True; break
                if not is_blocked:
                    self.edges.append({'u': i, 'v': closest_v, 'type': 'V',
                                       'c': c_u, 'r1': r_u, 'r2': v['r']})

    def _build_conflict_map(self):
        for j in range(len(self.edges)):
            edge_j = self.edges[j]
            for i in range(j):
                edge_i = self.edges[i]
                if edge_i['type'] == edge_j['type']: continue
                
                h = edge_i if edge_i['type'] == 'H' else edge_j
                v = edge_j if edge_i['type'] == 'H' else edge_i
                
                if (v['r1'] < h['r'] < v['r2']) and (h['c1'] < v['c'] < h['c2']):
                    self.conflict_map[j].append(i)

    def solve(self):
        num_edges = len(self.edges)
        num_islands = len(self.islands)
        
        start_usage = tuple([0] * num_islands)
        pq = [(0, 0, 0, 0, start_usage)]
        
        visited = set()
        visited.add((0, 0))
        
        while pq:
            f, g, idx, state_int, usage = heapq.heappop(pq)
            
            if idx == num_edges:
                valid_counts = True
                for i in range(num_islands):
                    if usage[i] != self.islands[i]['val']:
                        valid_counts = False; break
                if not valid_counts: continue

                if self._check_connectivity(state_int):
                    return self._format_output(state_int)
                continue

            edge = self.edges[idx]
            u, v = edge['u'], edge['v']
            
            for bridges in [0, 1, 2]:
                if usage[u] + bridges > self.islands[u]['val']: continue
                if usage[v] + bridges > self.islands[v]['val']: continue
                
                is_crossing = False
                if bridges > 0:
                    for prev_idx in self.conflict_map[idx]:
                        prev_bridges = (state_int // (3**prev_idx)) % 3
                        if prev_bridges > 0:
                            is_crossing = True; break
                if is_crossing: continue

                new_state_int = state_int + (bridges * (3**idx))
                
                state_key = (idx + 1, new_state_int)
                
                if state_key in visited:
                    continue 
                
                visited.add(state_key)

                new_usage_list = list(usage)
                new_usage_list[u] += bridges
                new_usage_list[v] += bridges
                new_usage = tuple(new_usage_list)
                
                new_g = g + 1
                heapq.heappush(pq, (new_g, new_g, idx + 1, new_state_int, new_usage))
        
        return []

    def _check_connectivity(self, state_int):
        adj = {i: [] for i in range(len(self.islands))}
        count_bridges = 0
        for idx, edge in enumerate(self.edges):
            bridges = (state_int // (3**idx)) % 3
            if bridges > 0:
                count_bridges += 1
                adj[edge['u']].append(edge['v'])
                adj[edge['v']].append(edge['u'])
        
        if count_bridges == 0 and len(self.islands) > 1: return False
        
        visited = set([0])
        queue = [0]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(self.islands)

    def _format_output(self, state_int):
        result_list = []
        for idx, edge in enumerate(self.edges):
            bridges = (state_int // (3**idx)) % 3
            if bridges > 0:
                isl_u = self.islands[edge['u']]
                isl_v = self.islands[edge['v']]
                pos_pair = ((isl_u['r'], isl_u['c']), (isl_v['r'], isl_v['c']))
                result_list.append((pos_pair, bridges))
        return result_list
