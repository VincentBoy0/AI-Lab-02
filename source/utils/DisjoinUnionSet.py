class DSU:
    def __init__(self, n = 0):
        self.n = n
        self.par = [-1 for _ in range(n + 1)]
        self.connected_components = n
    
    def root(self, v):
        if self.par[v] < 0:
            return v
        self.par[v] = self.root(self.par[v])
        return self.par[v]

    def join(self, x, y):
        x = self.root(x)
        y = self.root(y)

        if x == y:
            return False
        if self.par[y] < self.par[x]:
            x, y = y, x
        self.par[x] += self.par[y]
        self.par[y] = x 
        self.connected_components -= 1
        return True
    
    def check_connected(self):
        return self.connected_components == 1