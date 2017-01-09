import itertools

class Clue:
  x = 0
  y = 0
  possibilities = []

  def __init__(self):
    pass

  def __str__(self):
    return '(x=%d y=%d possibilities=%s)' % (self.x, self.y, self.possibilities)

class Sudoku:
  def __init__(self, sudoku, diagonal=False):
    self.len = len(sudoku)
    for row in sudoku:
      if len(row) != self.len:
        raise ValueError("The sudoku is missing some values.")
    self.line = range(self.len)
    self.matrix = [[i // self.len, i % self.len] for i in range(self.len ** 2)]
    self.link_map = self._create_link_map(diagonal)
    self.depth_matrix = [[[float(len(self.link_map[i][j])), i, j] for j in self.line] for i in self.line]
    self.depth_line = list(itertools.chain.from_iterable(self.depth_matrix))
    k = max(e[0] for e in self.depth_line) + 2
    for e in self.depth_line:
      e[0] = self.len - e[0] / k
    # noinspection PyUnusedLocal
    self.x = [[list(range(-self.len, 0)) for j in self.line] for i in self.line]
    for i, j in self.matrix:
      value = sudoku[i][j]
      if value:
        self.set(value, i, j)

  def _create_link_map(self, diagonal=False):
    n_region = int(self.len ** .5)
    if n_region ** 2 != self.len:
      raise ValueError("Unsupported size of sudoku.")
    region = [[i // n_region, i % n_region] for i in self.line]
    m = []
    for i in self.line:
      column = []
      for j in self.line:
        ceil = []
        ceil.extend([[e, j] for e in self.line if e != i])
        ceil.extend([[i, e] for e in self.line if e != j])
        for a, b in region:
          x = a + i // n_region * n_region
          y = b + j // n_region * n_region
          if x != i and y != j:
            ceil.append([x, y])
        if diagonal:
          if i == j:
            ceil.extend([[e, e] for e in self.line if e != i])
          if i == self.len - j - 1:
            ceil.extend([[e, self.len - e - 1] for e in self.line if e != j])
        column.append(ceil)
      m.append(column)
    return m

  def set(self, value, x, y):
    if 0 < value <= self.len and -value in self.x[x][y]:
      self._set(-value, x, y)
      self.depth_line.remove(self.depth_matrix[x][y])
    else:
      raise ValueError('Failed to set %d to [%d;%d]!' % (value, y + 1, x + 1))
    self.depth_line.sort(key=lambda e: e[0])

  def clue(self):
    clue = Clue()
    clue.x = self.depth_line[0][1]
    clue.y = self.depth_line[0][2]
    clue.possibilities = [-e for e in self.x[clue.x][clue.y]]
    return clue

  def solve(self):
    solution = self._solve()
    self.x = solution
    return bool(solution)

  def _solve(self):
    if not self.depth_line:
      return self.x
    clue = self.depth_line[0]
    if not clue[0]:
      return None
    i, j = clue[1], clue[2]
    del self.depth_line[0]
    x_value = self.x[i][j]
    for value in x_value:
      log = []
      self._set(value, i, j, log)
      self.depth_line.sort(key=lambda e: e[0])
      if self._solve() is not None:
        return self.x
      for k in log:
        a, b = k >> 16, k & (1 << 16) - 1
        self.x[a][b].append(value)
        self.depth_matrix[a][b][0] += 1
    self.x[i][j] = x_value
    self.depth_line.insert(0, clue)
    self.depth_line.sort(key=lambda e: e[0])
    return None

  def _set(self, value, i, j, fallback=None):
    self.x[i][j] = [-value]
    for a, b in self.link_map[i][j]:
      try:
        self.x[a][b].remove(value)
        self.depth_matrix[a][b][0] -= 1
        if fallback is not None:
          fallback.append(a << 16 | b)
      except ValueError:
        pass

  @property
  def solution(self):
    return self.x

  @staticmethod
  def format(x):
    return ''.join([''.join([str(e[0]) for e in row]) for row in x])

nb_methodes = 1

# noinspection PyUnusedLocal
def resolution(A, methode=0):
  sudoku = Sudoku(A)
  if sudoku.solve():
    return [[[col[0] for col in row] for row in sudoku.solution]]
  return []
