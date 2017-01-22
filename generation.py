import random
import copy
import itertools

def init_matrix(size=9, value=0):
  state = []
  for i in xrange(size):
    state.append([value] * size)
  return state


def get_block(i, j):
  return i / 3 * 3 + j / 3


class Board(object):
  def __init__(self):
    self.game = init_matrix()
    self._possibilities = {}
    for i in xrange(9):
      for j in xrange(9):
        self._possibilities[(i, j)] = range(1, 10)

  def _recompute_possibilities(self):
    row_digits = init_matrix(size=10, value=1)
    column_digits = init_matrix(size=10, value=1)
    block_digits = init_matrix(size=10, value=1)
    for i in xrange(9):
      for j in xrange(9):
        cell = self.game[i][j]
        if cell:
          row_digits[i][cell] = 0
          column_digits[j][cell] = 0
          block_digits[get_block(i, j)][cell] = 0
    self._possibilities = {}
    for i in xrange(9):
      for j in xrange(9):
        if self.game[i][j]:
          continue
        possibilities = [1] * 10
        for k in xrange(1, 10):
          if not row_digits[i][k]:
            possibilities[k] = 0
          if not column_digits[j][k]:
            possibilities[k] = 0
          if not block_digits[get_block(i, j)][k]:
            possibilities[k] = 0
        temp = [x for x in xrange(1, 10) if possibilities[x]]
        self._possibilities[(i, j)] = temp

  def get(self, i, j):
    return self.game[i][j]

  def fill(self, i, j, value):
    self.game[i][j] = value
    self._recompute_possibilities()

  def clear(self, i, j):
    if self.game[i][j] == 0:
      return
    self.game[i][j] = 0
    self._recompute_possibilities()

  def get_possibilities(self, i, j):
    if self.game[i][j] != 0:
      raise ValueError('Cell is not empty.')

    if self._possibilities[(i, j)]:
      return self._possibilities[(i, j)]
    else:
      return None

  def __repr__(self):
    string = ''
    for index, row in enumerate(self.game):
      if index == 0:
        string += '\n'
      elif index % 3 == 0:
        string += '-------+-------+-------\n'
      row_string = ' '.join([str(x) if x else '_' for x in row])
      row_string = '%s | %s | %s' % (row_string[:5], row_string[6:11], row_string[12:])
      string += ' %s\n' % row_string
    return string

  def _is_conflicting(self, i1, j1, i2, j2):
    if self.game[i1][j1] == 0 or self.game[i2][j2]:
      return False
    if self.game[i1][j1] != self.game[i2][j2]:
      return False
    if i1 == i2:
      return True
    if j1 == j2:
      return True
    if get_block(i1, j1) == get_block(i2, j2):
      return True
    return False

  def is_valid(self):
    for i1 in xrange(9):
      for j1 in xrange(9):
        for i2 in xrange(9):
          for j2 in xrange(9):
            if (i1, j1) == (i2, j2):
              continue
            if self._is_conflicting(i1, j1, i2, j2):
              return False
    return True


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
    if isinstance(sudoku, Board):
      sudoku = sudoku.game
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

  @property
  def board(self):
    b = Board()
    for i in xrange(9):
      for j in xrange(9):
        b.fill(i, j, self.solution[i][j][0])
    return b

  @staticmethod
  def format(x):
    return ''.join([''.join([str(e[0]) for e in row]) for row in x])


class Digger(object):
  def __init__(self, digging_strategy):
    self.digging_strategy = digging_strategy

  def dig_cells(self, terminal_pattern):
    board = copy.deepcopy(terminal_pattern)
    dig_count = 0
    for (i, j) in self.digging_strategy.cells:
      if not self.digging_strategy.can_dig(board, i, j):
        continue
      prev_value = board.get(i, j)
      board.clear(i, j)
      possibilities = board.get_possibilities(i, j)
      possibilities.remove(prev_value)
      has_another_solution = False
      for new_value in possibilities:
        board.fill(i, j, new_value)
        sudoku = Sudoku(board)
        if sudoku.solve():
          has_another_solution = True
          break
      if has_another_solution:
        board.fill(i, j, prev_value)
      else:
        board.clear(i, j)
        dig_count += 1
      if dig_count >= self.digging_strategy.limit:
        return board
    return board


class DiggingStrategy(object):
  def __init__(self, difficulty):
    self.cells = []
    if not isinstance(difficulty, int):
      raise ValueError('invalid difficulty argument: expected int')
    if difficulty == 1:
      self.generate_randomized_cells()
      self.max_empty_cells = 4
      self.limit = 31
    elif difficulty == 2:
      self.generate_randomized_cells()
      self.max_empty_cells = 5
      self.limit = 45
    elif difficulty == 3:
      self.generate_jumping_once_cell()
      self.max_empty_cells = 6
      self.limit = 49
    elif difficulty == 4:
      self.generate_wandering_along_s()
      self.max_empty_cells = 7
      self.limit = 53
    elif difficulty == 5:
      self.generate_ordered_cells()
      self.max_empty_cells = 9
      self.limit = 59
    else:
      raise ValueError('invalid difficulty level: %d' % difficulty)

  def can_dig(self, board, i, j):
    nr_empty_cells = 0
    for k in xrange(9):
      if board.get(i, k) == 0:
        nr_empty_cells += 1
    if nr_empty_cells >= self.max_empty_cells:
      return False
    nr_empty_cells = 0
    for k in xrange(9):
      if board.get(k, j) == 0:
        nr_empty_cells += 1
    if nr_empty_cells >= self.max_empty_cells:
      return False
    return True

  def generate_ordered_cells(self):
    for i in xrange(9):
      for j in xrange(9):
        self.cells.append((i, j))

  def generate_randomized_cells(self):
    self.generate_ordered_cells()
    random.shuffle(self.cells)

  def generate_wandering_along_s(self):
    self.cells = []
    for i in xrange(9):
      for j in xrange(9):
        if i % 2 == 0:
          cell = (i, j)
        else:
          cell = (i, 8 - j)
        self.cells.append(cell)

  def generate_jumping_once_cell(self):
    self.generate_wandering_along_s()
    temp = []
    for i in xrange(0, len(self.cells), 2):
      temp.append(self.cells[i])
    for i in xrange(1, len(self.cells), 2):
      temp.append(self.cells[i])
    self.cells = temp


def _backtracking(board, givens, position):
  if position >= len(givens):
    return board
  i, j = givens[position]
  possibilities = board.get_possibilities(i, j)
  random.shuffle(possibilities)
  for value in possibilities:
    board.fill(i, j, value)
    solution = _backtracking(board, givens, position + 1)
    if solution:
      return solution
  return None


def las_vegas(givens_count=11):
  all_positions = []
  for i in xrange(9):
    for j in xrange(9):
      all_positions.append((i, j))
  givens = random.sample(all_positions, givens_count)
  partial_game = Board()
  partial_game = _backtracking(partial_game, givens, 0)
  sudoku = Sudoku(partial_game)
  if sudoku.solve():
    return sudoku.board
  return False


def generate_terminal_pattern():
  while True:
    terminal_pattern = las_vegas(givens_count=11)
    if terminal_pattern:
      return terminal_pattern


def generation():
  board = generate_terminal_pattern()
  digger = Digger(DiggingStrategy(5))
  return [digger.dig_cells(board).game, board.game]
