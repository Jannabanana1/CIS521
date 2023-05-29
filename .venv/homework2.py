from math import comb
import random
import copy

############################################################
# CIS 521: Homework 2
############################################################

student_name = "Jannatul Ferdaus"


############################################################
# Section 1: N-Queens
############################################################


def num_placements_all(n):
    x = comb(n * n, n)
    return x


def num_placements_one_per_row(n):
    return n ** n


def n_queens_valid(board):
    if len(set(board)) < len(board):
        return False
    for i in range(len(board)):
        for j in range(i + 1, len(board)):
            if j - i == abs(board[i] - board[j]):
                return False
    return True


def func(n):
    for i in range(n):
        for solution in n_queens_helper(n, [i]):
            yield solution


def n_queens_helper(n, board):
    if n_queens_valid(board):
        if len(board) == n:
            yield board
        else:
            for i in [col for col in range(n) if col not in board]:
                if (i != board[-1] and i != board[-1] + 1 and
                        i != board[-1] - 1):
                    newBoard = list(board)
                    newBoard.append(i)
                    for solution in n_queens_helper(n, newBoard):
                        if solution:
                            yield solution


def n_queens_solutions(x):
    return list(func(x))


############################################################
# Section 2: Lights Out
############################################################

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board
        self.row = len(board)
        self.column = len(board[0])
        self.movement = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]

    def get_board(self):
        return self.board

    def perform_move(self, row, col):
        if row >= 0 and row < self.row and col >= 0 and col < self.column:
            for x in self.movement:
                trow, tcol = row + x[0], col + x[1]
                if trow < 0 or tcol < 0 or trow == self.row \
                        or tcol == self.column:
                    continue
                self.board[trow][tcol] = not self.board[trow][tcol]

    def scramble(self):
        for i in range(self.row):
            for j in range(self.column):
                if random.random() < 0.5:
                    self.perform_move(i, j)

    def is_solved(self):
        for row in self.board:
            for elem in row:
                if elem:
                    return False
        return True

    def copy(self):
        new_board = copy.deepcopy(self.board)
        return LightsOutPuzzle(new_board)

    def successors(self):
        for i in range(self.row):
            for j in range(self.column):
                copy_p = self.copy()
                copy_p.perform_move(i, j)
                yield ((i, j), copy_p)

    def toTuple(self):
        board = tuple([tuple(row) for row in self.board])
        return board

    def find_solution(self):
        if self.is_solved():
            return []
        board_set = set()
        board_set.add(self.toTuple())
        front = 0
        end = 0
        queue = [[(-1, -1), self.copy(), -1]]
        boolean = False
        while (end <= front):
            p = queue[end][1]
            for move, new_p in p.successors():
                board = new_p.toTuple()
                if (board not in board_set):
                    board_set.add(board)
                    front += 1
                    queue.append([move, new_p, end])
                if new_p.is_solved():
                    boolean = True
                    break
            end += 1
            if boolean:
                break
        if boolean is False:
            return None
        sequence = []
        parent = front
        while True:
            if parent == 0:
                return list(reversed(sequence))
            sequence.append(queue[parent][0])
            parent = queue[parent][2]


def create_puzzle(rows, cols):
    board = [([False for j in range(cols)]) for i in range(rows)]
    return LightsOutPuzzle(board)


############################################################
# Section 3: Linear Disk Movement
############################################################

class LinearDisk(object):
    def __init__(self, cells, n):
        self.cells = cells
        self.length = len(cells)
        self.n = n

    def get_cells(self):
        return self.cells

    def perform_move(self, i, j):
        if i >= 0 and i < self.length and j >= 0 and j < self.length:
            self.cells[i], self.cells[j] = self.cells[j], self.cells[i]

    def is_solved(self):
        for i in range(self.n):
            if self.cells[self.length - i - 1] is False:
                return False
        return True

    def check_order(self):
        for i in range(self.n):
            if self.cells[self.length - i - 1] > (i + 1):
                return False
        return True

    def copy(self):
        new_cells = copy.deepcopy(self.cells)
        return LinearDisk(new_cells, self.n)

    def successors(self):
        c = self.cells
        length = self.length
        for i in range(length):
            if c[i] is True and i < length - 1:
                if c[i + 1] is False:
                    copy_c = self.copy()
                    copy_c.perform_move(i, i + 1)
                    yield ((i, i + 1), copy_c)
            if c[i] is True and i < length - 2:
                if c[i + 1] is True and c[i + 2] is False:
                    copy_c = self.copy()
                    copy_c.perform_move(i, i + 2)
                    yield ((i, i + 2), copy_c)


def solve_identical_disks(length, n):
    c = [True for i in range(n)] + [False for j in range(n, length)]
    disk = LinearDisk(c, n)
    if disk.is_solved():
        return []
    cells_set = set()
    cells_set.add(tuple(elem for elem in disk.get_cells()))
    front = 0
    back = 0
    queue = [[(0, 0), disk.copy(), -1]]
    is_solved = False
    while back <= front:
        p = queue[back][1]
        for move, new_p in p.successors():
            cells = tuple(elem for elem in new_p.get_cells())
            if cells not in cells_set:
                cells_set.add(cells)
                front += 1
                queue.append([move, new_p, back])
                if new_p.is_solved():
                    is_solved = True
                    break
        back += 1
        if is_solved:
            break
    if not is_solved:
        return None
    result = []
    parent = front
    from_to = []
    while True:
        if parent == 0:
            from_to = list(reversed(result))
            break
        result.append(queue[parent][0])
        parent = queue[parent][2]
    return from_to


class LinearDiskPart2(object):
    def __init__(self, cells, n):
        self.cells = cells
        self.length = len(cells)
        self.n = n

    def get_cells(self):
        return self.cells

    def perform_move(self, i, j):
        if i >= 0 and i < self.length and j >= 0 and j < self.length:
            self.cells[i], self.cells[j] = self.cells[j], self.cells[i]

    def is_solved(self):
        for i in range(self.n):
            if self.cells[self.length - i - 1] != i:
                return False
        return True

    def check_order(self):
        for i in range(self.n):
            if self.cells[self.length - i - 1] > i:
                return False
        return True

    def copy(self):
        new_cells = copy.deepcopy(self.cells)
        return LinearDiskPart2(new_cells, self.n)

    def successors(self):
        c = self.cells
        length = self.length
        for i in range(length):
            if c[i] >= 0:
                if i < length - 1:
                    if c[i + 1] < 0:
                        copy_c = self.copy()
                        copy_c.perform_move(i, i + 1)
                        yield ((i, i + 1), copy_c)
                if i < length - 2:
                    if c[i + 2] < 0 and c[i + 1] >= 0:
                        copy_c = self.copy()
                        copy_c.perform_move(i, i + 2)
                        yield ((i, i + 2), copy_c)
                if i >= 1:
                    if c[i - 1] < 0:
                        copy_c = self.copy()
                        copy_c.perform_move(i, i - 1)
                        yield ((i, i - 1), copy_c)
                if i >= 2:
                    if c[i - 2] < 0 and c[i - 1] >= 0:
                        copy_c = self.copy()
                        copy_c.perform_move(i, i - 2)
                        yield ((i, i - 2), copy_c)


def solve_distinct_disks(length, n):
    c = [i for i in range(n)] + [-1 for j in range(n, length)]
    disk = LinearDiskPart2(c, n)
    if disk.is_solved():
        return []
    cells_set = set()
    cells_set.add(tuple(elem for elem in disk.get_cells()))
    front = 0
    end = 0
    queue = [[(0, 0), disk.copy(), -1]]
    is_solved = False
    while end <= front:
        p = queue[end][1]
        for move, new_p in p.successors():
            cells = tuple(elem for elem in new_p.get_cells())
            if cells not in cells_set:
                cells_set.add(cells)
                front += 1
                queue.append([move, new_p, end])
                if new_p.is_solved():
                    is_solved = True
                    break
        end += 1
        if is_solved:
            break
    if not is_solved:
        return None
    items = []
    parent = front
    from_to = []
    while True:
        if parent == 0:
            from_to = list(reversed(items))
            break
        items.append(queue[parent][0])
        parent = queue[parent][2]
    return from_to


############################################################
# Section 4: Feedback
############################################################


feedback_question_1 = """
I spent 20 hours
"""

feedback_question_2 = """
using dfs/bfs was the most challenging as well as using the GUI
"""

feedback_question_3 = """
I would not use tkinter for GUI since it does not work for everyone.
I liked the games idea and the breadth first search idea since it
helped me understand it better. 
"""