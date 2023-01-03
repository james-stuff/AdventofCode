from collections import namedtuple


def manhattan_distance(point_1: (int,), point_2: (int,)) -> int:
    return sum([abs(d_2 - d_1) for d_1, d_2 in zip(point_1, point_2)])


Point = namedtuple("Point", "x y")
point_moves = {
    "U": lambda p: Point(p.x, p.y + 1),
    "D": lambda p: Point(p.x, p.y - 1),
    "L": lambda p: Point(p.x - 1, p.y),
    "R": lambda p: Point(p.x + 1, p.y),
}


def get_neighbours_in_grid(location: Point, grid: [[]]) -> [Point]:
    """list all neighbouring points (vertically or horizontally only) within a grid"""
    right_edge = len(grid[0]) - 1
    bottom_edge = len(grid) - 1
    def in_grid(p): return 0 <= p.x <= right_edge and 0 <= p.y <= bottom_edge
    return [*filter(lambda pt: in_grid(pt), [v(location) for v in point_moves.values()])]


def verify_solution(proposed: int, correct: int = None,  part_two: bool = False):
    part = "Two" if part_two else "One"
    print(f"\nPart {part} solution is {proposed}")
    if correct:
        assert proposed == correct



