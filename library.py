from collections import namedtuple
from itertools import product
import inspect
import re
import os


memo_pad = {}


def load(text: str = "") -> str:
    caller = inspect.currentframe()
    while not re.search(r"aoc_\d{4}.py$", caller.f_code.co_filename):
        caller = caller.f_back
    filename = caller.f_code.co_filename
    function_name = caller.f_code.co_name
    m_year, m_day = (
        re.search(pattern, string)
        for pattern, string in zip(
                [r"\d{4}.py", r"_\d+_"],
                [filename, function_name]
        )
    )
    if m_year and m_day:
        day = m_day.group()[1:-1]
        clear_memos(day)
        if text:
            return text
        folder = f"inputs\\{m_year.group()[:4]}"
        possible_files = [
            *filter(
                lambda fn: re.match(r"\D+" + f"{day}.txt$", fn),
                os.listdir(folder)
            )
        ]
        if possible_files and len(possible_files) == 1:
            with open(f"{folder}\\{possible_files[0]}") as input_file:
                return input_file.read().strip("\n")
    return ""


def clear_memos(day: str):
    global memo_pad
    for memo in memo_pad:
        if day in memo:
            var_type = type(memo_pad[memo])
            if memo_pad[memo]:
                memo_pad[memo] = var_type()


def load_grid(raw_text: str, exclude_chars: str = "#") -> dict:
    raw_text = load(raw_text)
    return {
        Point(y, x): char
        for y, row in enumerate(raw_text.split("\n"))
        for x, char in enumerate(row)
        if char not in exclude_chars
    }


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


def get_all_neighbours_in_grid(location: Point, grid: [[]]) -> [Point]:
    """list all eight neighbouring points within a grid"""
    right_edge = len(grid[0]) - 1
    bottom_edge = len(grid) - 1
    def in_grid(p): return 0 <= p.x <= right_edge and 0 <= p.y <= bottom_edge
    return [*filter(lambda pt: in_grid(pt),
                    [Point(location.x + x_diff, location.y + y_diff)
                     for x_diff, y_diff in product(range(-1, 2), repeat=2)
                    if (x_diff, y_diff) != (0, 0)]
                    )]


def verify_solution(proposed: int, correct: int = None,  part_two: bool = False):
    part = "Two" if part_two else "One"
    print(f"\nPart {part} solution is {proposed}")
    if correct:
        assert proposed == correct



