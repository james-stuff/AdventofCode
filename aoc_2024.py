import copy
import math
from main import Puzzle
import re
import library as lib
from aoc_2023 import point_moves_2023 as pm23
import itertools
import operator


class Puzzle24(Puzzle):
    def get_text_input(self) -> str:
        with open(f'inputs\\2024\\input{self.day}.txt', 'r') as input_file:
            return input_file.read()


def day_8_map_from_input(text: str = "") -> {}:
    if not text:
        text = Puzzle24(8).get_text_input().strip("\n")
    return {
        lib.Point(y, x): ch
        for y, row in enumerate(text.split("\n"))
        for x, ch in enumerate(row)
    }


def day_8_part_one(text: str = "") -> int:
    antinodes = set()
    m = day_8_map_from_input(text)
    node_types = {*filter(lambda v: v != ".", m.values())}
    for nt in node_types:
        node_locs = [k for k, v in m.items()
                     if v == nt]
        for i, nl in enumerate(node_locs):
            for j, other_node in enumerate(node_locs[i + 1:]):
                v_diff, h_diff = (other_node[n] - nl[n] for n in range(2))
                outer_1 = lib.Point(
                    *(pt + d for d, pt in zip((v_diff, h_diff), max(nl, other_node))
                ))
                outer_2 = lib.Point(
                    *(pt - d for d, pt in zip((v_diff, h_diff), min(nl, other_node))
                ))
                # print(f"{nl} -> {outer_1=} {outer_2=}")
                if all(dd % 3 == 0 for dd in (v_diff, h_diff)):
                    start = min(nl, other_node)
                    inner1 = lib.Point(
                        start[0] + (v_diff // 3),
                        start[1] + (h_diff // 3)
                    )
                    inner2 = lib.Point(
                        start[0] + (2 * (v_diff // 3)),
                        start[1] + (2 * (h_diff // 3))
                    )
                    # print(f"{nl} -> {inner1=} {inner2=}")
                    antinodes.update((inner1, inner2))
                antinodes.update(
                    (
                        on for on in (outer_1, outer_2)
                        if on in m
                    ))
    return len(antinodes)


def day_8_part_two(text: str = "") -> int:
    m = day_8_map_from_input(text)
    y_max, _ = max(m)
    antinodes = set()
    node_types = {*filter(lambda v: v not in ".#", m.values())}
    for nt in node_types:
        node_locs = [k for k, v in m.items()
                     if v == nt]
        for i, nl in enumerate(node_locs):
            for j, other_node in enumerate(node_locs[i + 1:]):
                antinodes.update(
                    day_8_two_points_to_line(nl, other_node, y_max)
                )
    return len(antinodes)


def day_8_two_points_to_line(
        pt1: lib.Point, pt2: lib.Point, limit: int) -> [lib.Point]:
    line = [lib.Point(*pt1)]
    v_to_h_ratio = tuple((pt2[c] - pt1[c] for c in range(2)))
    if all(dd != 0 for dd in v_to_h_ratio):
        v_to_h_ratio = tuple(
            (f // math.gcd(*v_to_h_ratio) for f in v_to_h_ratio)
        )
    for ratio in (v_to_h_ratio, (-rn for rn in v_to_h_ratio)):
        v_step, h_step = ratio
        new_pt = lib.Point(*pt1)
        while True:
            new_pt = lib.Point(new_pt[0] + v_step, new_pt[1] + h_step)
            if all(0 <= new_pt[c] <= limit for c in range(2)):
                line.append(new_pt)
            else:
                break
    return line


def day_7_load_line(line: str) -> (int, [int]):
    target, _, components = line.partition(": ")
    return int(target), [int(c) for c in components.split(" ")]


def day_7_load(text: str = "") -> {}:
    if not text:
        text = Puzzle24(7).get_text_input().strip("\n")
    return {
        k: v
        for k, v in [day_7_load_line(ln) for ln in text.split("\n")]
    }


def day_7_computes(target: int, components: [int],
                   with_concatenation: bool = False) -> bool:
    possible_ops = "*+"
    if with_concatenation:
        possible_ops += "|"
    for operators in itertools.product(
            possible_ops, repeat=len(components) - 1
    ):
        cumulative = components[0]
        for i, c in enumerate(components[1:]):
            op = operators[i]
            if op == "*":
                cumulative *= c
            elif op == "+":
                cumulative += c
            else:
                cumulative = int(f"{cumulative}{c}")
        if cumulative == target:
            return True
    return False


def day_7_part_one(text: str = "") -> int:
    return sum(k for k, v in day_7_load(text).items()
               if day_7_computes(k, v))


def day_7_part_two(text: str = "") -> int:
    input_data = day_7_load(text)
    cumulative = 0
    for k, v in input_data.items():
        if day_7_computes(k, v):
            cumulative += k
        else:
            if day_7_computes(k, v, with_concatenation=True):
                cumulative += k
    return cumulative


def day_6_load_map(text: str = "") -> {}:
    if not text:
        text = Puzzle24(6).get_text_input().strip("\n")
    return {
        lib.Point(y, x): char
        for y, row in enumerate(text.split("\n"))
        for x, char in enumerate(row)
    }


def day_6_part_one(text: str = "") -> int:
    return len({vl[0] for vl in day_6_visited_locations_with_facing(text)})


def day_6_visited_locations_with_facing(map_text: str) -> [(lib.Point, str)]:
    room = day_6_load_map(map_text)
    guard_pos = [k for k, v in room.items() if v == "^"][0]
    facing = "U"
    visited = [(guard_pos, facing)]
    room[guard_pos] = "."
    while guard_pos in room:
        loc_in_front = pm23[facing](guard_pos)
        if loc_in_front in room:
            if room[loc_in_front] == ".":
                guard_pos = loc_in_front
                visited.append((guard_pos, facing))
            else:
                facing = day_6_turn_to_right(facing)
        else:
            guard_pos = loc_in_front
    return visited


def day_6_distance_to_next_blocker(point: lib.Point,
                                   facing: str, area: {}) -> int:
    co_ord = 0 if facing in "UD" else 1
    fixed = not co_ord
    comparison = operator.gt if facing in "DR" else operator.lt
    blockers = [pt[co_ord] for pt, char in area.items()
                if pt[fixed] == point[fixed]
                and comparison(pt[co_ord], point[co_ord])
                and char == "#"]
    if blockers:
        distance_func = (lambda p: min(blockers) - p) \
            if facing in "DR" else (lambda p: p - max(blockers))
        return distance_func(point[co_ord])
    return 1_000_000


def day_6_back_up(point: lib.Point, facing: str) -> lib.Point:
    """Go back one step in the opposite direction"""
    directions = "URDL"
    return pm23[directions[(directions.index(facing) + 2) % 4]](point)


def day_6_turn_to_right(facing: str) -> str:
    """Change the facing to rotate 90 degrees to the right"""
    directions = "URDL"
    return directions[(directions.index(facing) + 1) % 4]


def day_6_move_n_steps_in_direction(
        point: lib.Point, facing: str, steps: int) -> lib.Point:
    f_dict = {
        "U": (-1, 0),
        "R": (0, 1),
        "D": (1, 0),
        "L": (0, -1),
    }
    delta = (steps * dd for dd in f_dict[facing])
    return lib.Point(*(c + d for c, d in zip(point, delta)))


def day_6_part_two(text: str = "") -> int:
    """Idea: it's only worth placing an obstacle in the
            area the guard can already reach (so get the
            set of visited points)"""
    # (8, 3) addition proves it's not just a rectangle
    m = day_6_load_map(text)
    new_obstacles = 0
    # a point may have been visited multiple times in
    # part one.  So only deal with its first appearance
    all_visited_data = day_6_visited_locations_with_facing(text)
    starting_point = all_visited_data[0][0]
    visited_points = {pt for pt, _ in all_visited_data
                      if pt != starting_point}
    m[starting_point] = "."
    pf_will_leave = set()
    known_routes_to_exit = []
    len_vp, point_counter = len(visited_points), 0
    for point in visited_points:
        point_counter += 1
        modded_map = copy.deepcopy(m)
        modded_map[point] = "#"
        _, f = [vp for vp in all_visited_data if vp[0] == point][0]
        # print(f"Point: {point}, originally facing {f}")
        obstacle_point = point
        pt = day_6_back_up(point, f)
        f = day_6_turn_to_right(f)
        will_visit = []
        no_escape = False
        while pt in modded_map:
            dist = day_6_distance_to_next_blocker(pt, f, modded_map) - 1
            pt = day_6_move_n_steps_in_direction(pt, f, dist)
            if (pt, f) in will_visit:
                new_obstacles += 1
                break
            will_visit.append((pt, f))
            if len(will_visit) > 3:
                last_three = will_visit[-3:]
                for kre in known_routes_to_exit:
                    if any(
                        kre[i:i + 3] == last_three
                        for i, loc in enumerate(kre[:-3])
                    ):
                        spot = kre.index(last_three[0])
                        # print(f"Found repeated route!! {last_three=} found={kre[spot:spot + 3]}")
                        if day_6_escape_route_blocked(kre[spot:], obstacle_point):
                            # print(" . . . but the route is blocked :(")
                            no_escape = True
                        else:
                            pt = (-1, -1), ""
                            break
            # if (pt, f) in pf_will_leave:
            #     print(f"\t{pt, f} is known to leave grid")
            #     break
            f = day_6_turn_to_right(f)
        if dist > 10_000:
            if all(
                lib.manhattan_distance(wv[0], obstacle_point) > 1
                for wv in will_visit[1:]
            ):
                known_routes_to_exit.append(will_visit)
                # pf_will_leave.update(will_visit[1:])
        if point_counter % 100 == 0:
            print(f"{point_counter:,} of {len_vp} points. "
                  f"{len(will_visit):,} visited before "
                  f"{'leaving grid' if dist > 10_000 else 'looping'} "
                  f"{len(known_routes_to_exit)=}")
    assert 0 < new_obstacles < day_6_part_one(text)
    return new_obstacles


def day_6_escape_route_blocked(route: [(lib.Point, str)],
                               blocker: lib.Point) -> bool:
    for i, loc in enumerate(route[:-1]):
        pt1, pt2 = (x[0] for x in (loc, route[i + 1]))
        if (pt1[0] < blocker[0] < pt2[0] or
                pt2[0] < blocker[0] < pt1[0]) and pt1[1] == blocker[1]:
            return True
        if (pt1[1] < blocker[1] < pt2[1] or
                pt2[1] < blocker[1] < pt1[1]) and pt1[0] == blocker[0]:
            return True
    return False


def day_5_load(text: str = "") -> ([(str,)],):
    if not text:
        text = Puzzle24(5).get_text_input().strip("\n")
    rules = [
        tuple(m.group().split("|"))
        for m in re.finditer(r"\d+\|\d+", text)
    ]
    updates = [
        um.group().split(",")
        for um in re.finditer(r"(\d+,)+\d+", text)
    ]
    return rules, updates


def day_5_is_valid(update: [str,], rules: [(str,)]) -> bool:
    for i, page_no in enumerate(update):
        following_pages = update[i + 1:]
        if any(
                (page_no == right) and (left in following_pages)
                for left, right in rules
        ):
            return False
    return True


def day_5_sum_of_middle_pages(booklets: [str,]) -> int:
    return sum(int(bklt[len(bklt) // 2]) for bklt in booklets)


def day_5_part_one(text: str = "") -> int:
    rules, updates = day_5_load(text)
    valid = [*filter(lambda u: day_5_is_valid(u, rules), updates)]
    return day_5_sum_of_middle_pages(valid)


def day_5_re_order_to_valid(pages: [str], rules: [(str,)]) -> [str]:
    print(f"Re-ordering {pages}")
    if len(pages) == 1:
        return pages
    left = 0
    for nn in pages:
        the_rest = [v for v in pages if v != nn]
        if not any((vr, nn) in rules for vr in the_rest):
            print(f"{nn} can safely be the leftmost page")
            left = nn
            break
    re_ordered = [left] + day_5_re_order_to_valid(
        [v for v in pages if v != left], rules
    )
    assert day_5_is_valid(re_ordered, rules)
    return re_ordered


def day_5_part_two(text: str = "") -> int:
    rules, updates = day_5_load(text)
    invalid = [*filter(lambda u: not day_5_is_valid(u, rules), updates)]
    return day_5_sum_of_middle_pages(
        day_5_re_order_to_valid(iv, rules) for iv in invalid
    )


def day_4_load_grid(text: str) -> {}:
    return {
        lib.Point(y, x): letter
        for y, row in enumerate(text.split("\n"))
        for x, letter in enumerate(row)
    }


def day_4_matches(x_loc: lib.Point, grid: {}) -> int:
    """count the number of wordsearch matches in the grid that start
        from the given location"""
    directions = ["U", "UL", "L", "DL", "D", "DR", "R", "UR"]
    target = "XMAS"
    matches = 0
    for d in directions:
        found_word = "X"
        next_co_ord = x_loc
        for step in range(3):
            for move in d:
                next_co_ord = pm23[move](next_co_ord)
            if next_co_ord not in grid:
                break
            found_word += grid[next_co_ord]
            if found_word != target[:step + 2]:
                break
        matches += (found_word == target)
    return matches


def day_4_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle24(4).get_text_input().strip("\n")
    grid = day_4_load_grid(text)
    x_s = [*filter(lambda k: grid[k] == "X", grid.keys())]
    return sum(day_4_matches(x, grid) for x in x_s)


def day_4_x_mas_match(a_loc: lib.Point, grid: {}) -> bool:
    directions = [("UL", "DR"), ("UR", "DL")]
    for diagonal in directions:
        diag_letters = ""
        for d in diagonal:
            point = a_loc
            for move in d:
                point = pm23[move](point)
            if point not in grid:
                return False
            diag_letters += grid[point]
        if set(diag_letters) != set("MS"):
            return False
    return True


def day_4_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle24(4).get_text_input().strip("\n")
    grid = day_4_load_grid(text)
    a_s = [*filter(lambda k: grid[k] == "A", grid.keys())]
    return sum(day_4_x_mas_match(a, grid) for a in a_s)


def day_3_sum_of_muls(content: str) -> int:
    mul_pairs = [
        (int(mm.group()) for mm in
         re.finditer(r"\d+", m.group()))
        for m in re.finditer(r"mul\(\d+,\d+\)", content)
    ]
    return sum(i * j for i, j in mul_pairs)


def day_3_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle24(3).get_text_input()
    return day_3_sum_of_muls(text)


def day_3_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle24(3).get_text_input()
    do_s = [0] + [m.end() for m in re.finditer(r"do\(\)", text)]
    dont_s = [m.start() for m in re.finditer(r"don't\(\)", text)] + [-1]
    on_slices = []
    for d in do_s:
        e = min(filter(lambda x: x > d, dont_s)) if d < max(dont_s) else -1
        if not on_slices or d > on_slices[-1].stop:
            on_slices.append(slice(d, e))
    return sum(
        day_3_sum_of_muls(text[os]) for os in on_slices
    )


def day_2_load_reports(text: str) -> [[int]]:
    return [
        [int(n) for n in report.split(" ")]
        for report in text.strip("\n").split("\n")
    ]


def day_2_is_safe(report: [int]) -> bool:
    if (sorted(report) == report or
            sorted(report, reverse=True) == report):
        for i, n in enumerate(report[1:]):
            if not 1 <= abs(n - report[i]) <= 3:
                return False
        return True
    return False


def day_2_can_be_made_safe(report: [int]) -> bool:
    if day_2_is_safe(report):
        return True
    for mask in range(len(report)):
        if day_2_is_safe(report[:mask] + report[mask + 1:]):
            return True
    return False


def day_2_count_safe_reports(text: str, part_two: bool = False) -> int:
    if not text:
        text = Puzzle24(2).get_text_input()
    checker_func = day_2_can_be_made_safe if part_two else day_2_is_safe
    return sum(
        checker_func(rept)
        for rept in day_2_load_reports(text)
    )


def day_2_part_one(text: str = "") -> int:
    return day_2_count_safe_reports(text)


def day_2_part_two(text: str = "") -> int:
    return day_2_count_safe_reports(text, True)


def day_1_extract_lists(text: str = "") -> ([int],):
    if not text:
        text = Puzzle24(1).get_text_input()
    terms = r"\d+ ", r"\d+\n"
    return (
        [int(m.group()[:-1]) for m in re.finditer(t, text)]
        for t in terms
    )


def day_1_part_one(text: str = "") -> int:
    s1, s2 = (sorted(nn) for nn in day_1_extract_lists(text))
    return sum(abs(v1 - v2) for v1, v2 in zip(s1, s2))


def day_1_part_two(text: str = "") -> int:
    nums1, nums2 = day_1_extract_lists(text)
    left_nums = set(nums1)
    return sum(
        n * nums1.count(n) * nums2.count(n)
        for n in left_nums
    )
