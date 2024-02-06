import itertools
import library as lib
from main import Puzzle
import re
import math
from itertools import cycle
from functools import reduce
from copy import deepcopy


class Puzzle23(Puzzle):
    def get_text_input(self) -> str:
        with open(f'inputs\\2023\\input{self.day}.txt', 'r') as input_file:
            return input_file.read()


day_18_distance_moves = {
    "R": lambda p, d: lib.Point(p.x, p.y + d),
    "L": lambda p, d: lib.Point(p.x, p.y - d),
    "U": lambda p, d: lib.Point(p.x - d, p.y),
    "D": lambda p, d: lib.Point(p.x + d, p.y),
}


def day_18_part_two() -> int:
    corners = day_18_find_corners(
        Puzzle23(18).get_text_input().strip("\n"), for_part_two=True)
    return day_18_find_total_area(corners)


def day_18_find_corners(text: str, for_part_two: bool = True) -> [lib.Point]:
    corners = []
    point = lib.Point(0, 0)
    for line in text.split("\n"):
        p1_info, _, hex_string = line.partition(" (")
        if for_part_two:
            direction, distance = day_18_hex_to_dig_instruction(hex_string.strip(")"))
        else:
            direction, _, distance = p1_info.partition(" ")
            distance = int(distance)
        point = day_18_distance_moves[direction](point, distance)
        assert point not in corners
        corners.append(point)
    return corners


def day_18_find_total_area(corners: set) -> int:
    x_range, y_range = (tuple(func(dim(pt) for pt in corners)
                              for func in (min, max))
                        for dim in (lambda p: p.x, lambda p: p.y))
    area = 0
    rows_with_corners = {point.x for point in corners}
    previous_row = 0
    active_ranges = []
    print("\nRows with corners:")
    for rwc in sorted(rows_with_corners):
        area += sum(ar[1] - ar[0] + 1 for ar in active_ranges) * (rwc - previous_row)
        print(f"ROW: {rwc}")
        if rwc == 5:
            print("dbg")
        y_bounds = sorted(cnr.y for cnr in filter(lambda c: c.x == rwc, corners))
        for edge_start, edge_end in ((yb, y_bounds[(i * 2) + 1])
                                     for i, yb in enumerate(y_bounds[::2])):
            """Disproved assumption: at least one corner of an edge would match up 
                with an active range.  But what if it is entirely WITHIN an active
                range, like the n-shape test?
                Relevant range filter doesn't even pick this up.  What are the 
                    scenarios?  Wholly within, or one end poking out?  Possible?"""
            relevant_range = [*filter(
                lambda ar: any(c in ar for c in (edge_start, edge_end)) or
                           all(ar[0] < c < ar[1] for c in (edge_start, edge_end)),
                active_ranges)]
            if relevant_range:
                if relevant_range[0] == (edge_start, edge_end):
                    active_ranges.remove(relevant_range[0])
                    area += edge_end - edge_start + 1
                    continue
                rr_start, rr_end = relevant_range[0]
                new_ranges = []
                extend = edge_start < rr_start or edge_end > rr_end
                print(f"\t{extend=}")
                if extend:
                    new_ranges.append((edge_start, rr_end)
                                      if edge_start < rr_start else (rr_start, edge_end))
                else:
                    if all(relevant_range[0][0] < c < relevant_range[0][1]
                           for c in (edge_start, edge_end)):
                        """wholly within: remove the relevant range 
                            and add TWO new ranges"""
                        new_ranges.append((relevant_range[0][0], edge_start))
                        new_ranges.append((edge_end, relevant_range[0][1]))
                        area -= 1
                    else:
                        new_ranges.append((edge_end, rr_end) if rr_start < edge_end < rr_end
                                          else (rr_start, edge_start))
                    area += edge_end - edge_start
                active_ranges.remove(relevant_range[0])
                for nr in new_ranges:
                    active_ranges.append(nr)
            else:
                active_ranges.append((edge_start, edge_end))
        active_ranges = sorted(active_ranges, key=lambda a: a[0])
        if any(ar[0] == active_ranges[i][1] for i, ar in enumerate(active_ranges[1:])):
            print("We have coalescence")
            new_ar = []
            for ind, acr in enumerate(active_ranges):
                if ind > 0 and acr[0] == active_ranges[ind - 1][1]:
                    new_ar[-1] = (new_ar[-1][0], acr[1])
                else:
                    new_ar.append(acr)
            active_ranges = new_ar
        print(f"\t{active_ranges=}")
        assert not (len(y_bounds) % 2)
        previous_row = rwc
    return area


def day_18_hex_to_dig_instruction(raw_hex: str) -> (str, int):
    direction = "RDLU"[int(raw_hex[-1])]
    return direction, int(raw_hex[1:-1], 16)


def day_18_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle23(18).get_text_input().strip("\n")
    return len(day_18_get_dug_out_points(text))


def day_18_get_dug_out_points(text: str) -> set:
    border, inner = (set() for _ in range(2))
    x, y = 0, 0
    all_directions = "URDL"
    for instruction in text.split("\n"):
        direction, _, steps = instruction[:instruction.index(" (")].partition(" ")
        for _ in range(int(steps)):
            x, y = point_moves_2023[direction](lib.Point(x, y))
            border.add(lib.Point(x, y))
            if lib.Point(x, y) in inner:
                inner.remove(lib.Point(x, y))
            inner_point = point_moves_2023[
                all_directions[(all_directions.index(direction) + 1) % 4]](lib.Point(x, y))
            if inner_point not in border:
                inner.add(inner_point)
    if len(border) > 100:
        print("border traversed")
    assert not border.intersection(inner)
    return day_18_flood_fill_inside(border, inner)


def day_18_flood_fill_inside(border: set, inner: set) -> set:
    min_space, max_space = (func(func(pt) for pt in border) for func in (min, max))
    print(f"{min_space=}, {max_space=}")
    for x in range(min_space, max_space + 1):
        for y in range(min_space, max_space + 1):
            point = lib.Point(x, y)
            if all(point not in group for group in (border, inner)):
                for drn in "LU":
                    if point_moves_2023[drn](point) in inner:
                        inner.add(point)
    return border | inner


day_17_city = []


def day_17_part_one(text: str = "") -> int:
    day_17_load_city(text)
    grid_size = len(day_17_city)
    best = 0
    diagonal_x, diagonal_y = 0, 0
    for _ in range(grid_size - 1):
        diagonal_y += 1
        best += day_17_city[diagonal_x][diagonal_y]
        diagonal_x += 1
        best += day_17_city[diagonal_x][diagonal_y]
    # print(f"{best=}")
    assert best == 133
    states = [(0, 0, "R", 0), (0, 0, "D", 0)]
    while states:
        states = day_17_next_turn(states)
        print(f"{len(states)} states -> ", end="")
        # states = [*filter(
        #     lambda st: day_17_best_case_heat_loss(st, grid_size) < best,
        #     states)]
        # current_best = min(day_17_best_case_heat_loss(stt, grid_size) for stt in states)
        # if current_best < best:
        #     print(f"\nNew best: {current_best}")
        #     best = current_best
        print(f"{len(states)}")
        if len(states) > 1_000_000:
            break
    scores = {state[-1] for state in states}
    print(sorted(scores))
    return best


def day_17_best_case_heat_loss(state: tuple, grid_size: int) -> int:
    x, y, _, heat_loss_so_far = state[-4:]
    min_steps = lib.manhattan_distance(lib.Point(x, y),
                                       lib.Point(*(grid_size - 1 for _ in range(2))))
    return heat_loss_so_far + min_steps


def day_17_next_turn(all_states: [()]) -> [()]:
    directions = "URDL"
    grid_size = len(day_17_city)
    next_states = []
    for state in all_states:
        x, y, facing, heat_loss_so_far = state[-4:]
        turns_so_far = state[:-4]
        x, y = point_moves_2023[facing](lib.Point(x, y))
        heat_loss_so_far += day_17_city[x][y]
        for turn in range(3):
            new_facing = facing
            if turns_so_far[-3:] != (turn for _ in range(3)):
                if turn:
                    if turn == 1:
                        new_facing = directions[(directions.index(facing) - 1) % 4]
                    if turn == 2:
                        new_facing = directions[(directions.index(facing) + 1) % 4]
                next_location = point_moves_2023[new_facing](lib.Point(x, y))
                if all(0 <= co_ord < grid_size for co_ord in next_location):
                    next_states.append((*turns_so_far, turn, x, y, new_facing, heat_loss_so_far))
    return sorted(next_states, key=lambda ns: ns[-1])


def day_17_load_city(text: str = ""):
    global day_17_city
    if not text:
        text = Puzzle23(17).get_text_input().strip("\n")
    day_17_city = [[int(num) for num in line] for line in text.split("\n")]


day_16_grid = []
day_16_energised = set()
day_16_deja_vu = set()


def day_16_part_two(text: str = "") -> int:
    global day_16_grid
    day_16_grid = day_16_load_grid(text)
    most_energised = 0
    grid_size = len(day_16_grid)
    directions = "DLUR"
    for di, drn in enumerate(directions):
        for line_no in range(grid_size):
            start_edge = grid_size if drn in "LU" else -1
            origin = lib.Point(line_no, start_edge) if di % 2 \
                else lib.Point(start_edge, line_no)
            day_16_trace_ray_until_exit(origin, drn)
            if len(day_16_energised) > most_energised:
                most_energised = len(day_16_energised)
            day_16_reset_globals()
    return most_energised


def day_16_part_one(text: str = "") -> int:
    global day_16_grid
    day_16_grid = day_16_load_grid(text)
    day_16_trace_ray_until_exit(lib.Point(0, -1), "R")
    for row in range(len(day_16_grid)):
        print("")
        for col in range(len(day_16_grid[0])):
            symbol = "#" if lib.Point(row, col) in day_16_energised else "."
            print(symbol, end="")
    return len(day_16_energised)


def day_16_load_grid(text: str = "") -> [[]]:
    day_16_reset_globals()
    if not text:
        text = Puzzle23(16).get_text_input().strip("\n")
    return text.split("\n")


def day_16_reset_globals():
    global day_16_energised, day_16_deja_vu
    day_16_energised, day_16_deja_vu = (set() for _ in range(2))


def day_16_trace_ray_until_exit(location: lib.Point, facing: str):
    edge = len(day_16_grid)
    while True:
        day_16_deja_vu.add((location, facing))
        if all(co > -1 for co in location):
            day_16_energised.add(location)
        location = point_moves_2023[facing](location)
        if all(0 <= co_ord < edge for co_ord in (location.x, location.y)):
            mirror = day_16_grid[location.x][location.y]
            if mirror in r"\/":
                reflection_pairs = ["DR", "UL"] if mirror == "\\" else ["DL", "UR"]
                facing = [d for rp in reflection_pairs
                          for d in rp
                          if facing in rp and d != facing][0]
            elif facing + mirror in ("U-", "D-", "L|", "R|"):
                all_directions = "DLUR"     # gave DIFFERENT result to URDL before separating the deja-vu check from in-grid check
                new_facings = "".join(d for i, d in enumerate(all_directions)
                                      if i % 2 != all_directions.index(facing) % 2)
                for nf in new_facings:
                    day_16_trace_ray_until_exit(location, nf)
                break
        else:
            break
        if (location, facing) in day_16_deja_vu:
            break


def day_15_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle23(15).get_text_input()
    boxes = {n: [] for n in range(256)}
    for step in text.split(","):
        step = step.strip("\n")
        label = re.search(r"\w+", step).group()
        box_id = day_15_string_hash(label)
        operation = step[len(label)]
        print(f"{label=}, {operation=} -> Box {box_id}")
        current_contents = boxes[box_id]
        print(f"\t{current_contents} --> ", end="")
        existing_instances = [lens for lens in current_contents if lens[0] == label]
        assert len(existing_instances) < 2
        if operation == "-":
            if existing_instances:
                current_contents.remove(existing_instances[0])
        elif operation == "=":
            if existing_instances:
                existing_lens = existing_instances[0]
                index = current_contents.index(existing_lens)
                existing_lens = existing_lens[0], int(step[-1])
                current_contents[index] = existing_lens
            else:
                current_contents.append((label, int(step[-1])))
        boxes[box_id] = current_contents
        print(boxes[box_id])
    return sum(
        (box_no + 1) * (i + 1) * c[1]
        for box_no, contents in boxes.items()
        for i, c in enumerate(contents)
    )


def day_15_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle23(15).get_text_input()
    return sum(day_15_string_hash(step.strip("\n")) for step in text.split(","))


def day_15_string_hash(whole_string: str) -> int:
    current_value = 0
    for ch in whole_string:
        current_value = day_15_single_character_hash(ch, current_value)
    return current_value
    # return reduce(lambda ch, next_ch: day_15_single_character_hash(), whole_string)


def day_15_single_character_hash(char: str, current_value: int = 0) -> int:
    current_value += ord(char)
    current_value *= 17
    return current_value % 256


def day_14_part_two(text: str = "") -> int:
    board = day_14_load_board(text)
    load_history = {}
    repeat_period = 0
    for spin in range(1, 201):
        for rotation in "NWSE":
            board = day_14_tilt_board(board, rotation)
        load_history[spin] = day_14_north_support_beams_loading(board)
    def next_seven_values(nn: int): return [load_history[nn + k] for k in range(7)]
    for test_iteration in range(50, 151):
        nsv = next_seven_values(test_iteration)
        for forward_steps in range(1, 100):
            if next_seven_values(test_iteration + forward_steps) == nsv:
                repeat_period = forward_steps
                if next_seven_values(test_iteration + (2 * forward_steps)) == nsv:
                    break
    print(f"{repeat_period=}")
    long_interval = 1000000000
    periods_in_long_interval = long_interval // repeat_period
    periods_in_150 = 150 // repeat_period
    matching_key = long_interval - \
                   ((periods_in_long_interval - periods_in_150) * repeat_period)
    return load_history[matching_key]


def day_14_part_one(text: str = "") -> int:
    return day_14_load_after_tilt_board_north(day_14_load_board(text))


def day_14_north_support_beams_loading(board: {}) -> int:
    return sum(loc[0] for loc, rock in board.items() if rock == "O")


def day_14_tilt_board(board: dict, direction: str) -> dict:
    tilted = {loc: stop for loc, stop in board.items() if stop == "#"}
    vertical = direction in "NS"
    increasing = direction in "NE"
    scan, roll = (1, 0) if vertical else (0, 1)
    # print(f"{direction=}")
    for line in range(max(k[scan] for k in board.keys()) + 1):
        stoppers = [0] + \
                   [kk[roll] for kk, v in board.items()
                    if kk[scan] == line and v == "#"] + \
                   [max(point[roll] for point in board.keys()) + 1]
        stoppers = stoppers[::1 if increasing else -1]
        # print(f"{line=}, {stoppers=}")
        for i, next_stopper in enumerate(stoppers[1:]):
            previous_stopper = stoppers[i]
            rock_filter = {
                True: lambda point:
                previous_stopper < point[roll] < next_stopper,
                False: lambda point:
                previous_stopper > point[roll] > next_stopper,
            }[increasing]
            rolling_rocks = len([pt for pt, ch in board.items()
                                 if pt[vertical] == line
                                 and rock_filter(pt)
                                 and ch == "O"])
            for rr in range(rolling_rocks):
                fixed, variable = line, next_stopper + \
                                  ((-1 if increasing else 1) * (rr + 1))
                ordered = (variable, fixed) if vertical else (fixed, variable)
                tilted[ordered] = "O"
            # print(f"\t{next_stopper=}, {rolling_rocks=}")
    return tilted


def day_14_load_after_tilt_board_north(board: dict) -> int:
    northern_boundary = max(point[0] for point in board.keys()) + 1
    aggregate_load = 0
    for column in range(max(k[1] for k in board.keys()) + 1):
        stoppers = [kk[0] for kk, v in board.items()
                    if kk[1] == column and v == "#"]
        print(f"{column=}, {stoppers=}")
        for i, stopper_row_id in enumerate(stoppers + [northern_boundary]):
            previous_stopper = 0 if i == 0 else stoppers[i - 1]
            rolling_rocks = len([pt for pt, ch in board.items()
                                 if pt[1] == column
                                 and previous_stopper < pt[0] < stopper_row_id
                                 and ch == "O"])
            print(f"\t{stopper_row_id=}, {rolling_rocks=}")
            aggregate_load += day_14_rock_load(rolling_rocks, stopper_row_id)
    return aggregate_load


def day_14_rock_load(n_rocks: int, stopper_row: int) -> int:
    southern_rock = stopper_row - n_rocks
    return day_14_triangular(n_rocks) + (n_rocks * (southern_rock - 1))


def day_14_triangular(n: int) -> int:
    if n < 2:
        return n
    return n + day_14_triangular(n - 1)


def day_14_load_board(text: str = "") -> dict:
    """resultant space has rows starting at 1 and increasing in a northerly direction"""
    if not text:
        text = Puzzle23(14).get_text_input().strip("\n")
    return {
        (r + 1, c + 1): char
        for r, row in enumerate(text.split("\n")[::-1])
        for c, char in enumerate(row)
        if char in "O#"
    }


def day_13_part_two(text: str = "") -> int:
    grids = day_13_load_grids(text)
    return sum(day_13_mirror_score_for_part_two(g) for g in grids)


def day_13_mirror_score_for_part_two(grid: [str]) -> int:
    old_v, old_h = day_13_find_vertical_mirror(grid), day_13_find_horizontal_mirror(grid)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            test_grid = deepcopy(grid)
            test_grid[i] = test_grid[i][:j] +\
                           {".": "#", "#": "."}[test_grid[i][j]] + \
                           test_grid[i][j + 1:]
            new_vertical = day_13_find_vertical_mirror(test_grid, old_v)
            if new_vertical and new_vertical != old_v:
                return new_vertical
            new_horizontal = day_13_find_horizontal_mirror(test_grid, old_h)
            if new_horizontal and new_horizontal != old_h:
                return 100 * new_horizontal


def day_13_part_one(text: str = "") -> int:
    grids = day_13_load_grids(text)
    return sum(
        day_13_find_vertical_mirror(grid) + (100 * day_13_find_horizontal_mirror(grid))
        for grid in grids
    )


def day_13_find_horizontal_mirror(grid: [str], old_value: int = -1) -> int:
    return day_13_find_mirror(grid, len(grid), old_value)


def day_13_find_vertical_mirror(grid: [str], old_value: int = -1) -> int:
    grid = [[row[c] for _, row in enumerate(grid)]
            for c in range(len(grid[0]))]
    return day_13_find_mirror(grid, len(grid), old_value)


def day_13_find_mirror(grid: [str], perpendicular_axis_length: int,
                       old_value: int = -1) -> int:
    for test_line in range(1, perpendicular_axis_length):
        sides = grid[:test_line][::-1], grid[test_line:]
        if all(sides) and all(s1 == s2 for s1, s2 in zip(*sides)) and test_line != old_value:
            return test_line
    return 0


def day_13_load_grids(text: str = "") -> [str]:
    if not text:
        text = Puzzle23(13).get_text_input().strip("\n")
    return [sub_grid.split("\n") for sub_grid in text.split("\n\n")]


def day_12_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle23(12).get_text_input()
    return sum(day_12_count_all_possible_arrangements(record)
               for record in text.split("\n"))


def day_12_count_all_possible_arrangements(damaged_record: str) -> int:
    if " " not in damaged_record:
        return 0
    return day_12_dictionary_based_solution(*day_12_get_record_details(damaged_record))


def day_12_get_record_details(damaged_record: str) -> (str, [int]):
    sequence, _, groups_data = damaged_record.partition(" ")
    group_lengths = [int(gl) for gl in groups_data.split(",")]
    return sequence, group_lengths


def day_12_dictionary_based_solution(springs: str, groups: [int]) -> int:
    print(f"{springs}\t{groups=}")
    if (not springs) or (not groups):
        if "#" in springs:
            raise ValueError
        return 1
    if all(c == "?" for c in springs):
        return day_12_count_possible_arrangements(springs, groups)
        return day_12_combinations_per_unknown_section(len(springs), groups)
    group_limits = day_12_group_limits_by_social_distancing(springs, groups, return_max_end=True)
    print(f"{group_limits=}")
    if all(gl[1] == gl[0] + gr - 1 for gl, gr in zip(group_limits, groups)):
        print(f"{springs} returns with social distancing shortcut")
        return 1
    """Dictionary is {n: {possible group indices}} for each character position
        in springs.  Initially empty list for each one.
        - work through groups in descending order of size.  If complete group
            is found, or a group that cannot be any other, update lists accordingly
        - remove any position key that definitely cannot be a hash
        - traverse springs from both ends to narrow it down further"""
    def break_problem(gi_to_remove: int, springs_cutoffs: (int,)) -> int:
        cut_at, continue_from = springs_cutoffs
        if cut_at < 0:
            print(f"{cut_at=} should never be negative. {springs=}, {groups=}")
            raise ValueError
        return day_12_dictionary_based_solution(
            springs[:cut_at], groups[:gi_to_remove]) *\
               day_12_dictionary_based_solution(
                   springs[continue_from:],
                   groups[gi_to_remove + 1:]
               )

    def is_enough_space(current_index: int, group_length: int) -> bool:
        r_start, r_end = day_12_get_contiguous_range(dictionary, current_index)
        return r_end - r_start >= group_length

    dictionary = {index: set() for index in range(len(springs))
                  if springs[index] != "."}
    for k in range(len(springs)):
        if springs[k] == "#":
            dictionary[k] = {gi for gi in range(len(groups))
                              if k in range(group_limits[gi][0], group_limits[gi][1] + 1)
                              and is_enough_space(k, groups[gi])}
    """Find starts and/or ends to determine whole runs:"""
    for k in range(len(springs)):
        # known starts:
        if k - 1 not in dictionary and k in dictionary and len(dictionary[k]) == 1:
            print("USES KNOWN START")
            remove_group_index =[*dictionary[k]][0]
            return break_problem(remove_group_index,
                                 (max(k - 1, 0), k + groups[remove_group_index] + 1))
        # known ends:
        if k + 1 not in dictionary and k in dictionary and len(dictionary[k]) == 1:
            print("USES KNOWN END")
            remove_group_index = [*dictionary[k]][0]
            return break_problem(remove_group_index,
                                 (max(k - groups[remove_group_index], 0), k + 2))
    """Use first and last hashes if there is no doubt which group they belong to"""
    print(f"Before: {dictionary=}")
    for gi in range(len(groups)):
        definite_finds = {kk: vv for kk, vv in dictionary.items()
                          if vv == {gi}}
        if definite_finds and max(definite_finds) - min(definite_finds) + 1 == groups[gi]:
            # finds first run (possibly broken) of unambiguous hashes whose length
            #   is equal to the relevant group length
            print(f"DEFINITE FIND: {min(definite_finds)}-{max(definite_finds)}")
            return break_problem(gi, (max(0, min(definite_finds) - 1),
                                      max(definite_finds) + 2))
    return day_12_count_possible_arrangements(springs, groups)
        # inferred blocks - GETS IT WRONG:
        # if k in dictionary:
        #     if len(dictionary[k]) == 1:
        #         found_group_index = [*dictionary[k]][0]
        #         possibles = [*filter(
        #             lambda dk: found_group_index in dictionary[dk], dictionary)]
        #         if max(possibles) - min(possibles) + 1 == groups[found_group_index]:
        #             start = min(possibles)
        #             print(f"INFERRED BLOCK: {min(possibles)}-{max(possibles)}")
        #             return break_problem(
        #                 found_group_index,
        #                 (max(start - 1, 0), start + groups[found_group_index] + 1)
        #             )
        #     else:
        #         for ind in dictionary[k]:
        #             if sum([ind in dictionary[kk] for kk in dictionary]) == 1:
        #                 dictionary[k] = {ind}
        #                 break
        #                 # TODO: This is interfering with the dict.  Could it bite me?
    for k in range(len(springs)):
        if springs[k] == "?":
            dictionary[k] = {gi for gi in range(len(groups))
                              if k in range(group_limits[gi][0], group_limits[gi][1] + 1)
                              and is_enough_space(k, groups[gi])}
    print(f"{dictionary=}")
    if springs == "?.??#?#?#.????":
        print("debug")
    s_first = e_first = 0
    while e_first - s_first < groups[0]:
        s_first, e_first = day_12_get_contiguous_range(
            dictionary, min(k for k in dictionary if k > e_first))
    if e_first - s_first == len(springs):
        print(f"It's all one big unknown!")
        return day_12_combinations_per_unknown_section(len(springs), groups)
    if e_first == max(dictionary) + 1:
        return day_12_dictionary_based_solution(springs[s_first:e_first], groups)
    # break off the first set and see how many groups I can get in?
    max_gi = max(max(s) if s else 0
                 for s in [v for k, v in dictionary.items() if k <= e_first])
    for gri in range(len(groups)):
        if gri > max_gi or e_first - s_first < day_12_min_space(groups[:gri + 1]) - 1:
            sub_problems = [
                (springs[s_first:e_first], groups[:gri]),
                (springs[min(k for k in dictionary.keys() if k > e_first):], groups[gri:]),
            ]
            break
    print(f"{sub_problems=}")
    return math.prod(day_12_dictionary_based_solution(*sp) for sp in sub_problems)
    for k in range(len(springs)):
        if springs[k] == "?":
            for i_grp in range(len(groups)):
                if k in range(group_limits[i_grp][0], group_limits[i_grp][1] + 1):
                    existing_possible_groups = [*dictionary[k]] if k in dictionary else []
                    if is_enough_space(
                            k, day_12_min_space(existing_possible_groups + [i_grp])):
                        dictionary[k].add(i_grp)
                    else:
                        break

    for long_group in sorted(groups, reverse=True):
        continue
        if long_group > 1 and groups.count(long_group) == 1:
            next_longest = max(0, *(gr if gr < long_group else 0 for gr in groups))
            matches = [m for m in re.finditer("#+", springs)
                       if long_group >=
                       m.end() - m.start() >
                       next_longest]
            if len(matches) == 1:
                if matches[0].end() - matches[0].start() == long_group:
                    # whole group is found, so remove it and adjoining spaces
                    # todo: also remove group from groups list at this point?
                    #       When a group is eliminated, what to do?  return 1?
                    #       Return the product of the function called on each of
                    #       the remaining group(s) - realising the middle could
                    #       have been cut out?
                    for j in range(matches[0].start() - 1, matches[0].end() + 1):
                        if j in dictionary:
                            del dictionary[j]
                else:
                    for k in range(matches[0].start(), matches[0].end()):
                        dictionary[k] = [long_group]
                    # todo: see if there are just the right number of adjacent
                    #       spots in the dictionary to be able to confidently fill?

    dictionary = day_12_traverse_springs(springs, groups, dictionary)
    print(f"{dictionary=}")
    return 1


def day_12_get_contiguous_range(dict_keys: [], current_key: int) -> (int,):
    """returns the key values that would match up with the appropriate range() args"""
    if current_key not in dict_keys:
        raise ValueError
    start_range = end_range = current_key
    while end_range in dict_keys:
        end_range += 1
    while start_range in dict_keys:
        start_range -= 1
    return start_range + 1, end_range


def day_12_count_possible_arrangements(springs: str, groups: [int]) -> int:
    if springs.count("#") == sum(groups):
        return 1

    def is_valid(start: int, group_id: int) -> bool:
        return "." not in springs[start:start + groups[group_id]] and\
               "#" not in "".join(springs[i] if i in range(len(springs)) else ""
                                  for i in (start - 1, start + groups[group_id]))

    start_params = day_12_refine_start_parameters(
        day_12_group_limits_by_social_distancing(springs, groups),
        springs, groups)
    if len(groups) == 1:
        min_start, max_start = start_params[0]
        return sum([
            is_valid(s, 0)
            for s in range(min_start, max_start + 1)
        ])
    else:
        possible_arrangements = 0
        min_start_1, max_start_1 = start_params[0]
        for st in range(min_start_1, max_start_1 + 1):
            if is_valid(st, 0):
                possible_arrangements += day_12_count_possible_arrangements(
                    springs[st + groups[0] + 1:], groups[1:]
                )
        return possible_arrangements


# def day_12_locate_unmistakable_groups(section: str, groups: [int], dict_so_far: {}) -> {}:
def day_12_traverse_springs(springs: str, groups: [int], dict_so_far: {}) -> {}:
    hg = [g for g in groups]
    grp_dict = {i: grp for i, grp in enumerate(groups)}
    gi = [*range(len(groups))]
    limits = day_12_group_limits_by_social_distancing(springs, groups, return_max_end=True)
    # print(limits)
    # todo: use social distancing limits (start_params).  If a hash is found
    #       that only fits one set of limits, can do something with that info
    # todo: would it actually be more useful to have min. start and max. end limits?
    previous_char = "."
    # for i, ch in enumerate(springs):
    #     if ch == "#":
    #         if previous_char == ".":
    #             for di in range(i, i + hg[0] + 1):
    #                 dict_so_far[di] = [gi[0]]
    #                 if i + hg[0] + 1 in dict_so_far:
    #                     del dict_so_far[i + hg[0] + 1]
    #     previous_char = ch
    return dict_so_far


def day_12_refine_start_parameters(params: [(int,)],
                                   section: str, groups: [int]) -> [(int,)]:
    """Use known hash positions to further tie down possible starting points"""
    for i in day_12_known_hash_string_starts(section):
        possible_params = [*filter(
            lambda p: p[0] <= i < p[1] + groups[params.index(p)], params)]
        if len(possible_params) == 1:
            new_param = possible_params[0]
            i_groups = params.index(new_param)
            new_param = (i, i)
            params[i_groups] = new_param
    for i in day_12_known_hash_string_ends(section):
        possible_params = [*filter(
            lambda p: p[0] <= i - groups[params.index(p)] + 1 < p[1] + groups[params.index(p)], params)]
        if len(possible_params) == 1:
            new_param = possible_params[0]
            i_groups = params.index(new_param)
            new_param = (i - groups[i_groups] + 1, i - groups[i_groups] + 1)
            params[i_groups] = new_param
    return params


def day_12_group_limits_by_social_distancing(section: str, hash_groups: [int],
                                             return_max_end: bool = False) -> [(int,)]:
    group_limits = []
    for gi, hashes_length in enumerate(hash_groups):
        min_start = day_12_min_space(hash_groups[:gi])
        max_start = len(section) - day_12_min_space(hash_groups[gi + 1:]) - hashes_length
        if return_max_end:
            group_limits.append((min_start,
                                 len(section) - day_12_min_space(hash_groups[gi + 1:]) - 1))
        else:
            group_limits.append((min_start, max_start))
    return group_limits


def day_12_multi_step_solution(section: str, hash_groups: [int]) -> int:
    """First, use social distancing to get first and last possible starts"""
    print(f"Running multi-step solution for {section}, {hash_groups}")
    start_params = day_12_group_limits_by_social_distancing(section, hash_groups)
    if all(sp[0] == sp[1] for sp in start_params):
        print(f"{section} returns with social distancing shortcut")
        return 1
    """Can known starts and ends of hash-strings help?"""
    print(f"{start_params=} --> ", end="")
    known_starts = day_12_known_hash_string_starts(section)
    known_ends = day_12_known_hash_string_ends(section)
    for ks in known_starts:
        found_params = [*filter(lambda sp: sp[0] <= ks <= sp[1], start_params)]
        if len(found_params) == 1:
            sp_ind = start_params.index(found_params[0])
            start_params = day_12_fix_start(start_params, sp_ind, ks, hash_groups)
        elif all(ch == "." for ch in section[:ks]):
            start_params = day_12_fix_start(start_params, 0, ks, hash_groups)
    for ke in known_ends:
        found_params = [*filter(lambda sp_hg:
                                sp_hg[0][0] <= ke - sp_hg[1] + 1 <= sp_hg[0][1],
                                zip(start_params, hash_groups))]
        if len(found_params) == 1:
            sp_ind = start_params.index(found_params[0][0])
            group_length = hash_groups[sp_ind]
            start_params = day_12_fix_start(start_params, sp_ind, ke - group_length + 1,
                                            hash_groups)
        elif all(ch == "." for ch in section[ke + 1:]):
            start_params = day_12_fix_start(start_params, len(hash_groups) - 1,
                                            ke - hash_groups[-1] + 1, hash_groups)
    if all(sp[0] == sp[1] for sp in start_params):
        print(f"{start_params}")
        print("\t. . . completes using known starts/ends")
        return 1
    """Is whole of first or last group present at a location where it couldn't be any other?"""
    fmt, lmt = ("#{" + f"{gs}" + "}" for gs in (hash_groups[0], hash_groups[-1]))
    first_match = re.search(fmt, section)
    if first_match and first_match.start() < hash_groups[0] + 1:
        start_params = day_12_fix_start(start_params, 0, first_match.start(), hash_groups)
    last_match = re.search(lmt, section[::-1])
    if last_match:
        lm_start = len(section) - last_match.start() - hash_groups[-1]
        if lm_start > len(section) - 2 - hash_groups[-1] or \
                all(ch == "." for ch in section[lm_start + hash_groups[-1]:]):
            start_params = day_12_fix_start(start_params, len(start_params) - 1,
                                            lm_start, hash_groups)
    print(f"{start_params} --> ", end="")
    if all(sp[0] == sp[1] for sp in start_params):
        print("\t. . . completes by finding complete group(s) and/or fixing starts")
        return 1
    """Try to further narrow down positions using known values:"""
    for spi, mm in enumerate(start_params):
        s, e = mm
        if s == e:
            continue
        new_s, new_e = s, e
        if s > 0 and section[s - 1] == "#":
            new_s = s + 1
        for i in range(s, e + hash_groups[spi]):
            if section[i] == ".":
                new_s = i + 1
            if section[i] == "#":
                new_s = i
                break
            if section[i] == "?" and \
                    all(ch in "#?" for ch in section[i:i + hash_groups[spi]]):
                break
        if e + hash_groups[spi] < len(section) and section[e + hash_groups[spi]] == "#":
            new_e = e - 1
        for i in range(e + hash_groups[spi] - 1, s - 1, -1):
            if section[i] == ".":
                new_e = i - hash_groups[spi]
            if section[i] == "#":
                new_e = i - hash_groups[spi] + 1
                break
            if section[i] == "?" and \
                    all(ch in "#?" for ch in section[i - hash_groups[spi] + 1:i + 1]):
                break
        if new_s > s or new_e < e:
            start_params[spi] = (new_s, new_e)
            if new_s == new_e:
                day_12_fix_start(start_params, spi, new_s, hash_groups)
    print(f"{start_params}")
    if all(sp[0] == sp[1] for sp in start_params):
        print("\t. . . completes by additional narrowing down based on known characters")
        return 1
    """ Search for contiguous areas of unknowns or hashes.
        Each group must be wholly contained within one of these.
        But there can be one of these that doesn't contain any group.
        Look for groups that can only be within one particular run"""
    return_value = 1
    remaining_unknowns = [ind for ind, spread in enumerate(start_params)
                          if spread[1] > spread[0]]
    assert remaining_unknowns
    q_or_hash_runs = [(m.start(), m.end() - 1) for m in
                      re.finditer(r"[?#]+", section)]
    for qh_first, qh_last in q_or_hash_runs:
        contained_unknowns = [*filter(
            lambda ru: start_params[ru][0] + hash_groups[ru] <= qh_last + 1 and
                       start_params[ru][1] >= qh_first,
            remaining_unknowns)]
        if len(contained_unknowns) == 1:
            print(f"\tsolving {hash_groups} using a single contained unknown")
            u_index = contained_unknowns[0]
            space = min(start_params[u_index][1] + hash_groups[u_index], qh_last + 1) - \
                    max(qh_first, start_params[u_index][0])
            return_value *= day_12_combinations_per_unknown_section(space, [hash_groups[u_index]])
            remaining_unknowns.remove(u_index)
        elif contained_unknowns:
            first_u = contained_unknowns[0]
            start_space = max(qh_first, start_params[first_u][0])
            end_space = min(qh_last + 1,
                            start_params[first_u][1] + hash_groups[first_u] + 1)
            hg_to_process, valid_hg = [], []
            for i, unkn in enumerate(contained_unknowns):
                hg_to_process.append(hash_groups[unkn])
                if start_space + \
                        sum(hg_to_process) + len(hg_to_process) - 1 <= qh_last:
                    end_space = min(qh_last + 1,
                                    start_params[unkn][1] + hash_groups[unkn] + 1)
                    remaining_unknowns.remove(unkn)
                    valid_hg.append(hash_groups[unkn])
                else:
                    break
            if valid_hg:
                print(f"\tAttempting solution with multiple contained unknowns")
                return_value *= day_12_combinations_per_unknown_section(
                    end_space - start_space, valid_hg)
    """What to do if a #/? run is long enough to accommodate multiple groups?
        Try from both ends?  Some hashes may fix positions . . . """
    """Finally, use the combinations function to get a result for unknown sections.
            Some kind of rule involving overlap will determine if multiple groups
            share the same unknown space"""
    return return_value


def day_12_fix_start(min_max_starts: [(int,)],
                     mms_index: int, new_value: int,
                     groups: [int]) -> [(int,)]:
    """Update start parameters list when we know a start value.  Make sure
        neighbours in the list are updated to ensure social distancing"""
    min_max_starts[mms_index] = (new_value, new_value)
    if mms_index > 0:
        new_prev_max = new_value - 2 - groups[mms_index - 1] + 1
        if min_max_starts[mms_index - 1][1] > new_prev_max:
            ns, _ = min_max_starts[mms_index - 1]
            if ns == new_prev_max:
                day_12_fix_start(min_max_starts, mms_index - 1, new_prev_max, groups)
            else:
                min_max_starts[mms_index - 1] = (ns, new_prev_max)
    if mms_index < len(min_max_starts) - 1:
        new_next_min = new_value + groups[mms_index] + 1
        if min_max_starts[mms_index + 1][0] < new_next_min:
            _, ns = min_max_starts[mms_index + 1]
            if ns == new_next_min:
                day_12_fix_start(min_max_starts, mms_index + 1, new_next_min, groups)
            else:
                min_max_starts[mms_index + 1] = (new_next_min, ns)
    return min_max_starts


def day_12_known_hash_string_ends(section: str) -> [int]:
    starts_in_reversed = day_12_known_hash_string_starts(section[::-1])
    return [len(section) - 1 - sr for sr in starts_in_reversed[::-1]]


def day_12_known_hash_string_starts(section: str) -> [int]:
    return [ks.start() + ks.group().startswith(".")
            for ks in re.finditer(r"\.#|^#", section)]


def day_12_min_space(hash_groups: [int], at_end: bool = True) -> int:
    return sum(hash_groups) + len(hash_groups) + (not at_end)


def day_12_combinations_per_unknown_section(section_length: int, groups: [int]) -> int:
    """needs to be just the section_length in which group is free to move.
        External social distancing needs to be handled outside this function"""
    print(f"\tday_12_combinations_per_unknown_section: {section_length=}, {groups=}")
    if len(groups) == 1:
        group_length = groups[0]
        return section_length - group_length + 1
    else:
        return sum(day_12_combinations_per_unknown_section(
            section_length - (groups[0] + n + 1), groups[1:]
        )
                   for n in range(section_length - sum(groups[1:]) - len(groups[1:])))


def day_11_part_two(text: str = "", expansion_factor: int = 1_000_000) -> int:
    observed_universe = day_11_find_galaxies(text)
    expanded = day_11_expand(observed_universe, factor=expansion_factor - 1)
    return sum(lib.manhattan_distance(p1, p2)
               for p1, p2 in day_11_all_galaxy_pairs(expanded))


def day_11_part_one(text: str = "") -> int:
    observed_universe = day_11_find_galaxies(text)
    expanded = day_11_expand(observed_universe)
    return sum(lib.manhattan_distance(p1, p2)
               for p1, p2 in day_11_all_galaxy_pairs(expanded))


def day_11_expand(universe: set, factor: int = 1) -> set:
    row_max, col_max = (max(u[i] for u in universe) + 1 for i in range(2))
    blank_rows = [r for r in range(row_max) if not any(p[0] == r for p in universe)]
    blank_cols = [c for c in range(col_max) if not any(p[1] == c for p in universe)]
    # print(f"{blank_rows=}, {blank_cols=}")
    return {(
        r + (len([br for br in blank_rows if br < r] * factor)),
        c + (len([bc for bc in blank_cols if bc < c] * factor))
    )
            for r, c in universe}


def day_11_all_galaxy_pairs(galaxies: set) -> itertools.combinations:
    return itertools.combinations(galaxies, 2)


def day_11_find_galaxies(text: str = "") -> set:
    if not text:
        text = Puzzle23(11).get_text_input()
    return {(r, c)
            for r, row in enumerate(text.split("\n"))
            for c, char in enumerate(row)
            if char == "#"}


point_moves_2023 = {
    "R": lambda p: lib.Point(p.x, p.y + 1),
    "L": lambda p: lib.Point(p.x, p.y - 1),
    "U": lambda p: lib.Point(p.x - 1, p.y),
    "D": lambda p: lib.Point(p.x + 1, p.y),
}


day_10_map = {}


def day_10_part_two(text: str = "") -> int:
    day_10_load_map(text)
    s_point = lib.Point(*[pt for pt, symbol in day_10_map.items() if symbol == "S"][0])
    pipe, _ = day_10_trace_pipe_from(s_point)
    known_inside_points = day_10_all_inside_edge_points(pipe)
    return day_10_count_enclosed_by_flood_fill(known_inside_points, pipe)


def day_10_count_enclosed_by_flood_fill(inside: {(int,)}, pipe: [(int,)]) -> int:
    def joins(point: lib.Point, clump: set) -> bool:
        """This works because points hidden within the already-known enclosed
            border points already have such points to the left of and above them"""
        for drn in "LU":
            if point_moves_2023[drn](point) in clump:
                return True
        return False
    grid_max = max(max(k) for k in day_10_map.keys())
    for x in range(grid_max + 1):
        for y in range(grid_max + 1):
            point = lib.Point(x, y)
            if (point not in day_10_map) or (point not in pipe):
                if joins(point, inside):
                    print(f"{x}, {y} joins inside set")
                    inside.add(point)
    return len(inside)


def day_10_all_inside_edge_points(pipe: [(int,)]) -> set:
    moves_to_inner_points = {
        ("F", "D"): ["RD"],
        ("F", "R"): ["L", "U"],
        ("-", "L"): ["D"],
        ("-", "R"): ["U"],
        ("7", "L"): ["LD"],
        ("7", "D"): ["R", "U"],
        ("|", "D"): ["R"],
        ("|", "U"): ["L"],
        ("J", "L"): ["D", "R"],
        ("J", "U"): ["LU"],
        ("L", "U"): ["L", "D"],
        ("L", "R"): ["UR"],
    }
    """Don't know whether pipe is being traversed clockwise or anti-clockwise, 
        so do one traverse in each direction, and the true inner set will be 
        the smaller one"""
    set_1, set_2 = (set() for _ in range(2))
    for recipient_set, pipe_journey in zip((set_1, set_2), (pipe, pipe[::-1])):
        for i, this_pipe_point in enumerate(pipe_journey):
            next_pipe_point = pipe_journey[(i + 1) if i < len(pipe_journey) - 1 else 0]
            # print(day_10_map[next_pipe_point], end=" ")
            next_section = day_10_map[next_pipe_point]
            move_going_out_of = [*filter(
                lambda d: day_10_get_next_location_round(next_pipe_point, this_pipe_point) ==
                          point_moves_2023[d](next_pipe_point), "UDLR")][0]
            # print(move_going_out_of, end="")
            if next_section == "S":
                break
            # print(f"\n{next_pipe_point=}, {move_going_out_of=} --> ", end="")
            for path in moves_to_inner_points[(next_section, move_going_out_of)]:
                point = lib.Point(*next_pipe_point)
                for mv in path:
                    point = point_moves_2023[mv](point)
                # print(f"{point}, ", end="")
                if point not in day_10_map or point not in pipe_journey:
                    recipient_set.add(point)
    print(len(set_1), len(set_2))
    return min((set_1, set_2), key=lambda s: len(s))


def day_10_part_one(text: str = "") -> int:
    day_10_load_map(text)
    found_loops, found_incomplete_pipes = [], []
    s_point = [lib.Point(*pt) for pt, symbol in day_10_map.items() if symbol == "S"][0]
    if s_point in day_10_map and all(s_point not in s for s in
                                   (found_loops, found_incomplete_pipes)):
        pipe, is_loop = day_10_trace_pipe_from(s_point)
        # print(f"Returned pipe: {pipe}")
        if is_loop:
            found_loops.append(pipe)
        elif len(pipe) > 1:
            found_incomplete_pipes.append(pipe)
    print(f"Longest loop has {max(len(loop) for loop in found_loops)} sections")
    return max(len(loop) for loop in found_loops) // 2


def day_10_trace_pipe_from(starting_location: (int,)) -> ([], bool):
    complete_loop = False
    visited = []
    # valid_neighbours = [day_10_get_next_location_round(
    #     starting_location, point_moves_2023[dirn](starting_location))
    #     for dirn in "DULR"]
    valid_neighbours = [*filter(lambda loc:
                                loc in day_10_map and
                                day_10_is_connectable(starting_location, loc),
        [point_moves_2023[dirn](starting_location) for dirn in "DLUR"])]
    if len(valid_neighbours) > 1:
        print(f"{starting_location} is possibly in a loop")
        visited.append(starting_location)
        location = starting_location
        previous_location = valid_neighbours[0]
        next_location = None
        while True:
            next_location = day_10_get_next_location_round(location, previous_location)
            if day_10_is_connectable(location, next_location):
                if next_location in visited:
                    complete_loop = True
                    break
                visited.append(next_location)
            else:
                break
            previous_location, location = location, next_location
    return visited, complete_loop


def day_10_get_next_location_round(location: (int,), previous: (int,)) -> (int,):
    """find the next valid location to move to, based on where you are and what shape
        pipe section you're in"""
    current_section = day_10_map[location]
    moves = {"F": "DR", "-": "LR", "7": "DL",
             "|": "DU", "J": "UL", "L": "UR",
             "S": "URDL"}
    valid_neighbours = [point_moves_2023[drn](lib.Point(*location))
                        for drn in moves[current_section]]
    if current_section == "S":
        valid_neighbours = [*filter(lambda vn: day_10_is_connectable(location, vn),
                                    valid_neighbours)]
    return [*filter(lambda vn: vn != lib.Point(*previous) and vn in day_10_map,
                    valid_neighbours)][0]


def day_10_is_connectable(loc_1: (int,), loc_2: (int,)) -> bool:
    """NB. does not check for adjacency in loop"""
    invalid_moves = {pipe: pipe for pipe in "F7JL"}     # right-angle pipes cannot connect to other instance of themselves
    invalid_moves["|"] = "-"    # vertical cannot connect to horizontal and vice-versa
    invalid_moves["-"] = "|"
    if all(location in day_10_map for location in (loc_1, loc_2)):
        first_pipe = day_10_map[loc_1]
        if first_pipe == "S":
            valid_next_sections = {"L": "-FL", "R": "-7J", "U": "|F7", "D": "|JL"}
            direction = [k for k in point_moves_2023.keys()
                         if point_moves_2023[k](loc_1) == loc_2][0]
            return day_10_map[loc_2] in valid_next_sections[direction]
        return day_10_map[loc_2] not in invalid_moves[first_pipe]
    return False


def day_10_get_starting_point() -> lib.Point:
    return lib.Point(*[pt for pt, symbol in day_10_map.items() if symbol == "S"][0])


def day_10_load_map(text: str = ""):
    global day_10_map
    if not text:
        text = Puzzle23(10).get_text_input().strip("\n")
    day_10_map = {(r, c): symbol
                  for r, row in enumerate(text.split("\n"))
                  for c, symbol in enumerate(row)
                  if symbol != "."}


def day_9_part_one(text: str = "") -> int:
    return sum(day_9_predict_next_number(day_9_fully_decompose(seq))
               for seq in day_9_load_sequences(text))


def day_9_part_two(text: str = "") -> int:
    return sum(day_9_extrapolate_backwards(day_9_fully_decompose(seq))
               for seq in day_9_load_sequences(text))


def day_9_extrapolate_backwards(pyramid: [int]) -> int:
    return reduce(lambda x, y: y - x, [p[0] for p in pyramid[::-1]])


def day_9_predict_next_number(pyramid: [int]) -> int:
    return reduce(lambda x, y: x + y, [p[-1] for p in pyramid[::-1]])


def day_9_fully_decompose(sequence: [int]) -> [[int]]:
    next_seq = sequence
    diff_steps = [sequence]
    while any(n != 0 for n in next_seq):
        next_seq = day_9_diff_sequence(next_seq)
        diff_steps.append(next_seq)
    return diff_steps


def day_9_diff_sequence(sequence: [int]) -> [int]:
    return [sequence[i + 1] - n for i, n in enumerate(sequence[:-1])]


def day_9_load_sequences(text: str = "") -> [[int]]:
    if not text:
        text = Puzzle23(9).get_text_input().strip("\n")
    return [[int(n) for n in line.split(" ")] for line in text.split("\n")]


def day_8_part_two(text: str = "") -> int:
    # if text:
    #     return day_8_part_two_traverse(*day_8_load_instructions(text))  # naive method for examples
    turns, route_map = day_8_load_instructions(text)

    def get_repeat_period(location: str) -> int:
        looping_directions = cycle(turns)
        visited_zs = set()
        last_good, repeat = 0, 0
        print(location, end="->")
        for step in range(1, 50_000):
            location = route_map[location][{"L": 0, "R": 1}[next(looping_directions)]]
            if location[-1] == "Z":
                print(location, end="->")
                print("\n", step)
                return step
                if location in visited_zs:
                    repeat = step - last_good   # this was no good if only two steps in 50,000
                    last_good = step
                visited_zs.add(location)
        return repeat

    periods = [get_repeat_period(loc) for loc in
               [*filter(lambda point: point[-1] == "A", route_map.keys())]]
    print(periods)
    return math.lcm(*periods)


def day_8_part_two_traverse(directions: str, node_map: {}) -> int:
    directions = cycle(directions)
    locations, steps = [*filter(lambda point: point[-1] == "A", node_map.keys())], 0
    while not all(loc[-1] == "Z" for loc in locations):
        next_turn = next(directions)
        locations = [node_map[pt][{"L": 0, "R": 1}[next_turn]] for pt in locations]
        steps += 1
    return steps


def day_8_part_one() -> int:
    return day_8_traverse(*day_8_load_instructions())


def day_8_traverse(directions: str, node_map: {}) -> int:
    directions = cycle(directions)
    location, steps = "AAA", 0
    while location != "ZZZ":
        location = node_map[location][{"L": 0, "R": 1}[next(directions)]]
        steps += 1
    return steps


def day_8_load_instructions(text: str = "") -> (str, {}):
    if not text:
        text = Puzzle23(8).get_text_input().strip("\n")
    directions, _, forks = text.partition("\n\n")
    node_map = {}
    for line in forks.split("\n"):
        point, _, options = line.partition(" = ")
        node_map[point] = tuple(m.group() for m in re.finditer(r"(\w){3}", options))
    return directions, node_map


day_7_card_values = "".join(f'{i}' for i in range(2, 10)) + "TJQKA"


def day_7_part_two(text: str = "") -> int:
    return day_7_core_method(day_7_score_hand_with_jokers, text)


def day_7_part_one(text: str = "") -> int:
    return day_7_core_method(day_7_score_hand, text)


def day_7_core_method(scoring_function: callable, text: str = "") -> int:
    if not text:
        text = Puzzle23(7).get_text_input()
    ranked_hands = sorted([line for line in text.split("\n") if line],
                          key=lambda data: scoring_function(data[:5]))
    return sum(int(row[6:]) * (rank + 1) for rank, row in enumerate(ranked_hands))


def day_7_score_hand_with_jokers(hand: str) -> int:
    joker_positions = [i for i, cd in enumerate(hand[::-1]) if cd == "J"]
    best_card = ""
    if hand == "JJJJJ":
        best_card = "A"
    else:
        best_card = max(set(hh for hh in hand if hh != "J"), key=lambda card: hand.count(card))
    hand = hand.replace("J", best_card)
    jokered_score = day_7_score_hand(hand)
    js_text = "".join("0" if j // 2 in joker_positions else dd
                      for j, dd in enumerate(str(jokered_score)[::-1]))[::-1]
    joker_penalty = sum(10 ** ((2 * jp) + 1) for jp in joker_positions)
    return int(js_text) - joker_penalty


def day_7_score_hand(hand: str) -> int:
    """-> 11-digit number: 1st = hand-type score (0-6),
        each remaining pair is value of individual card"""
    type_score, score = 0, 0
    match len(set(hand)):
        case 1:         # all match
            type_score = 6
        case 2:
            type_score = 5 if max(hand.count(cd) for cd in set(hand)) == 4 else 4
            # four of a kind or full house
        case 3:
            type_score = 3 if max(hand.count(cd) for cd in set(hand)) == 3 else 2
            # three of a kind or two pair
        case 4:
            type_score = 1
    score = (type_score * 10 ** 10) + \
            sum(day_7_card_values.index(card) * 10 ** (2 * i)
                for i, card in enumerate(hand[::-1]))
    return score


def day_6_part_two(text: str = "") -> int:
    raw_times, raw_distances = day_6_load_race_data(text)
    time, distance_to_beat = (int("".join(str(nn) for nn in group))
                              for group in (raw_times, raw_distances))
    roots = tuple(int(r) for r in day_6_solve_quadratic(-1, time, -distance_to_beat))
    return roots[1] - roots[0]


def day_6_solve_quadratic(a: int, b: int, c: int):
    # (based on ChatGPT)
    discriminant = math.sqrt(b ** 2 - 4 * a * c)
    return ((-b + (sign_factor * discriminant)) / (2 * a)
            for sign_factor in (1, -1))


def day_6_part_one(text: str = "") -> int:
    times, distances = day_6_load_race_data(text)
    ways_to_win_per_race = []
    for time, distance_to_beat in zip(times, distances):
        ways_to_win_per_race.append(
            len([*filter(
                lambda hold_time: hold_time * (time - hold_time) > distance_to_beat,
                range(1, time + 1))])
        )
    return math.prod(ways_to_win_per_race)


def day_6_load_race_data(text: str = ""):
    if not text:
        text = Puzzle23(6).get_text_input()
    detail_rows = text.split("\n")
    return tuple(
        tuple(int(nn.group()) for nn in re.finditer(r"\d+", detail_rows[r]))
        for r in range(2)
    )


def day_5_part_two(text: str = "") -> int:
    # range_details, mappings = day_5_load_data(text)
    # seed_ranges = [(r, range_details[(i * 2) + 1])
    #                for i, r in enumerate(range_details[::2])]
    # for location in range(2_000_000_000):
    #     test_value = location
    #     for step in [*mappings.keys()][1:][::-1]:
    #         test_value = day_5_reverse_convert(mappings[step], test_value)
    #     if day_5_is_seed(test_value, seed_ranges):
    #         return location
    # return 0
    range_details, mappings = day_5_load_data(text)
    seed_ranges = [(r, range_details[(i * 2) + 1])
                   for i, r in enumerate(range_details[::2])]
    total_movements = {sr: 0 for sr in seed_ranges}
    for step_key in mappings:
        total_movements = day_5_cumulative_offsets_next_step(
            total_movements, mappings[step_key])
    return min(k[0] + v for k, v in total_movements.items())


def day_5_cumulative_offsets_next_step(offsets: {}, step_mappings: [(int,)]) -> {}:
    new_co = {}
    for seed_range, offset in offsets.items():
        first_original_seed, range_length = seed_range
        last_original_seed = first_original_seed + range_length - 1
        unused_ranges = [(first_original_seed, last_original_seed)]
        for mapping in step_mappings:
            kill_list, add_list = [], []
            sm_dest, sm_start, sm_len = mapping
            for first_seed, last_seed in unused_ranges:
                if sm_start <= first_seed + offset and \
                        sm_start + sm_len - 1 >= last_seed + offset:
                    """Wholly within mapping range"""
                    kill_list.append((first_seed, last_seed))
                    new_co[(first_seed, last_seed - first_seed + 1)] = \
                        offset + sm_dest - sm_start
                else:
                    if first_seed == 79 and sm_start == 77:
                        print("dbg")
                    overlap_starts_in_ur = \
                        first_seed + offset < sm_start <= last_seed + offset
                    overlap_ends_in_ur = \
                        first_seed + offset < sm_start + sm_len - 1 < last_seed + offset
                    if overlap_starts_in_ur and overlap_ends_in_ur:
                        kill_list.append((first_seed, last_seed))
                        new_co[(sm_start - offset, sm_len)] = \
                            offset + sm_dest - sm_start
                        add_list.append((first_seed, sm_start - offset - 1))
                        add_list.append((sm_start + sm_len - offset, last_seed))
                    elif overlap_ends_in_ur:
                        kill_list.append((first_seed, last_seed))
                        new_co[(first_seed,
                                sm_start + sm_len - first_seed - offset)] = \
                            offset + sm_dest - sm_start
                        add_list.append((sm_start + sm_len - offset, last_seed))
                    elif overlap_starts_in_ur:
                        kill_list.append((first_seed, last_seed))
                        new_co[(sm_start - offset,
                                last_seed + offset + 1 - sm_start)] = \
                            offset + sm_dest - sm_start
                        add_list.append((first_seed, sm_start - offset - 1))
            for kill in kill_list:
                unused_ranges.remove(kill)
            for add in add_list:
                unused_ranges.append(add)
        for ur in unused_ranges:
            first, last = ur
            new_co[(first, last - first + 1)] = offset
    return new_co


def day_5_cumulative_offsets_next_step_old(offsets: {}, step_mappings: [(int,)]) -> {}:
    new_co = {}
    print("===== NEW STEP =====")
    for seed_range, current_offset in sorted(offsets.items()):
        print(f"Before: {seed_range=}, {current_offset=}")
        start_seed, range_length = seed_range

        divisions = []
        for mapping_range in step_mappings:
            if mapping_range == (45, 77, 23) and seed_range[0] == 79:
                print("debug")
            if mapping_range[1] <= start_seed + current_offset and \
                    sum(mapping_range[1:]) >= start_seed + current_offset + range_length:
                new_co[seed_range] = current_offset + mapping_range[0] - mapping_range[1]
                print(f"\tWhole seed range {seed_range} with {current_offset=} falls within mapping range {mapping_range}")
                break
            else:
                os_map_start, os_map_end = mapping_range[1], sum(mapping_range[1:]) - 1
                print(f"\t{mapping_range=} --> {os_map_start} {os_map_end}")
                for end_point, is_start in zip((os_map_start, os_map_end), (True, False)):
                    if start_seed + current_offset <= end_point < start_seed + current_offset + range_length:
                        divisions.append((end_point, is_start,
                                          mapping_range[0] - mapping_range[1] + (not is_start)))
        if not divisions and seed_range not in new_co:
            new_co[seed_range] = offsets[seed_range]
        elif divisions:
            divisions = sorted(divisions, key=lambda d: d[0])
            change_offset = (not divisions[0][1]) or \
                            (divisions[0][1] and divisions[0][0] == start_seed + current_offset)
            print(f"--> {divisions=}")
            marker = start_seed
            remaining_length = range_length
            for ii, details in enumerate(divisions):
                div_start, st, offset_change = details
                if div_start > start_seed + current_offset:
                    length = (div_start - start_seed - current_offset) if ii == 0 \
                        else (div_start - divisions[ii - 1][0])
                    if length:
                        new_co[(marker, length)] = (current_offset + offset_change) \
                            if change_offset else current_offset
                    marker += length
                    remaining_length -= length
                change_offset = not change_offset
            if remaining_length:
                new_co[(marker, remaining_length)] = (current_offset + offset_change) \
                            if change_offset else current_offset
        print(f"After: {new_co}")
    if not new_co:
        new_co = offsets
    assert sum(seeds for _, seeds in new_co.keys()) == \
           sum(seeds for _, seeds in offsets.keys())
    print(f"After: {new_co=}")
    return new_co


def day_5_ranges_overlap(r1: (int,), r2: (int,)) -> bool:
    """tuples are the numbers you would give to range() function"""
    r1_s, r1_e = r1
    r2_s, r2_e = r2
    r1_e -= 1
    r2_e -= 1
    if r1_e < r2_s or r2_e < r1_s:
        return False
    return True


def day_5_is_seed(seed_id: int, seed_ranges: [(int,)]) -> bool:
    return any(seed_min <= seed_id < seed_min + length
               for seed_min, length in seed_ranges)


def day_5_reverse_convert(list_of_vars: [(int,)], forward_source: int) -> int:
    for vvv in list_of_vars:
        fwd_src_start, _, length = vvv
        if forward_source in range(fwd_src_start, fwd_src_start + length):
            return day_5_reverse_range_function(vvv)(forward_source)
    return forward_source


def day_5_reverse_range_function(variables: (int,)) -> callable:
    reverse_dest_start, forward_src_start, _ = variables
    return lambda fwd_src: fwd_src + forward_src_start - reverse_dest_start


def day_5_part_one(text: str = "") -> int:
    seeds, mappings = day_5_load_data(text)
    eventual_locations = []
    for sd in seeds:
        category_number = sd
        for next_step in mappings:
            category_number = day_5_convert(mappings[next_step], category_number)
        eventual_locations.append(category_number)
    return min(eventual_locations)


def day_5_convert(list_of_vars: [(int,)], source: int) -> int:
    for vvv in list_of_vars:
        _, src_start, length = vvv
        if source in range(src_start, src_start + length):
            return day_5_range_function(vvv)(source)
    return source


def day_5_range_function(variables: (int,)) -> callable:
    dest_start, src_start, _ = variables
    return lambda source: source + dest_start - src_start


def day_5_load_data(text: str = "") -> ([int,], {}):
    if not text:
        text = Puzzle23(5).get_text_input().strip("\n")
    seeds = [int(s) for s in
             re.match(r"seeds: ([0-9]|\s)+", text).group().strip().split(" ")
             if s.isnumeric()]
    mappings = {"seed": []}
    map_finder = re.finditer(r"\n.+ map:\n", text)
    for map_match in map_finder:
        destination = re.search(r"-\w+\s", map_match.group()).group().strip("- ")
        raw_numbers = re.search(r"([0-9]|\s)+", text[map_match.end():]).group().strip()
        mappings[destination] = [tuple(int(nn) for nn in rn.split(" "))
                                 for rn in raw_numbers.split("\n")]
    assert len(mappings) == 8
    return seeds, mappings


def day_4_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle23(4).get_text_input().strip("\n")
    card_pile = (t for t in text.split("\n"))
    last_card = int(re.findall(r"\d+:", text)[-1][:-1])
    instance_counter = {c: 1 for c in range(1, last_card + 1)}
    for card_no in range(1, last_card + 1):
        winners = day_4_count_winners(next(card_pile))
        for c_id_increment in range(1, winners + 1):
            instance_counter[card_no + c_id_increment] += instance_counter[card_no]
    return sum(instance_counter.values())


def day_4_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle23(4).get_text_input().strip("\n")
    return sum(day_4_score_card(card) for card in text.split("\n"))


def day_4_score_card(card_text: str) -> int:
    winners = day_4_count_winners(card_text)
    return 2 ** (winners - 1) if winners else 0


def day_4_count_winners(card_text: str) -> int:
    useful_part = card_text[card_text.index(":") + 1:]
    w, _, g = useful_part.partition("|")
    winners, got = (set(int(n) for n in re.findall(r"\d+", nos)) for nos in (w, g))
    return len(winners.intersection(got))


def day_3_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle23(3).get_text_input().strip("\n")
    grid = text.split("\n")
    asterisks = day_3_find_all_asterisks(grid)
    gear_ratios = []
    for ast in asterisks:
        part_nos = []
        for row_id in (ast.x - 1, ast.x + 1):
            if 0 <= row_id < len(grid):
                re_matches_in_adjacent_row = [*re.finditer(r"\d+", grid[row_id])]
                for m in re_matches_in_adjacent_row:
                    if m.start() in range(ast.y - 1, ast.y + 2) or \
                            m.end() in range(ast.y, ast.y + 2) or \
                            {*range(ast.y - 1, ast.y + 2)}.issubset({*range(m.start(), m.end())}):
                        part_nos.append(m.group())
        matches_in_this_row = [*re.finditer(r"\d+", grid[ast.x])]
        for mtr in matches_in_this_row:
            if mtr.end() == ast.y or mtr.start() == ast.y + 1:
                part_nos.append(mtr.group())
        if len(part_nos) == 2:
            gear_ratios.append(int(part_nos[0]) * int(part_nos[1]))
    return sum(gear_ratios)


def day_3_find_all_asterisks(grid: [lib.Point]) -> [lib.Point]:
    return [lib.Point(x, start_y)
            for x, row in enumerate(grid)
            for start_y in
            (m.start() for m in re.finditer(r"\*", row))]


def day_3_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle23(3).get_text_input().strip("\n")
    grid = text.split("\n")
    valid_part_nos = []
    for row_id, row in enumerate(grid):
        print(row)
        for match in re.finditer(r"\d+", row):
            print(f"{match.group()} starts at {match.start()}, ends at {match.end()}", end=", ")
            neighbours = set()
            start, end = (func() for func in (match.start, match.end))
            end -= 1
            neighbours.update(lib.get_all_neighbours_in_grid(lib.Point(row_id, start), grid))
            neighbours.update(lib.get_all_neighbours_in_grid(lib.Point(row_id, end), grid))
            neighbours.update([lib.Point(rr, cc)
                               for rr in (row_id - 1, row_id + 1)
                               for cc in range(start + 2, end - 1)
                               if 0 <= rr < len(grid) and 0 <= cc < len(grid[0])])
            neighbours = {*filter(lambda n: n.x != row_id
                                            or n.y not in range(start, end + 1), neighbours)}
            print(neighbours)
            if any(not grid[neigh.x][neigh.y].isnumeric() and
                   grid[neigh.x][neigh.y] != "." for neigh in neighbours):
                valid_part_nos.append(int(match.group()))
    return sum(valid_part_nos)


def day_2_part_two(text: str = "") -> int:
    all_games = day_2_load_game_text(text)
    limits = {"red": 12, "green": 13, "blue": 14}   # lazily copied - don't need limits

    def get_power(game_contents: [{}]) -> int:
        power = 1
        for colour in limits:
            power *= max(pick[colour] if colour in pick else 0 for pick in game_contents)
        return power
    return sum(get_power(g) for g in all_games.values())


def day_2_part_one(text: str = "") -> int:
    all_games = day_2_load_game_text(text)
    limits = {"red": 12, "green": 13, "blue": 14}

    def game_is_possible(game_id: int) -> bool:
        picks = all_games[game_id]
        for p in picks:
            for colour, quantity in p.items():
                if quantity > limits[colour]:
                    return False
        return True

    return sum(
        k for k in all_games.keys()
        if game_is_possible(k)
    )


def day_2_load_game_text(text: str = "") -> {}:
    if not text:
        text = Puzzle23(2).get_text_input().strip("\n")
    games = {}
    lines = text.split("\n")
    for line in lines:
        game_no = int(re.search(r"\d+:", line).group()[:-1])
        _, _, details = line.partition(": ")
        picks = []
        for selection in details.split("; "):
            this_pick = {}
            for cubes in selection.split(", "):
                number, _, colour = cubes.partition(" ")
                this_pick[colour] = int(number)
            picks.append(this_pick)
        games[game_no] = picks
    return games


def day_1_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle23(1).get_text_input()
    numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }
    solution = 0
    base_search = r"\d|" + "|".join(numbers.keys())
    for line in text.split("\n"):
        if not line:
            break
        first, last = (day_1_find_number(line, fl, re_search)
                       for fl, re_search
                       in zip(range(1, -1, -1),
                              (base_search, f"(?=({base_search}))")))
        print(line, end="")
        first, last = (numbers[x] if not x.isnumeric() else int(x) for x in (first, last))
        print(" --> ", first, last)
        solution += (10 * first) + last
    return solution


def day_1_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle23(1).get_text_input()
    return sum([
        sum(10 ** f * day_1_find_number(line, f) for f in range(1, -1, -1))
        for line in text.split("\n")])


def day_1_find_number(whole_line: str, first: bool | int = True,
                      search_term: str = "") -> int | str:
    if whole_line:
        if search_term:
            if first:
                return re.search(search_term, whole_line).group()
            else:
                return re.findall(search_term, whole_line)[-1]
        else:
            return int(re.search(r"\d", whole_line[::1 if first else -1]).group())
    return 0

