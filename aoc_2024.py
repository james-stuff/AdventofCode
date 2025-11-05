import collections
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


day_19_known_possible = set()
day_19_impossible = set()
day_19_known_ways = {}


def day_19_part_two(text: str = "") -> int:
    available, strings = day_19_load(text)
    ways = []
    for s in strings:
        c = day_19_count_ways(s, available)
        ways.append(c)
        print(f"\t{c} ways of getting '{s}'\n")
    print(f"{len(ways)=}")
    print(f"Number of zeroes: {len([*filter(lambda w: w == 0, ways)])}")
    return sum(ways)


def day_19_count_ways(wanted: str, towels: set) -> int:
    print(f"\tWorking on {wanted}")
    if len(wanted) == 0:
        return 1
    if wanted in day_19_known_ways:
        return day_19_known_ways[wanted]
    usable = [
        *filter(
            lambda towel: wanted.startswith(towel),
            towels
        )
    ]
    if usable:
        no_of_ways = sum(
            day_19_count_ways(wanted[len(ut):], towels)
            for ut in usable
        )
        day_19_known_ways[wanted] = no_of_ways
        return no_of_ways
    day_19_impossible.add(wanted)
    print(f"Confirmed impossible: {wanted}")
    day_19_known_ways[wanted] = 0
    return 0


def day_19_load(text: str = "") -> (set,):
    global day_19_known_possible, day_19_impossible, day_19_known_ways
    day_19_known_possible = set()
    day_19_known_ways = {}
    if not text:
        text = Puzzle24(19).get_text_input().strip("\n")
    c, _, d = text.partition("\n\n")
    components = {*c.split(", ")}
    desired = d.split("\n")
    return components, desired


def day_19_part_one(text: str = "") -> int:
    global day_19_impossible
    day_19_impossible = set()
    available, strings = day_19_load(text)
    return len(
        [
            *filter(
                lambda v: v,
                (
                    day_19_allowable(s, available)
                    for s in strings
                )
            )
        ]
    )


def day_19_allowable(wanted: str, available_bits: set) -> bool:
    if wanted in day_19_impossible:
        return False
    print(f"\tWorking on {wanted}")
    allowable_bits = [
        *filter(
            lambda b: wanted.startswith(b),
            available_bits
        )
    ]
    if allowable_bits:
        if any(bit == wanted for bit in allowable_bits):
            print("Confirmed possible")
            return True
        if any(
            day_19_allowable(wanted[len(bb):], available_bits)
            for bb in allowable_bits
        ):
            return True
    day_19_impossible.add(wanted)
    print("Confirmed impossible")
    return False


def day_18_part_one(text: str = "") -> int:
    walkable = day_18_walkable_points(text)
    solution, _ = day_18_dijkstra_to_end(walkable)
    return solution


def day_18_dijkstra_to_end(walkable: set, give_path: bool = False) -> int:
    traceback_walkable = {*walkable}

    origin, destination = (f(walkable) for f in (min, max))
    distances = {w: 1_000_000_000 for w in walkable}
    distances[lib.Point(0, 0)] = 0
    while walkable:
        closest = min(walkable, key=lambda wp: distances[wp])
        for neighbour in day_18_neighbours(closest, walkable):
            ngb_distance = distances[closest] + 1
            if ngb_distance < distances[neighbour]:
                distances[neighbour] = ngb_distance
        walkable.remove(closest)

    if give_path and distances[destination] < 1_000_000_000:
        tip = destination
        path = []
        while tip != origin:
            previous = [
                *filter(lambda ng: distances[ng] == distances[tip] - 1,
                        day_18_neighbours(tip, traceback_walkable))
            ][0]
            path.append(previous)
            tip = previous
        # print(f"Found path: {path[::-1]}\n of length: {len(path)}")
        return distances[destination], path
    return distances[destination], []


def day_18_neighbours(point: lib.Point, available: set) -> [lib.Point]:
    return [
        mv(point)
        for mv in pm23.values()
        if mv(point) in available
    ]


def day_18_walkable_points(text: str = "") -> set:
    grid_size, n_first_to_fall = 7, 12
    if not text:
        text = Puzzle24(18).get_text_input()
        grid_size, n_first_to_fall = 71, 1024
    blocked = {
        lib.Point(*(int(n) for n in m.group().split(",")))
        for m in [*re.finditer(r"\d+,\d+", text)][:n_first_to_fall]
    }
    return {
        lib.Point(x, y)
        for x, y in itertools.product(range(grid_size), repeat=2)
    } - blocked


def day_18_part_two(text: str = "") -> str:
    """Get the Dijkstra method to record each step
        on the shortest path.
        Add further bytes until one of them lands on this path
        Recalculate Dijkstra
        Repeat until there is no way to get there"""
    n_fallen = 12 if text else 1024
    if not text:
        text = Puzzle24(18).get_text_input()
    walkable = day_18_walkable_points()
    _, optimal_path = day_18_dijkstra_to_end(
        {*walkable}, give_path=True
    )
    text = "\n".join(text.split("\n")[n_fallen:])
    iterator = re.finditer(r"\d+,\d+", text)
    while True:
        to_drop = lib.Point(
            *(int(n) for n in next(iterator).group().split(","))
        )
        walkable.remove(to_drop)
        if to_drop in optimal_path:
            distance, optimal_path = day_18_dijkstra_to_end(
                {*walkable}, True
            )
            if distance > 71 ** 2:
                return ",".join(f"{c}" for c in to_drop)
    return "0,0"


def day_17_part_one(text: str = "") -> str:
    return day_17_execute(*day_17_load(text))


def day_17_part_two(text: str = "") -> int:
    register, program = day_17_load(text)
    # register["A"] = 0
    a = 0
    for i in reversed(range(len(program))):
        print(f"{i=}, {a=}")
        a <<= 3
        increments = 0
        while day_17_execute(
                {"A": a, "B": 0, "C": 0}, program
        ) != ",".join(f"{p}" for p in program[i:]):
            a += 1
            increments += 1
        if increments > 7:
            print(f"\tRequired increment of {increments}")
    return a

    # def previous_octet(prog: [int], reg_a: int) -> int:
    #     print(f"{reg_a=}, {prog=}")
    #     if not prog:
    #         return reg_a
    #     output = prog[-1]
    #     reg_a <<= 3
    #     while int(day_17_execute(
    #             {"A": reg_a, "B": 0, "C": 0}, program
    #     )[0]) != output:
    #         reg_a += 1
    #     reg_a = previous_octet(prog[:-1], reg_a)
    #
    #     # for _ in range(8):
    #     #     reg_a += 1
    #     #     if int(day_17_execute(
    #     #             {"A": reg_a, "B": 0, "C": 0}, program
    #     #     )[0]) == output:
    #     #         reg_a = previous_octet(prog[:-1], reg_a)
    #     #         break
    #
    # return previous_octet(program, 0)


def day_17_predict_register_a(output: [int]) -> int:
    a_value = 0
    for j, m in enumerate(output[::-1]):
        a_value = a_value * 8
        for a_incr in range(8):
            last_3 = day_17_get_next_3_bits_of_a(a_value + a_incr, m)
            if last_3 is not None:
                a_value += last_3
                if a_incr:
                    # a_value = a_value + a_incr
                    print(f"{a_incr=}, {last_3=}, {a_value=}")
                break
    return a_value


def day_17_get_next_3_bits_of_a(reg_a: int, out: int) -> int:
    # if reg_a % 8:
    #     print("debug")
    for p in range(8):
        b = p ^ 5
        c = ((8 * (reg_a // 8)) + p) // (2 ** b)
        output = (b ^ 6 ^ c) % 8
        if output == out:
            print(f"\t{reg_a=}, {out=}, {b=}, {c=}, return value={p}")
            return p


def day_17_step_deconstruct(output_value: int) -> int:
    """which mod-8 value produces the desired output_value?"""
    """Last step: output = last three binary digits of register B
            We know register A is 1-7"""
    b = [n ^ 5 for n in range(1, 8)]
    reg_a_mod_8 = min(((2 ** bb) * output_value) ^ 6 ^ bb for bb in b)
    return reg_a_mod_8


def day_17_load(text: str = "") -> ({}, [int]):
    if not text:
        text = Puzzle24(17).get_text_input()
    reg = {
        m.group()[0]: int(m.group()[3:])
        for m in re.finditer(r"[ABC]: \d+", text)
    }
    program = [
        int(n) for n in
        re.search(r"(\d,)+\d", text).group().split(",")
    ]
    return reg, program


def day_17_execute(register: {}, program: [int]) -> str:
    p_len = len(program)
    pointer = 0
    output = []
    while pointer < p_len:
        p_increment = 2
        opcode, operand = program[pointer:pointer + 2]
        match opcode:
            case 0:     # adv
                operand = day_17_combo_op(register, operand)
                register["A"] = register["A"] // (2 ** operand)
            case 1:     # bxl
                register["B"] = register["B"] ^ operand
            case 2:     # bst
                register["B"] = day_17_combo_op(register, operand) % 8
            case 3:     # jnz
                if register["A"] != 0:
                    pointer = operand
                    p_increment = 0
            case 4:     # bxc
                register["B"] = register["B"] ^ register["C"]
            case 5:     # out
                output.append(f"{day_17_combo_op(register, operand) % 8}")
            case 6:     # bdv
                operand = day_17_combo_op(register, operand)
                register["B"] = register["A"] // (2 ** operand)
            case 7:     # cdv
                operand = day_17_combo_op(register, operand)
                register["C"] = register["A"] // (2 ** operand)
        pointer += p_increment
    return ",".join(output)


def day_17_combo_op(register: {}, raw_operand: int) -> int:
    if raw_operand > 3:
        assert raw_operand != 7
        return register[chr(65 + raw_operand - 4)]
    return raw_operand


day_16_start, day_16_end = (lib.Point(0, 0) for _ in range(2))
day_16_distances_table, day_16_walks_table, day_16_previous_table = \
    ({} for _ in range(3))
day_16_walks_taken = set()


def day_16_part_one(text: str = "") -> int:
    day_16_set_up(text)
    return min(
        day_16_distances_table[(day_16_end, fc)]
        if (day_16_end, fc) in day_16_distances_table else 1_000_000_000
        for fc in (True, False)
    )


def day_16_directional_points(walkable: set) -> set:
    d = set()
    for w in walkable:
        neighbours = day_16_walkable_neighbours(w, walkable)
        x_s, y_s = {n.y for n in neighbours}, {n.x for n in neighbours}
        if len(x_s) == 1:
            d.add((w, True))
        elif len(y_s) == 1:
            d.add((w, False))
        else:
            d.update({(w, True), (w, False)})
    return d


def day_16_pd_neighbours(drnl_pt: (lib.Point, bool), directional: set) -> set:
    reachable = set()
    my_loc, my_dir = drnl_pt
    if (my_loc, not my_dir) in directional:
        reachable.add((my_loc, not my_dir))
    reachable.update(
        {
            (mv(my_loc), my_dir)
            for mv in pm23.values()
            if (mv(my_loc), my_dir) in directional
        }
    )
    return reachable


def day_16_part_two(text: str = "") -> int:
    global day_16_distances_table, day_16_walks_table
    # day_16_set_up(text)
    walks_taken = set()

    # assume part one has run already with the same text
    end_keys = [(day_16_end, bb) for bb in (False, True)]
    min_dist = 1_000_000_000
    end = end_keys[0]
    if end in day_16_distances_table:
        min_dist = day_16_distances_table[end]
    if (end_keys[1] in day_16_distances_table
            and day_16_distances_table[end_keys[1]] < min_dist):
        end = end_keys[1]
    print(f"{end=}")

    queue = [[end]]
    while queue:
        route = queue.pop()
        previous_walks_data = dict(filter(
            lambda i: i[0] == route[-1]
            , day_16_previous_table.items()
        ))
        walks = [s for v in previous_walks_data.values() for s in v]
        for pw in walks:
            queue.append(route + [pw[0]])
            walks_taken.add(pw)

    moving_walks = [w[1][0] for w in walks_taken if w[1][0] != w[0][0]]
    duplication = len(moving_walks) - len(set(moving_walks))
    # +1 is to count the starting point
    return sum(wt[2] % 100 for wt in walks_taken) - duplication + 1


def day_16_set_up(text: str = ""):
    global day_16_distances_table, day_16_walks_table, day_16_previous_table
    maze = day_16_load_maze(text)
    dp = day_16_directional_points(maze)
    junctions = {
        *filter(
            lambda pt: len(day_16_walkable_neighbours(pt[0], maze)) > 2,
            dp)
    }
    junctions.update([dpp for dpp in dp if dpp[0] in (day_16_start, day_16_end)])
    junctions.add((day_16_start, False))
    day_16_walks_table = day_16_build_walks_table(junctions, maze)
    day_16_distances_table, day_16_previous_table = day_16_dijkstra_distances(
        junctions, day_16_walks_table
    )


def day_16_dijkstra_distances(directional_juncs: set, walks: {()}) -> ({},):
    if len(day_16_distances_table):
        return day_16_distances_table
    j_dist = {jf: 1_000_000_000 for jf in directional_juncs}
    j_dist[(day_16_start, False)] = 0
    previous = {jfp: set() for jfp in directional_juncs}
    while directional_juncs:
        closest = min(directional_juncs, key=lambda wp: j_dist[wp])
        neighbour_walks = [*filter(lambda w: w[0] == closest, walks)]
        for nw in neighbour_walks:
            ngb_distance = j_dist[closest] + nw[2]
            if ngb_distance == j_dist[nw[1]]:
                previous[nw[1]].add(nw)
            elif ngb_distance < j_dist[nw[1]]:
                j_dist[nw[1]] = ngb_distance
                previous[nw[1]] = {nw}
        directional_juncs.remove(closest)
    return j_dist, previous


def day_16_build_walks_table(directional_junctions: set, maze: dict) -> {()}:
    walks = set()
    for j in directional_junctions:
        facing = j[1]
        for ngb in day_16_walkable_neighbours(j[0], maze):
            if facing and ngb.y != j[0].y:
                continue
            if (not facing) and ngb.x != j[0].x:
                continue
            cost = 1
            moving_vertically = facing
            walk = [j[0], ngb]
            while True:
                next_available = [
                    *filter(lambda p: p not in walk and p != j[0],
                            day_16_walkable_neighbours(walk[-1], maze))
                            ]
                if not next_available:
                    break
                walk.append(next_available[0])
                cost += 1
                vert = walk[-1].y == walk[-2].y
                if vert != moving_vertically:
                    moving_vertically = vert
                    cost += 1_000
                if (walk[-1], moving_vertically) in directional_junctions:
                    break
            if (walk[-1], moving_vertically) in directional_junctions:
                walks.add((j, (walk[-1], moving_vertically), cost))
        if (j[0], not j[1]) in directional_junctions:
            walks.add((j, (j[0], not j[1]), 1_000))
    return walks


def day_16_walkable_neighbours(point: lib.Point, maze: {}) -> [lib.Point]:
    return [
        mv(point) for mv in pm23.values()
        if mv(point) in maze
    ]


def day_16_load_maze(text: str = "") -> {}:
    global day_16_start, day_16_end
    if not text:
        text = Puzzle24(16).get_text_input().strip("\n")
    walkable = []
    for y, row in enumerate(text.split("\n")):
        for x, char in enumerate(row):
            if char in ".SE":
                walkable.append(lib.Point(y, x))
                if char == "S":
                    day_16_start = lib.Point(y, x)
                if char == "E":
                    day_16_end = lib.Point(y, x)
    return frozenset(walkable)


def day_15_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle24(15).get_text_input().strip("\n")
    layout, directions = text.split("\n\n")
    warehouse = day_15_load_warehouse(layout)
    dimensions = (min(filter(lambda k: warehouse[k] == "\n", warehouse)) + 1,
                  layout.count("\n") + 1)
    robot_pos = [*filter(lambda k: warehouse[k] == "@", warehouse)][0]
    warehouse[robot_pos] = "."
    for move in directions:
        if move == "\n":
            continue
        empty = day_15_look_ahead(
            warehouse, dimensions, robot_pos, move)
        if empty != robot_pos:
            pos_in_front = day_15_look_ahead(
                warehouse, dimensions, robot_pos, move, True
            )
            if empty != pos_in_front:
                warehouse[pos_in_front] = "."
                warehouse[empty] = "O"
            robot_pos = pos_in_front
    return day_15_gps_score(warehouse, dimensions[0])


def day_15_gps_score(wh: {}, wh_width: int) -> int:
    o_positions = [*filter(lambda k: wh[k] in "O[", wh)]
    return sum(
        (100 * (p // wh_width)) + (p % wh_width)
        for p in o_positions
    )


def day_15_look_ahead(warehouse: {}, dims: (int,),
                      initial_pos: int, direction: str,
                      immediate_neighbour: bool = False) -> int:
    wh_width, wh_height = dims
    wh_key_steps = {"^": -wh_width, ">": 1, "v": wh_width, "<": -1}
    line_step = wh_key_steps[direction]
    me = initial_pos
    while me + line_step in warehouse:
        me += line_step
        if immediate_neighbour:
            return me
        if warehouse[me] == ".":
            return me
    return initial_pos


def day_15_p2_widen_warehouse(warehouse: {}) -> {}:
    wide = {}
    for loc in warehouse:
        doubled = (loc * 2) - len(
            [*filter(lambda i: i[0] < loc and i[1] == "\n", warehouse.items())]
        )
        match warehouse[loc]:
            case "O":
                wide[doubled] = "["
                wide[doubled + 1] = "]"
            case "\n":
                wide[doubled] = "\n"
            case "@":
                wide[doubled] = "@"
                wide[doubled + 1] = "."
            case _:
                wide[doubled] = wide[doubled + 1] = warehouse[loc]
    return wide


def day_15_load_warehouse(text: str) -> {}:
    return {
        i: ch
        for i, ch in enumerate(text)
        if ch != "#"
    }


def day_15_part_two(text: str = "", start_pos: int = 0) -> int:
    if not text:
        text = Puzzle24(15).get_text_input().strip("\n")
    layout, directions = text.split("\n\n")
    warehouse = day_15_p2_widen_warehouse(
        day_15_load_warehouse(layout)
    )
    dimensions = (min(filter(lambda k: warehouse[k] == "\n", warehouse)) + 1,
                  layout.count("\n") + 1)
    robot_pos = [*filter(lambda k: warehouse[k] == "@", warehouse)][0]
    warehouse[robot_pos] = "."
    if start_pos:
        robot_pos = start_pos
    for move in directions:
        match move:
            case "\n":
                pass
            case move if move in "<>":
                empty = day_15_look_ahead(
                    warehouse, dimensions, robot_pos, move)
                if empty != robot_pos:
                    pos_in_front = day_15_look_ahead(
                        warehouse, dimensions, robot_pos, move, True
                    )
                    if pos_in_front > empty:
                        for p in range(empty, pos_in_front):
                            warehouse[p] = warehouse[p + 1]
                    elif pos_in_front < empty:
                        for p in range(empty, pos_in_front, -1):
                            warehouse[p] = warehouse[p - 1]
                    robot_pos = pos_in_front
                    warehouse[pos_in_front] = "."
            case _:
                """check if there's a blockage above or below
                    Scenarios:
                        - there is a blockage and robot cannot move
                        - there is empty space immediately ahead 
                            so robot moves forward
                        - there is a wall immediately ahead so 
                            nothing happens
                        - one or more blocks get pushed ahead 
                            because they can all move forward by one"""
                pos_in_front = day_15_look_ahead(
                    warehouse, dimensions, robot_pos, move, True
                )
                if warehouse[pos_in_front] == ".":
                    robot_pos = pos_in_front
                elif warehouse[pos_in_front] in "[]":
                    """ write a function that returns a list of blocks
                        that will move.  If it returns empty, do nothing.
                        Otherwise, move them all and advance robot_pos"""
                    blocks_to_move = day_15_moveable_blocks(
                        pos_in_front, warehouse, dimensions, move)
                    if -1 not in blocks_to_move:
                        distance = dimensions[0] * (1 if move == "v" else -1)
                        for bl in sorted(
                                set(blocks_to_move),
                                reverse=move == "v"
                        ):
                            warehouse[bl + distance] = "["
                            warehouse[bl + distance + 1] = "]"
                            warehouse[bl] = warehouse[bl + 1] = "."
                        robot_pos = pos_in_front
    return day_15_gps_score(warehouse, dimensions[0])


def day_15_moveable_blocks(
        b_pos: int, wh: {}, dims: (int,), dirn: str) -> [int]:
    """list the left edge positions of each block that can move forward"""
    if wh[b_pos] == "]":
        b_pos -= 1
    to_next_row = (1 if dirn == "v" else -1) * dims[0]
    above_left = b_pos + to_next_row
    if all(
            above_left + o in wh and wh[above_left + o] == "."
            for o in range(2)
    ):
        return [b_pos]
    elif any(above_left + p not in wh for p in range(2)):
        return [-1]
    return [b_pos] + [
        bl
        for q in range(2)
        for bl in day_15_moveable_blocks(above_left + q, wh, dims, dirn)
        if wh[above_left + q] in "[]"
    ]


def day_14_part_two() -> int:
    """Assume that a Christmas tree will contain many pairs of
        points distributed symmetrically about the vertical axis"""
    robots = day_14_load_data(Puzzle24(14).get_text_input().strip("\n"))
    grid_w_x_h = 101, 103
    for time in range(-1, -10, -1):
        positions = day_14_all_robot_positions(robots, grid_w_x_h, time)
        day_14_display_grid(positions, grid_w_x_h)
        if day_14_looks_like_a_christmas_tree(positions, grid_w_x_h):
            day_14_display_grid(positions, grid_w_x_h)
            return time
        # took 26 mins to do a million iterations and still no Christmas tree :(
    return 0


def day_14_looks_like_a_christmas_tree(
        robot_positions: [(int,)], grid_dims: (int,)) -> bool:
    left, right = (
        {*day_14_halve_grid(
            robot_positions, bool(ft),
            across_middle=False, grid_dims=grid_dims
        )}
        for ft in range(2)
    )
    centre = grid_dims[0] // 2
    symmetrical = [
        *filter(lambda pt: ((2 * centre) - pt[0], pt[1]) in right, left)
    ]
    # print(f"{len(symmetrical)=}")
    if len(symmetrical) > 0.6 * len(left):
        return True
    return False


def day_14_display_grid(robot_positions: [(int,)], grid_dims: (int,)):
    w, h = grid_dims
    for line in range(h):
        print("".join(
            # "#" if tuple((x, line)) in robot_positions else "."
            f"{robot_positions.count(tuple((x, line)))}" if tuple((x, line)) in robot_positions else "."
            for x in range(w)
        ))


def day_14_part_one(text: str = "") -> int:
    grid_w_x_h = 11, 7
    if not text:
        text = Puzzle24(14).get_text_input().strip("\n")
        grid_w_x_h = 101, 103
    robot_p_v_tuples = day_14_load_data(text)
    final_positions = day_14_all_robot_positions(
        robot_p_v_tuples, grid_w_x_h
    )

    quadrant_scores = [
        len(
            day_14_halve_grid(
                day_14_halve_grid(
                    final_positions, first, False, grid_w_x_h),
                second, True, grid_w_x_h)
        )
        for first, second in itertools.product([True, False], repeat=2)
    ]
    return math.prod(quadrant_scores)


def day_14_halve_grid(
        points: [(int,)], upper_half: bool,
        across_middle: bool, grid_dims: (int,)) -> [(int,)]:
    side_length = grid_dims[across_middle]
    floor = (side_length // 2) + 1 if upper_half else 0
    ceiling = floor + (side_length // 2)
    return [
        pt for pt in points
        if floor <= pt[across_middle] < ceiling
    ]


def day_14_all_robot_positions(
        robot_data: [((int,),)], grid_dims: (int,),
        time_seconds: int = 100) -> [(int,)]:
    return [
        day_14_robot_position(*pv, grid_dims, iterations=time_seconds)
        for pv in robot_data
    ]


def day_14_robot_position(
        original: (int,), velocity: (int,),
        grid_dimensions: (int,), iterations: int = 100) -> (int,):
    return tuple((
        (p + (iterations * v)) % d
        for p, v, d in zip(original, velocity, grid_dimensions)
    ))


def day_14_load_data(text: str) -> [((int,),)]:
    re_num = r"-?\d+"
    return [
        tuple(
            tuple((int(nm.group())
                   for nm in re.finditer(re_num, m.group())))
            for m in re.finditer(f"[p|v]={re_num},{re_num}", line)
        )
        for line in text.split("\n")
    ]


def day_13_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle24(13).get_text_input()
    return sum(
        day_13_prize_cost(
            day_13_extract_game_data(raw_game_data)
        )
        for raw_game_data in text.split("\n\n")
    )


def day_13_extract_game_data(text: str) -> {}:
    game_data = {}
    gd_keys = "abp"
    for k, data in zip(
            gd_keys,
            re.finditer(r": X[+-=]\d+, Y[+-=]\d+", text)
    ):
        values = (
            int(m.group())
            for m in re.finditer(r"[+-]*\d+", data.group())
        )
        game_data[k] = tuple(values)
    return game_data


def day_13_prize_cost(gd: {}) -> int:
    """Zero if prize unreachable"""
    if gd["b"][0] * gd["a"][1] == gd["b"][1] * gd["a"][0]:
        return 0
    bb = ((gd["p"][0] * gd["a"][1]) - (gd["p"][1] * gd["a"][0])) / ((gd["b"][0] * gd["a"][1]) - (gd["b"][1] * gd["a"][0]))
    aa = (gd["p"][1] - (bb * gd["b"][1])) / gd["a"][1]
    if all(math.isclose(n, int(n), rel_tol=0, abs_tol=0.001) for n in (aa, bb)):
        return (3 * int(aa)) + int(bb)
    return 0


def day_13_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle24(13).get_text_input()
    all_game_data = [
        day_13_extract_game_data(t)
        for t in text.split("\n\n")
    ]
    total_spending = 0
    big_number = 10000000000000
    for gd in all_game_data:
        gd["p"] = tuple((x + big_number for x in gd["p"]))
        total_spending += day_13_prize_cost(gd)
    return total_spending


def day_12_load_map_data() -> [{}]:
    text = Puzzle24(12).get_text_input()
    return {
        ch: {m.start() for m in
             re.finditer(ch, text)}
        for ch in {*text}
    }


def day_12_find_regions(map_data: {}) -> {}:
    regions = {}
    for letter in map_data:
        if letter != "\n":
            l_regions = []
            letter_inst = {*map_data[letter]}
            while letter_inst:
                pos = [*letter_inst][0]
                block = day_12_all_contiguous(pos, map_data)
                l_regions.append(block)
                letter_inst -= block
            regions[letter] = l_regions
    return regions


def day_12_all_contiguous(pos: int, map_data: {}, existing: {} = None) -> {int}:
    if existing is None:
        existing = set()
    if day_12_like_neighbours(pos, map_data):
        contiguous = {pos}
        existing.add(pos)
        n_sets = [day_12_all_contiguous(p1, map_data, existing)
                  for p1 in day_12_like_neighbours(pos, map_data)
                  if p1 not in existing]
        for s in n_sets:
            contiguous.update(s)
        return contiguous
    return {pos}


def day_12_like_neighbours(pos: int, map_data: {},
                           inverse: bool = False, text: str = "") -> {int}:
    letter = [k for k, v in map_data.items()
              if pos in v][0]
    if inverse:
        return {
            pp for pp in day_12_all_neighbours(pos, map_data)
            if pp < 0 or pp >= len(text)
               or pp in map_data["\n"]
               or text[pp] != letter
        }
    occurrences = map_data[letter]
    return day_12_all_neighbours(pos, map_data).intersection(occurrences)


def day_12_all_neighbours(pos: int, map_data: {}) -> {int}:
    line_length = min(map_data["\n"]) + 1
    return {pos + i for i in (1, -1, line_length, -line_length)}


def day_12_part_one() -> int:
    md = day_12_load_map_data()
    regions_by_letter = day_12_find_regions(md)
    total_cost = 0
    for letter in regions_by_letter:
        for reg in regions_by_letter[letter]:
            area = len(reg)
            perimeter = sum(
                4 - len(day_12_like_neighbours(pp, md))
                for pp in reg
            )
            total_cost += area * perimeter
    return total_cost


def day_12_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle24(12).get_text_input()
    md = day_12_load_map_data()
    every_square = [
        loc
        for k, v in md.items()
        for loc in v
        if k.isalpha()]
    fence_pairs = [
        (sq, ngb)
        for sq in every_square
        for ngb in day_12_like_neighbours(sq, md, inverse=True, text=text)
    ]
    print(f"{len(fence_pairs)=}")
    return sum(
        day_12_count_sides(r, fence_pairs) * len(r)
        for reg in day_12_find_regions(md).values()
        for r in reg
        # if reg == "T"   # TODO: change back to "\n"
    )


def day_12_count_sides(region_locs: {}, boundary_pairs: []) -> int:
    total_sides, line_length = 0, 141
    b_pairs = [*filter(lambda bp: bp[0] in region_locs, boundary_pairs)]
    for facing_diff, f_name in zip(
            [1, -1, line_length, -line_length],
            ["left", "right", "up", "down"]
    ):
        facing_this_way = [
            *filter(lambda bb: bb[0] == bb[1] + facing_diff, b_pairs)
        ]
        ftw_sides = 0
        # print(facing_this_way)
        ftw_pos = [pp[0] for pp in facing_this_way]
        while ftw_pos:
            n = min(ftw_pos)
            run = []
            while n in ftw_pos:
                run.append(n)
                n += abs(line_length if abs(facing_diff) == 1 else 1)
            ftw_pos = [*filter(lambda x: x not in run, ftw_pos)]
            ftw_sides += 1
        # print(f"There are {ftw_sides} {f_name}-facing sides")
        total_sides += ftw_sides
    return total_sides


def day_11_part_one(text: str = "") -> int:
    stones = day_11_load_stones(text)
    for _ in range(25):
        stones = day_11_blink(stones)
    return len(stones)


def day_11_part_two(text: str = "") -> int:
    stones = {
        s: 1
        for s in day_11_load_stones(text)
    }
    transformations = {}
    for b_count in range(75):
        next_stones = collections.defaultdict(int)
        for st, no_of_st in stones.items():
            if st in transformations:
                tr = transformations[st]
                for nn in {*tr}:
                    next_stones[nn] += (no_of_st * tr.count(nn))
            else:
                trans = day_11_blink([st])
                transformations[st] = trans
                for nn in {*trans}:
                    next_stones[nn] += (no_of_st * trans.count(nn))
        stones = next_stones
        if not (b_count + 1) % 5:
            print(f"Blinked {b_count + 1} times, now have {len(stones)} stones")
            print(f"\t{len(transformations)=}")
    return sum(stones.values())


def day_11_load_stones(text: str = "") -> [int]:
    if not text:
        text = Puzzle24(11).get_text_input()
    return [int(n) for n in text.split(" ")]


def day_11_blink(stones: [int]) -> [int]:
    transformed = []
    for stone in stones:
        if stone == 0:
            transformed.append(1)
        else:
            as_string = f"{stone}"
            if not len(as_string) % 2:
                midway = len(as_string) // 2
                transformed.extend(
                    [
                        int(f"{as_string[:midway]}"),
                        int(f"{as_string[midway:]}")
                    ])
            else:
                transformed.append(stone * 2024)
    return transformed


def day_10_part_one(text: str = "") -> int:
    m = day_10_load_map(text)
    trailheads, summits = (m[i] for i in (0, 9))
    return sum(day_10_can_reach(th, su, m)
               for th in trailheads
               for su in summits)


def day_10_can_reach(trailhead: int, summit: int, area: {}) -> bool:
    path_queue = [[trailhead]]
    while path_queue:
        path = path_queue.pop()
        height = [k for k, v in area.items() if path[-1] in v][0]
        for ngb in filter(
                lambda pt: pt in area[height + 1],
                day_10_neighbours(path[-1], area)
        ):
            if ngb == summit:
                return True
            elif height < 8:
                path_queue.append(path + [ngb])
    return False


def day_10_part_two(text: str = "") -> int:
    m = day_10_load_map(text)
    trailheads, summits = (m[i] for i in (0, 9))
    return sum(day_10_count_ways_to_reach(th, su, m)
               for th in trailheads
               for su in summits)


def day_10_count_ways_to_reach(trailhead: int, summit: int, area: {}) -> int:
    possible_routes = 0
    path_queue = [[trailhead]]
    while path_queue:
        path = path_queue.pop()
        height = [k for k, v in area.items() if path[-1] in v][0]
        for ngb in filter(
                lambda pt: pt in area[height + 1],
                day_10_neighbours(path[-1], area)
        ):
            if ngb == summit:
                possible_routes += 1
            elif height < 8:
                path_queue.append(path + [ngb])
    return possible_routes


def day_10_load_map(text: str = "") -> {}:
    if not text:
        text = Puzzle24(10).get_text_input()
    topo_map = {
        int(t) if t.isnumeric() else t:
        {m.start() for m in re.finditer(t, text)}
        for t in "".join(f"{n}" for n in range(10)) + "\n"
    }
    # for k in topo_map.keys():
    #     print(f"{k:>2}: {topo_map[k]}")
    return topo_map


def day_10_neighbours(location: int, topography: {}) -> {}:
    line_length = min(topography["\n"]) + 1
    return (location + (d * j) for d, j in
            itertools.product((-1, 1), (1, line_length)))


def day_9_part_one(text: str = "") -> int:
    return day_9_checksum(
        day_9_compact(day_9_load_raw_map(text))
    )


def day_9_load_raw_map(text: str = "") -> {}:
    if not text:
        text = Puzzle24(9).get_text_input().strip("\n")
    layout = {}
    get_file = True
    loc, file_id = 0, -1
    for ch in text:
        size = int(ch)
        if get_file:
            file_id += 1
            layout[file_id] = loc, size
        loc += size
        get_file = not get_file
    assert all(s <= 9 for ll, s in layout.values())
    print(f"{len(text)=}")
    assert len(layout) == (len(text) // 2) + 1
    return layout


def day_9_compact(file_pos: {}) -> {}:
    free_space = day_9_find_free_space(file_pos)

    def fs_to_left(pos: int) -> {}:
        return {p: l for p, l in free_space.items() if p < pos}

    compacted = {}
    for file_id in sorted(file_pos.keys(), reverse=True):
        position, length = file_pos[file_id]
        if sum(fs_to_left(position).values()):
            new_compacted_item = []
            while length:
                fsl = fs_to_left(position).items()
                next_free_space = min(fsl) if fsl else (0, 0)
                start, available = next_free_space
                if available:
                    use_fs = min(length, available)
                    length -= use_fs
                    del free_space[start]
                    new_compacted_item.append((start, use_fs))
                    if available - use_fs:
                        free_space[start + use_fs] = available - use_fs
                else:
                    new_compacted_item.append((position, length))
                    break
            compacted[file_id] = new_compacted_item
        else:
            compacted[file_id] = [file_pos[file_id]]
    # print(f"After: {free_space=}")
    # print(f"{compacted=}")
    for k in file_pos:
        assert sum(v[1] for v in compacted[k]) == file_pos[k][1]
        # print(f"{k} ok;", end="")
    return compacted


def day_9_find_free_space(file_pos: {}) -> {}:
    free_space = {}
    for fid in range(max(file_pos.keys())):
        s, l = file_pos[fid]
        fs_start = s + l
        fs_length = file_pos[fid + 1][0] - fs_start
        if fs_length:
            free_space[fs_start] = fs_length
    # print(f"{free_space=}")
    # does it matter that final element of free space = 0?
    assert all(s <= 9 for s in free_space.values())
    print(f"Zeroes in free_space values: {sum(fs_len == 0 for fs_len in free_space.values())}")
    return free_space


def day_9_checksum(compacted_map: {}) -> int:
    checksum = 0
    for k, v in compacted_map.items():
        for li in v:
            start, length = li
            for index in range(length):
                checksum += (start + index) * k
    return checksum


def day_9_part_two(text: str = "") -> int:
    return day_9_checksum(
        day_9_p2_compact(day_9_load_raw_map(text))
    )


def day_9_p2_compact(file_pos: {}) -> {}:
    new_layout = {}
    free_space = day_9_find_free_space(file_pos)

    def large_enough_fs_to_left(pos: int, required_size: int) -> {}:
        return {p: s for p, s in free_space.items()
                if p < pos and s >= required_size}

    for file_id in sorted(file_pos.keys(), reverse=True):
        position, length = file_pos[file_id]
        available_spots = large_enough_fs_to_left(position, length)
        if available_spots:
            leftmost_pos, free_length = min(available_spots.items())
            new_layout[file_id] = [(leftmost_pos, length)]
            del free_space[leftmost_pos]
            free_length -= length
            if free_length:
                free_space[leftmost_pos + length] = free_length
        else:
            new_layout[file_id] = [file_pos[file_id]]
    return new_layout


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
    room = day_6_load_map(text)
    return len(
        {
            vl[0] for vl in
            day_6_visited_locations_with_facing(
                room, [k for k, v in room.items() if v == "^"][0])
            if vl[0] in room
        }
    )


def day_6_visited_locations_with_facing(
        room: {},
        guard_pos: lib.Point,
        facing: str = "U") -> [(lib.Point, str)]:
    visited = {(guard_pos, facing)}
    room[guard_pos] = "."
    while guard_pos in room:
        loc_in_front = pm23[facing](guard_pos)
        if loc_in_front in room:
            if room[loc_in_front] == ".":
                guard_pos = loc_in_front
                if (guard_pos, facing) in visited:
                    break
                visited.add((guard_pos, facing))
            else:
                facing = day_6_turn_to_right(facing)
        else:
            guard_pos = loc_in_front
            visited.add((guard_pos, facing))
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


def day_6_part_two_simple(text: str = "") -> int:
    m = day_6_load_map(text)
    new_obstacles = 0
    starting_point = [k for k, v in m.items() if v == "^"][0]
    all_visited_data = day_6_visited_locations_with_facing(m, starting_point)
    candidate_points = {pt for pt in all_visited_data
                        if pt[0] != starting_point}
    m[starting_point] = "."
    loop_counter = 0
    for candidate, f in candidate_points:
        loop_counter += 1
        m[candidate] = "#"
        pt = day_6_back_up(candidate, f)
        f = day_6_turn_to_right(f)
        will_visit = day_6_visited_locations_with_facing(m, pt, f)
        if all(wv[0] in m for wv in will_visit):
            new_obstacles += 1
        m[candidate] = "."
        if loop_counter % 100 == 0:
            print(f"{loop_counter:,}: {new_obstacles=}")
    return new_obstacles



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
    candidate_points = {pt for pt in all_visited_data
                        if pt[0] != starting_point}
    m[starting_point] = "."
    pf_will_leave = set()
    known_routes_to_exit = []
    len_cp, point_counter = len(candidate_points), 0
    for candidate, f in candidate_points:
        point_counter += 1
        # modded_map = {**m}
        m[candidate] = "#"
        # _, f = [vp for vp in all_visited_data if vp[0] == candidate][0]
        pt = day_6_back_up(candidate, f)
        f = day_6_turn_to_right(f)
        will_visit = []
        no_escape = False
        while pt in m:
            pt = day_6_move_n_steps_in_direction(
                pt, f,
                day_6_distance_to_next_blocker(pt, f, m) - 1
            )
            if (pt, f) in will_visit:
                new_obstacles += 1
                break
            elif pt not in m:
                if all(
                        lib.manhattan_distance(wv[0], candidate) > 1
                        for wv in will_visit
                ):
                    if will_visit and will_visit[0] not in pf_will_leave:
                        known_routes_to_exit.append(will_visit)
                    pf_will_leave.update(will_visit)
            will_visit.append((pt, f))
            if (pt, f) in pf_will_leave and len(will_visit) > 10:
                last_three = will_visit[-3:]
                for kre in known_routes_to_exit:
                    if ((pt, f) in kre) and any(
                        kre[i:i + 3] == last_three
                        for i, loc in enumerate(kre[:-3])
                    ):
                        spot = kre.index(last_three[0])
                        # print(f"Found repeated route!! {last_three=} found={kre[spot:spot + 3]}")
                        if day_6_escape_route_blocked(kre[spot:], candidate):
                            # print(" . . . but the route is blocked :(")
                            no_escape = True
                        else:
                            pt = (-1, -1), ""
                            break
            f = day_6_turn_to_right(f)
        m[candidate] = "."
        if point_counter % 100 == 0:
            print(f"{point_counter:,} of {len_cp} points. "
                  f"{len(will_visit):,} visited before "
                  f"{'leaving grid' if pt not in m else 'looping'} "
                  f"{len(known_routes_to_exit)=}")
    assert 0 < new_obstacles < day_6_part_one(text)
    return new_obstacles


def day_6_escape_route_blocked(route: [(lib.Point, str)],
                               blocker: lib.Point) -> bool:
    for i, loc in enumerate(route[:-1]):
        pt1, pt2 = (x[0] for x in (loc, route[i + 1]))
        if (pt1[0] <= blocker[0] <= pt2[0] or
                pt2[0] <= blocker[0] <= pt1[0]) and pt1[1] == blocker[1]:
            return True
        if (pt1[1] <= blocker[1] <= pt2[1] or
                pt2[1] <= blocker[1] <= pt1[1]) and pt1[0] == blocker[0]:
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
