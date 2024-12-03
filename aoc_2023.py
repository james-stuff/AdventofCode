import collections
import copy
import itertools
import pprint
import heapq
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


day_23_walkable, day_23_slopes = {}, {}


def day_23_load_scene(text: str = ""):
    global day_23_walkable, day_23_slopes
    if not text:
        text = Puzzle23(23).get_text_input().strip("\n")
    walkable = []
    move_translation = {"<": "L", ">": "R", "^": "U", "v": "D"}
    for y, row in enumerate(text.split("\n")):
        for x, char in enumerate(row):
            if char in "<>^v.":
                walkable.append(lib.Point(y, x))
                if char != ".":
                    day_23_slopes[lib.Point(y, x)] = move_translation[char]
    day_23_walkable = frozenset(walkable)
    # print(f"{len(day_23_walkable)=}\t{len(day_23_slopes)=}")
    # print(day_23_slopes)


def day_23_part_one(text: str = "") -> int:
    day_23_load_scene(text)
    walk_queue = collections.deque([[lib.Point(0, 1)]])
    completed = []
    destination = lib.Point(*max(day_23_walkable))
    while walk_queue:
        route = walk_queue.popleft()
        for new_route in day_23_extend_route(route):
            if new_route[-1] == destination:
                completed.append(new_route)
            else:
                walk_queue.append(new_route)
    print(f"{len(completed)=}.  Walks are of lengths:")
    for walk in completed:
        print(len(walk) - 1, end="\t")
    return max(len(cw) - 1 for cw in completed)


def day_23_extend_route(route: [lib.Point], watch_slopes: bool = True) -> [[lib.Point]]:
    possible_routes = []
    allowed_moves = point_moves_2023.values()
    if watch_slopes and route[-1] in day_23_slopes:
        allowed_moves = [point_moves_2023[day_23_slopes[route[-1]]]]
    for move in allowed_moves:
        step_to = move(route[-1])
        if step_to in day_23_walkable and step_to not in route:
            possible_routes.append(route + [step_to])
    return possible_routes


def day_23_part_two(text: str = "") -> int:
    day_23_load_scene(text)
    walk_queue = collections.deque([[lib.Point(0, 1)]])
    completed = []
    destination = lib.Point(*max(day_23_walkable))
    log_size = 1
    while walk_queue:
        route = walk_queue.popleft()
        for new_route in day_23_extend_route(route, watch_slopes=False):
            if new_route[-1] == destination:
                completed.append(new_route)
            else:
                walk_queue.append(new_route)
        if len(walk_queue) >= log_size:
            log_size *= 2
            print(f"{len(walk_queue)}, {len(route)}")
    print(f"{len(completed)=}.  Walks are of lengths:")
    for walk in completed:
        print(len(walk) - 1, end="\t")
    return max(len(cw) - 1 for cw in completed)


def day_22_load_bricks(text: str = "") -> {}:
    if not text:
        text = Puzzle23(22).get_text_input().strip("\n")

    def co_ord_str_to_tuple(csv: str) -> (int,):
        return tuple(int(v) for v in csv.split(","))

    bricks = {}
    for row in text.split("\n"):
        end_1, end_2 = row.split("~")
        start = co_ord_str_to_tuple(end_1)
        end = co_ord_str_to_tuple(end_2)
        bricks[start] = tuple(e - s + 1 for e, s in zip(end, start))
    return bricks


def day_22_collapse(bricks: {}) -> {}:
    """in order of ascending initial height, drop each
        brick as low as it will go """
    fallen = {}
    
    def resting_level(br: (int,), dim: (int,)) -> int:
        for z in range(br[2], 0, -1):
            br_lower_by_one = {
                k: v
                for k, v in fallen.items()
                if k[2] + v[2] == z
            }
            if br_lower_by_one:
                br_footprint = day_22_brick_footprint(br, dim)
                if any(br_footprint.intersection(
                        set(
                            (xf, yf)
                            for xf in range(brkf[0], brkf[0] + brdf[0])
                            for yf in range(brkf[1], brkf[1] + brdf[1])
                        )
                    )
                    for brkf, brdf in br_lower_by_one.items()
                ):
                    return z
        return 1

    for brick in sorted(bricks, key=lambda b: b[2]):
        new_z = max(1, resting_level(brick, bricks[brick]))
        fallen[*brick[:2], new_z] = bricks[brick]
    return fallen


def day_22_brick_footprint(location: (int,), dimensions: (int,)) -> set:
    return set(
        (x, y)
        for x in range(location[0], location[0] + dimensions[0])
        for y in range(location[1], location[1] + dimensions[1])
    )


def day_22_get_supporting_relationships(structure: {}) -> {}:
    sr = {k: [] for k in structure.keys()}
    for br, dims in structure.items():
        top = br[2] + dims[2]
        one_level_up = {k: v for k, v in structure.items() if k[2] == top}
        if one_level_up:
            my_footprint = day_22_brick_footprint(br, dims)
            sr[br] = [
                hb_loc for hb_loc, hb_dims in one_level_up.items()
                if my_footprint.intersection(day_22_brick_footprint(hb_loc, hb_dims))
            ]
    return sr


def day_22_part_one(text: str = "") -> int:
    collapsed = day_22_collapse(
        day_22_load_bricks(text)
    )
    supporting = day_22_get_supporting_relationships(collapsed)
    """For each brick in supporting:
        if every brick it supports is also supported by at least
        one other brick, or it is supporting no bricks, it can be safely removed"""
    return len(day_22_removable_bricks(supporting))


def day_22_removable_bricks(supporting: {}) -> [(int,)]:
    safe = []
    all_supported_bricks = [a for v in supporting.values() for a in v]
    for brick in supporting:
        if supporting[brick]:
            if all(
                all_supported_bricks.count(supported) > 1
                for supported in supporting[brick]
            ):
                safe.append(brick)
        else:
            safe.append(brick)
    return safe


day_22_cr_memo = {}


def day_22_chain_reaction(supports: {}, brick_to_remove: (int,)) -> int:
    """how many OTHER bricks fall if this one is removed?"""
    falling = 0
    local_supports = copy.deepcopy(supports)
    print(f"{brick_to_remove=}", end=" -> ")

    def is_supported(brick: (int,)) -> bool:
        return brick in [
            a for v in local_supports.values()
            for a in v
        ]

    unsupported = [brick_to_remove]
    while unsupported:
        brick_to_fall = unsupported.pop()
        falling += (brick_to_fall is not brick_to_remove)
        might_fall = local_supports[brick_to_fall]
        del local_supports[brick_to_fall]
        unsupported.extend(
            [*filter(lambda b: not is_supported(b), might_fall)]
        )
    print(f"{falling=}")
    return falling


def day_22_part_two(text: str = "") -> int:
    collapsed = day_22_collapse(
        day_22_load_bricks(text)
    )
    supporting = day_22_get_supporting_relationships(collapsed)
    no_effect = day_22_removable_bricks(supporting)
    return sum(
        day_22_chain_reaction(supporting, brick)
        for brick in sorted(collapsed.keys(), key=lambda k: k[2], reverse=True)
        if brick not in no_effect
    )


day_21_rocks = {}


def day_21_load_garden(layout: str) -> {}:
    return {
        lib.Point(r, c): symbol == "S"
        for r, row in enumerate(layout.split("\n"))
        for c, symbol in enumerate(row)
        if symbol != "#"
    }


def day_21_count_reachable_plots(garden: {}, steps: int) -> int:
    for _ in range(steps):
        garden = day_21_make_step(garden)
    return len([*filter(lambda v: v, garden.values())])


def day_21_make_step_and_print_garden(garden: {}) -> {}:
    garden = day_21_make_step(garden)
    day_21_print_garden(garden)


def day_21_print_garden(garden: {}, view_radius: int = 11):
    # side = max(garden)[0] + 1
    # side = 21

    def char(c: int, r: int) -> str:
        pt = lib.Point(c, r)
        if pt in garden:
            if garden[pt]:
                return "O"
            elif lib.manhattan_distance(pt, origin) <= steps:
                if lib.manhattan_distance(pt, origin) % 2 != even:
                    return "@"
            return "."
        return "#"

    print("")
    origin = (5, 5)
    even = garden[origin]
    steps = max(lib.manhattan_distance(origin, pt)
                for pt in filter(lambda k: garden[k], garden))
    view_range = 5 - view_radius, 5 + view_radius + 1
    for col in range(*view_range):
        print(" ".join(char(col, row) for row in range(*view_range)))
    return garden


def day_21_rocks_within(garden: {}, radius: int) -> int:
    g_size = max(garden)[0] + 1
    area = [(r, c)
            for r in range(-g_size, g_size)
            for c in range(-g_size, g_size)
            if lib.manhattan_distance((5, 5), (r, c)) <= radius
            ]
    return len([*filter(lambda p: p not in garden, area)])


def day_21_make_step(garden: {}) -> {}:
    new_garden = {k: v for k, v in garden.items()}
    for loc in filter(lambda k: garden[k], garden):
        new_garden[loc] = False
        for mv in point_moves_2023.values():
            adj_loc = mv(loc)
            if adj_loc in garden:
                new_garden[adj_loc] = True
    return new_garden


def day_21_run_for_n_steps(garden: {}, ns: int, record: bool = False) -> int:
    frontline = [[*filter(lambda k: garden[k], garden)][0]]
    even_visited, odd_visited = (set() for _ in range(2))
    results = {}
    for i in range(ns):
        new_fl = []
        for point in frontline:
            for move in point_moves_2023.values():
                new_loc = move(point)
                if new_loc in garden:
                    if ((i % 2 and new_loc not in even_visited) or
                            ((not i % 2) and new_loc not in odd_visited)):
                        if new_loc not in new_fl:
                            new_fl.append(new_loc)
            if i % 2:
                odd_visited.add(point)
            else:
                even_visited.add(point)
        frontline = new_fl
        if record:
            results[i + 1] = len(frontline) + len(even_visited if i % 2 else odd_visited)
            if i % 50 == 0:
                print(f"Running step {i + 1}")
    if record:
        return results
    return len(frontline) + len(odd_visited if ns % 2 else even_visited)


def day_21_duplicate_garden(original: {}, offset: (int,)) -> {}:
    y_offset, x_offset = offset
    size = max(original)[0] + 1
    return {
        lib.Point(k[0] + (y_offset * size), k[1] + (x_offset * size)):
            v if offset == (0, 0) else False
        for k, v in original.items()
    }


def day_21_count_reachable_plots_in_infinite_garden(
        base_garden: {}, target_steps: int, results: {}) -> int:
    bg_size = max(base_garden)[0] + 1

    def shortfall(steps_in: int) -> int:
        return ((steps_in + 1) ** 2) - results[steps_in]

    def shortfall_difference(steps_in: int, back_steps: int) -> int:
        return shortfall(steps_in) - shortfall(steps_in - back_steps)

    shortfall_diffs = [
        shortfall_difference(k, bg_size)
        for k in range(bg_size * 5, max(results.keys()) + 1, bg_size)
    ]
    increases = [
        sf - shortfall_diffs[i]
        for i, sf in enumerate(shortfall_diffs[1:])
    ]
    increment = increases[0]
    assert all(d == increment for d in increases)
    print(f"The shortfall from (n + 1)-squared grows by {increment} every {bg_size} steps")
    start = max(filter(lambda kk: kk % bg_size == target_steps % bg_size, results.keys()))
    print(f"{start=}")
    starting_shortfall = ((start + 1) ** 2) - results[start]
    big_steps = (target_steps - start) // bg_size
    final_shortfall_diff = (shortfall_difference(start, bg_size) * big_steps) + (increment * (big_steps + 1) * big_steps // 2)
    return ((target_steps + 1) ** 2) - starting_shortfall - final_shortfall_diff


def day_21_first_n_results(base_garden: {}, n: int = 100) -> {}:
    r_dup = (n * 2) // max(base_garden)[0]
    # print(f"\n{r_dup=}")
    big_garden = {}
    for y in range(-r_dup, r_dup + 1):
        for x in range(-r_dup, r_dup + 1):
            big_garden.update(
                day_21_duplicate_garden(base_garden, (y, x))
            )
    print(f"{len(big_garden) = }")
    # results = {}
    # for step in range(n):
    #     big_garden = day_21_make_step(big_garden)
    #     results[step + 1] = sum(big_garden.values())
    #     if step % 5 == 0:
    #         print(f"Running simulation: {step}th step")
    # return results
    return day_21_run_for_n_steps(big_garden, n, record=True)


def day_20_load_connections(text: str) -> {}:
    connections, module_ids = ({} for _ in range(2))
    for match in re.finditer("[&%]*.+ -> .+", text):
        row = match.group()
        module_id = row.split(" ")[0]
        module_ids[module_id[1:]] = module_id
        recipients = tuple(
            re.search(r"\s\w.*", row).group()[1:].split(", ")
        )
        connections[module_id] = recipients
    for k, v in connections.items():
        if any(recipient in module_ids for recipient in v):
            connections[k] = tuple(module_ids[r] for r in v)
    assert len(connections) == len(text.strip("\n").split("\n"))
    return connections


def day_20_set_up(text: str = "") -> ({}):
    if not text:
        text = Puzzle23(20).get_text_input()
    connections = day_20_load_connections(text)
    flip_flops = {m: False for m in connections if m[0] == "%"}
    conjunctions = {}
    for k in filter(lambda ck: ck[0] == "&", connections):
        conjunctions[k] = {
            c: False
            for c, v in connections.items()
            if k in v
        }
    assert len(flip_flops) == text.count("%")
    assert len(conjunctions) == text.count("&")
    return flip_flops, conjunctions, connections


def day_20_push_the_button(ff: {}, conj: {}, conn: {}, cycle_id: int) -> (int,):
    pulse_queue = collections.deque([("button", "broadcaster", False)])
    return day_20_process_button_press(ff, conj, conn, pulse_queue, cycle_id)


data_log = []


def day_20_process_button_press(ff: {}, conj: {}, conn: {}, pulse_queue: collections.deque, cycle_id: int = 0) -> {}:
    global data_log
    if cycle_id == 3877:
        data_log = [["origin", "recipient", "pulse"]]
    counts = [0, 0]
    max_queue_size = 1
    while pulse_queue:
        origin, recipient, pulse = pulse_queue.popleft()
        if recipient == "rx" and not pulse:
            print("BOOM!!!")
        # if origin in ["gp", "bn", "rt", "cz"]:
        #     print(f"&{origin} sends {pulse}")
        # if destination == "vf":
        #     print(f"vf is sent {pulse} by {origin}")
        counts[pulse] += 1
        send_pulse = None
        if recipient in ff and (not pulse):
            ff[recipient] = not ff[recipient]
            send_pulse = ff[recipient]
        elif recipient in conj:
            conj[recipient][origin] = pulse
            send_pulse = not all(conj[recipient].values())
        elif recipient == "broadcaster":
            send_pulse = pulse
        if send_pulse is not None:
            for next_destination in conn[recipient]:
                if cycle_id == 3877:
                    data_log.append([recipient, next_destination, int(send_pulse)])
                pulse_queue.append((recipient, next_destination, send_pulse))
        max_queue_size = max(max_queue_size, len(pulse_queue))
    # print(f"{max_queue_size=}")
    # if max_queue_size not in (24, 28, 29):
    #     print(f"*** {cycle_id=}, {max_queue_size=}")
        # print(f" Conjunction states: {', '.join(c + ':' + str(len([k for k, v in conj[c].items() if v])) + '/' + str(len(c)) for c in conj)}")
    # if cycle_id % 1_000 == 0:
    #     c_states = {c: (len([*filter(lambda v: v, conj[c].values())]), len(conj[c])) for c in conj}
    #     print(f" {c_states}")
    with open("Day20one_press_output.csv", "w") as file:
        file.write(
            "\n".join(
                ",".join(f"{item}" for item in row)
                for row in data_log
            )
        )
    return tuple(counts)


def day_20_part_one(text: str = "") -> int:
    pb_args = day_20_set_up(text)
    lows, highs = 0, 0
    for _ in range(1_000):
        l, h = day_20_push_the_button(*pb_args, _)
        lows += l
        highs += h
    return lows * highs


def day_20_part_two() -> int:
    pb_args = day_20_set_up()
    _, conjunctions, _ = pb_args
    button_presses = 0
    while button_presses < 50_200:
        button_presses += 1
        day_20_push_the_button(*pb_args, button_presses)
        c_values = [sum(conjunctions[cj].values()) for cj in conjunctions]#["pm", "mk", "pk", "hf"]]
        if button_presses > 50_000:
            print([f"{c}{j}" for c, j in zip(conjunctions.keys(), c_values)])
        if not any(c_values):
            pprint.pprint(conjunctions)
            print(f"Condition met on cycle {button_presses}")
            # doesn't get there in under 2 minutes
            break
    return button_presses


def day_19_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle23(19).get_text_input()

    def part_two_process(state: tuple, all_workflows: str) -> [tuple]:
        out_states = []
        raw_instruction = re.search(f"\n{state[-1]}" + "{.+}", all_workflows).group()
        s, e = (raw_instruction.index(ch) for ch in "{}")
        raw_tests = raw_instruction[s + 1:e]
        for test in raw_tests.split(","):
            if re.search(r"[xmas][<>]", test):
                state, new = day_19_p2_state_splitter(state, test)
                outcome = test[test.index(":") + 1:]
                if outcome.islower():
                    new = tuple((*new[:-1], outcome))
                    out_states.append(new)
                elif outcome == "A":
                    accepted.add(new)
                elif outcome == "R":
                    rejected.add(new)
            elif test.islower() and 1 < len(test) < 4:
                out_states.append(tuple((*state[:-1], test)))
            elif test == "A":
                accepted.add(state)
            elif test == "R":
                rejected.add(state)
        return out_states
    workflows, _ = day_19_load_inputs(text)
    initial = tuple((*(1, 4_001) * 4, "in"))
    queue = [initial]
    accepted = set()
    rejected = set()
    while queue:
        next_state = queue.pop()
        print(f"Queue size: {len(queue)}, accepted: {len(accepted)}")
        queue += part_two_process(next_state, workflows)
    pprint.pprint(accepted)
    return day_19_number_of_combinations(accepted)


def day_19_number_of_combinations(states: set) -> int:
    return sum(
        math.prod(st[n + 1] - st[n] for n in range(0, 8, 2))
        for st in states)


def day_19_p2_state_splitter(state: tuple, test: str) -> tuple:
    letter = test[0]
    inequality = test[1]
    threshold = int(test[2:test.index(":")])
    index_of_min = "xmas".index(letter) * 2
    # if not (state[index_of_min] < threshold < state[index_of_min + 1]):
    #     print(f"Threshold is outside current state range for {state=}, {test=}")
    #     assert False
    #     return state
    state, new = tuple(tuple(threshold + (inequality == ">")
                             if i == ci
                             else v
                             for i, v in enumerate(state))
                       for ci in (index_of_min, index_of_min + 1)
                       )
    if inequality == ">":
        state, new = new, state
    return state, new


def day_19_part_one(input_text: str = "") -> int:
    workflows, part_details = day_19_load_inputs(input_text)
    return sum(sum(p.values()) for p in part_details if day_19_is_accepted(p, workflows))


def day_19_is_accepted(part: {}, workflows: str) -> bool:
    print(f"Part: {part}")
    assert all(letter in part for letter in "xmas")
    return day_19_process_instruction(part, workflows)


def day_19_process_instruction(part: {}, workflows: str, line_id: str = "in") -> bool:
    raw_instruction = re.search(f"\n{line_id}" + "{.+}", workflows).group()
    # print(raw_instruction)
    print(line_id, end=" -> ")
    s, e = (raw_instruction.index(ch) for ch in "{}")
    raw_tests = raw_instruction[s + 1:e]
    for test in raw_tests.split(","):
        result = test
        # print(test)
        if re.search(r"[xmas][<>]", test):
            # print(f"\tThis is an inequality test")
            letter = test[0]
            assert letter in "xmas"
            inequality = test[1]
            assert inequality in "<>"
            threshold = int(test[2:test.index(":")])
            assert 0 < threshold <= 4_000
            next_step = test[test.index(":") + 1:]
            if next_step.islower():
                assert 1 < len(next_step) < 4
                assert re.search(next_step, workflows)
            else:
                assert next_step in "AR"
            # print(f"It wants me to do {next_step} if {letter} {inequality} {threshold}")
            result = next_step if eval(f"part['{letter}'] {inequality} {threshold}") else ""
            # print(f"{result=}")
        # else:
        #     input(f"Test pattern not found in {test}, on line {raw_instruction}.  Continue?")
        if result:
            if result in "AR":
                print(result)
                return result == "A"
            return day_19_process_instruction(part, workflows, result)
    assert False


def day_19_load_inputs(text: str = "") -> (str, [{}]):
    if not text:
        text = Puzzle23(19).get_text_input().strip("\n")
    instructions, _, raw_part_data = text.partition("\n\n")
    part_parameters = [{m.group()[0]: int(m.group()[2:])
                        for m in re.finditer(r"\w=\d+", raw_part)}
                       for raw_part in raw_part_data.split("\n")]
    return "\n" + instructions, part_parameters     # add "\n" to ensure line_id search always works


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


def day_17_part_one(text: str = ""):
    if not text:
        text = Puzzle23(17).get_text_input().strip("\n")
    day_17_load_city(text)
    return day_17_using_dijkstra()


def day_17_part_two(text: str = ""):
    if not text:
        text = Puzzle23(17).get_text_input().strip("\n")
    day_17_load_city(text)
    return day_17_using_dijkstra(4, 10)


def day_17_using_dijkstra(min_blocks: int = 1, max_blocks: int = 3) -> int:
    side = len(day_17_city)
    day_17_space = {(x, y, d, s)
                    for x in range(side)
                    for y in range(side)
                    for d in "URDL"
                    for s in range(1, max_blocks + 1)}
    distances_table = {p: 1_000_000 for p in day_17_space}
    altered = []
    for origin in ((0, 0, dd, 0) for dd in "RD"):
        day_17_space.add(origin)
        distances_table[origin] = 0
        heapq.heappush(altered, (0, origin))
    print(f"Space has {len(day_17_space)} points")
    multiple_end_points = {pt: 1_000_000
                           for pt in day_17_space
                           if pt[0] == pt[1] == side - 1
                           and pt[2] in "DR"
                           and pt[3] >= min_blocks}
    while day_17_space and max(multiple_end_points.values()) == 1_000_000:
        next_point = heapq.heappop(altered)[1]
        travelled_so_far = distances_table[next_point]
        if next_point in multiple_end_points:
            multiple_end_points[next_point] = travelled_so_far
        neighbours = day_17_find_neighbours(next_point, min_blocks)
        for ngh in neighbours:
            if ngh in day_17_space:
                x, y, _, _ = ngh
                neighbour_distance = travelled_so_far + day_17_city[x][y]
                if neighbour_distance < distances_table[ngh]:
                    distances_table[ngh] = neighbour_distance
                    heapq.heappush(altered, (neighbour_distance, ngh))
        day_17_space.remove(next_point)
        if len(day_17_space) % 10_000 == 0:
            print(f" . . . {len(day_17_space)} points remaining, {len(altered)=}")
    pprint.pprint(multiple_end_points)
    return min(multiple_end_points.values())


def day_17_find_neighbours(location: (int,), min_blocks: int = 1) -> [(int,)]:
    """Generates neighbours in any direction except reverse.
        Does NOT ensure validity.  This should be done by checking
        each value is in the space"""
    new_neighbours = []
    x, y, drn, dist = location
    valid_directions = [c for i, c in enumerate("URDL")
                        if c == drn or
                        i % 2 != "URDL".index(drn) % 2]
    for d in valid_directions:
        new_x, new_y = point_moves_2023[d](lib.Point(x, y))
        new_dist = 0
        if d == drn:
            new_dist = dist + 1
        elif dist >= min_blocks:
            new_dist = 1
        if new_dist:
            new_neighbours.append((new_x, new_y, d, new_dist))
    return new_neighbours


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


day_12_memo = {}
day_12_m_usage_counts = {}


def day_12_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle23(12).get_text_input()
    total = 0
    for no, line in enumerate(text.strip("\n").split("\n")):
        print(f"===== Line {no + 1} =====")
        total += day_12_p2_process(line)
    # return sum(day_12_p2_process(line) for line in text.strip("\n").split("\n"))
    return total


def day_12_p2_process(damaged_record: str) -> int:
    p1_springs, p1_groups = day_12_get_record_details(damaged_record)
    return day_12_dictionary_based_solution(
        "?".join([p1_springs] * 5),
        [g for _ in range(5) for g in p1_groups]
    )


def day_12_part_one(text: str = "") -> int:
    global day_12_m_usage_counts
    day_12_m_usage_counts = {}
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
    group_limits = day_12_group_limits_by_social_distancing(springs, groups, return_max_end=True)
    print(f"{group_limits=}")
    if all(gl[1] == gl[0] + gr - 1 for gl, gr in zip(group_limits, groups)):
        print(f"{springs} returns with social distancing shortcut")
        return 1
    """Dictionary is {n: {possible group indices}} for each character position
        in springs.  Initially empty list for each one.
        - remove any position key that definitely cannot be a hash
        - traverse springs from both ends to narrow it down further?"""
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
                             if k in range(group_limits[gi][0],
                                           group_limits[gi][1] + groups[gi])
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
    # print(f"Before: {dictionary=}")
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
    springs = springs.strip(".")
    if (springs, ",".join(f"{g}" for g in groups)) in day_12_memo:
        # day_12_m_usage_counts[(springs, ",".join(f"{g}" for g in groups))] += 1
        return day_12_memo[(springs, ",".join(f"{g}" for g in groups))]

    def is_valid(start: int, group_id: int) -> bool:
        excluded_springs = springs[:start] + springs[start + groups[group_id]:]
        if excluded_springs.count("#") + groups[group_id] > sum(groups):
            return False
        return "." not in springs[start:start + groups[group_id]] and\
               "#" not in "".join(springs[i] if i in range(len(springs)) else ""
                                  for i in (start - 1, start + groups[group_id]))

    start_params = day_12_group_limits_by_social_distancing(springs, groups)
    if len(groups) == 1:
        min_start, max_start = start_params[0]
        arrangements = sum([
            is_valid(s, 0)
            for s in range(min_start, max_start + 1)
        ])
        # day_12_m_usage_counts[(springs, ",".join(f"{g}" for g in groups))] = 1
        # if len(springs) < 51:
        day_12_memo[(springs, ",".join(f"{g}" for g in groups))] = arrangements
        return arrangements
    else:
        possible_arrangements = 0
        min_start_1, max_start_1 = start_params[0]
        for st in range(min_start_1, max_start_1 + 1):
            if is_valid(st, 0):
                possible_arrangements += day_12_count_possible_arrangements(
                    springs[st + groups[0] + 1:], groups[1:]
                )
            if springs[st] == "#":  # group cannot start any later than first known hash
                break
        if len(springs) < 51:
            day_12_memo[(springs, ",".join(f"{g}" for g in groups))] = possible_arrangements
        return possible_arrangements


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


def day_12_min_space(hash_groups: [int], at_end: bool = True) -> int:
    return sum(hash_groups) + len(hash_groups) + (not at_end)


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

