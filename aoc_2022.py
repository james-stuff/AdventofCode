import copy
import itertools
from main import Puzzle
import library as lib
from collections import defaultdict, deque
from itertools import product, cycle
from copy import deepcopy
import re
import random


class Puzzle22(Puzzle):
    def get_text_input(self) -> str:
        with open(f'inputs_2022\\input{self.day}.txt', 'r') as input_file:
            return input_file.read()


day_21_monkeys = {}


def day_21_part_two() -> int:
    """Realised in exploratory testing that there is a simple linear
        relationship between the 'humn' value and eventual output.
        Output is only an int every 5th value, starting at 3, and decreases
        by 51 for every five increments of the input"""
    side, target = day_21_setup_for_part_two()
    day_21_monkeys["humn"] = 3
    starting_point = day_21_evaluate_recursively(side)
    return int((starting_point - target) * 5 // 51) + 3


def day_21_setup_for_part_two(text: str = "") -> (str, int):
    global day_21_monkeys
    if not text:
        text = Puzzle22(21).get_text_input()
    day_21_monkeys = day_21_load_monkeys(text)
    p1_root_operator = day_21_monkeys["root"][5]
    day_21_monkeys["root"] = day_21_monkeys["root"].replace(p1_root_operator, "==")
    day_21_monkeys["humn"] = "NaN"
    root_lhs, root_rhs = day_21_monkeys["root"][:4], day_21_monkeys["root"][8:]
    required_value, human_side = 0, ""
    for side in (root_lhs, root_rhs):
        print(f"{side=}")
        try:
            required_value = day_21_evaluate_recursively(side)
        except KeyError:
            human_side = side
    print(f"Human contributes to monkey {human_side}, which must return {required_value}")
    return human_side, required_value


def day_21_part_one() -> int:
    global day_21_monkeys
    day_21_monkeys = day_21_load_monkeys(Puzzle22(21).get_text_input())
    return int(day_21_evaluate_recursively("root"))


def day_21_load_monkeys(text: str) -> {}:
    def parse_instruction(instr: str):
        if instr.isnumeric():
            return int(instr)
        return instr

    return {line[:4]: parse_instruction(line[6:])
            for line in text.split("\n") if line.strip()}


def day_21_evaluate_recursively(monkey: str) -> int | str:
    op_or_value = day_21_monkeys[monkey]
    if isinstance(op_or_value, int):
        return op_or_value
    first_id, _, remainder = op_or_value.partition(" ")
    operation, _, second_id = remainder.partition(" ")
    first_input, second_input = (day_21_evaluate_recursively(name)
                                 for name in (first_id, second_id))
    return eval(f"{first_input} {operation} {second_input}")


def day_20_part_one():
    encrypted_list = Puzzle22(20).input_as_list(int)
    return day_20_extract_solution(day_20_mix(encrypted_list))


def day_20_part_two():
    encryption_key = 811589153
    encrypted_list = [v * encryption_key for v in Puzzle22(20).input_as_list(int)]
    mixed_list = day_20_mix(encrypted_list, mixing_steps=10)
    return day_20_extract_solution(mixed_list)


def day_20_mix(initial: [int], steps: int = 0, mixing_steps: int = 1) -> [int]:
    """keys and values in linked list are indices of the original list"""
    sequence_length = len(initial)
    if not steps:
        steps = sequence_length
    linked_list = {n: n + 1 for n in range(sequence_length)}
    linked_list[sequence_length - 1] = 0
    rev_ll = {v: k for k, v in linked_list.items()}
    for mix in range(mixing_steps):
        print(f"Mix step: {mix + 1}")
        for original_index in range(sequence_length)[:steps]:
            # if not original_index % 5:
            #     print(original_index)
            current_before = rev_ll[original_index]
            jumps = initial[original_index] % (sequence_length - 1)
            if jumps == 0:
                continue
            if jumps < 0:
                new_before = current_before
                for _ in range(-jumps):
                    new_before = rev_ll[new_before]
                new_after = linked_list[new_before]
            else:
                new_after = linked_list[original_index]
                for _ in range(jumps):
                    new_after = linked_list[new_after]
                new_before = rev_ll[new_after]
            linked_list[current_before] = linked_list[original_index]
            linked_list[original_index] = new_after
            linked_list[new_before] = original_index
            rev_ll[linked_list[current_before]] = current_before
            rev_ll[original_index] = new_before
            rev_ll[new_after] = original_index
            # print(original_index, initial[original_index])
            assert len(linked_list) == len(initial)
    out = []
    nn = 0
    for _ in range(sequence_length):
        out.append(initial[nn])
        nn = linked_list[nn]
    assert nn == 0
    return out


def day_20_extract_solution(mixed_values: [int]) -> int:
    zero_position = mixed_values.index(0)
    answer = 0
    cycled = cycle(mixed_values)
    for n in range(zero_position + 3001):
        val = next(cycled)
        if n > zero_position and n % 1000 == zero_position % 1000:
            answer += val
            print(f"Found: {val}", end="\t")
    return answer


def day_20_properties_tests(initial: [int], final: [int]) -> bool:
    # assert len(set(initial)) == len(initial)  -> there are duplicates in real thing
    assert len(final) == len(initial)
    print(f"initial: {len(set(initial))} -> final: {len(set(final))}")
    assert len(set(final)) == len(set(initial))
    assert set(final).issubset(set(initial))
    assert set(final) == set(initial)
    return True


ORE, CLAY, OBSIDIAN, GEODE = range(4)
day_19_costs = {
    ORE: {ORE: 4},
    CLAY: {ORE: 2},
    OBSIDIAN: {ORE: 3, CLAY: 14},
    GEODE: {ORE: 2, OBSIDIAN: 7},
}
day_19_productive_paths = {}
ore_robots_cap = max(day_19_costs[r][ORE] for r in day_19_costs)
clay_cap, obsidian_cap = (day_19_costs[r + 1][r] for r in (CLAY, OBSIDIAN))
day_19_doing_part_two = False


def day_19_part_two() -> int:
    global day_19_costs, day_19_doing_part_two
    day_19_doing_part_two = True
    results = {}
    text = Puzzle22(19).get_text_input()
    it = day_19_blueprint_start_indices(text)
    for i, blueprint in enumerate(it[:3]):
        print(f"+++ BLUEPRINT {i + 1} +++")
        if i == 2:
            results[i + 1] = 1
        #     break
        end_pos = it[i + 1] if i < len(it) - 1 else blueprint + 500
        day_19_costs = day_19_read_blueprint(text[blueprint:end_pos])
        day_19_set_robot_caps()
        results[i + 1] = day_19_breadth_first_search()
    print(results)
    answer = 1
    for v in results.values():
        answer *= v
    return answer


def day_19_method_for_part_two() -> int:
    b_and_b_run = day_19_flexible_branch_and_bound(extend_by_mins=15)
    for duration in 4, 4, 4:
        b_and_b_run = day_19_flexible_branch_and_bound(b_and_b_run, duration)
    return day_19_evaluate_complete_paths(b_and_b_run, part_two=True)


def day_19_part_one(test_input: str = None) -> int:
    global day_19_costs
    results = {}
    text = test_input if test_input else Puzzle22(19).get_text_input()
    it = day_19_blueprint_start_indices(text)
    for i, blueprint in enumerate(it):
        print(f"+++ BLUEPRINT {i + 1} +++")
        end_pos = it[i + 1] if i < len(it) - 1 else blueprint + 500
        day_19_costs = day_19_read_blueprint(text[blueprint:end_pos])
        # results[i + 1] = day_19_evaluate_complete_paths(
        #     day_19_branch_and_bound_best_paths())
        results[i + 1] = day_19_breadth_first_search()
    print(results)
    return day_19_get_total_score(results)


def day_19_blueprint_start_indices(input_text: str) -> [int]:
    return [match.start() for match in re.finditer("Blueprint", input_text)]


def day_19_read_blueprint(text: str) -> {}:
    def name_to_constant(name_of_robot: str) -> int:
        return globals()[name_of_robot.upper()]

    costs = {}
    outline_of_costs = re.findall(r"Each.*?\.", text)
    for sentence in outline_of_costs:
        robot = sentence.split()[1]
        cost_breakdown = re.findall(r"[0-9]+\s\w+", sentence)
        costs[name_to_constant(robot)] = {name_to_constant(resource): int(quantity)
                                          for item in cost_breakdown
                                          for quantity, _, resource in [item.partition(" ")]}
    return costs


def day_19_set_robot_caps():
    global ore_robots_cap, clay_cap, obsidian_cap
    ore_robots_cap = max(day_19_costs[r][ORE] for r in day_19_costs)
    clay_cap, obsidian_cap = (day_19_costs[r + 1][r] for r in (CLAY, OBSIDIAN))


def day_19_breadth_first_search() -> int:
    def geode_ceiling(path: {}) -> int:
        nonlocal geode_ceil_memo
        geodes_from_existing_robots = day_19_geode_count(path)
        best_robot, latest_time = max(path.values()), max(path.keys())
        if (best_robot, latest_time) in geode_ceil_memo:
            max_additional_geodes = geode_ceil_memo[(best_robot, latest_time)]
        else:
            earliest_geode_minute = max(path.keys()) + 1
            if max(path.values()) < OBSIDIAN: # or day_19_minutes_required_to_produce(GEODE, path) > 1:
                earliest_geode_minute += 1
            max_additional_geodes = sum(
                final_minute - mm
                for mm in range(earliest_geode_minute, final_minute + 1)
            )
            geode_ceil_memo[(best_robot, latest_time)] = max_additional_geodes
        return geodes_from_existing_robots + max_additional_geodes

    def upper_bound_check(path: {}) -> bool:
        if best_score:
            return geode_ceiling(path) >= best_score
        return True

    origin = {0: 0}
    best_score = 0
    geode_ceil_memo = {}
    final_minute = [24, 32][day_19_doing_part_two]
    visited, queue = 0, deque([origin])
    while queue:
        vertex = queue.popleft()
        visited += 1
        if vertex == {0: 0, 5: 0, 8: 0, 10: 1, 11: 0,
                         12: 1, 13: 1, 14: 1, 15: 1,
                         16: 1, 17: 1, 18: 2, 19: 1}:
            print("")
        if upper_bound_check(vertex):
            # For each neighbour not visited, mark it as visited, and enqueue it
            for neighbour in day_19_get_path_neighbours(vertex):
                """There is no duplication"""
                if upper_bound_check(neighbour):
                    queue.append(neighbour)
                    if max(neighbour.values()) >= OBSIDIAN:
                        lower_bound_path = copy.deepcopy(neighbour)
                        # calculate lower bound, producing a new geode robot whenever possible:
                        m = max(neighbour.keys())
                        while m < final_minute:
                            m += day_19_minutes_required_to_produce(GEODE, lower_bound_path)
                            if m < final_minute:
                                lower_bound_path[m] = GEODE
                        geodes_from_ngb = day_19_geode_count(lower_bound_path)
                        if geodes_from_ngb > best_score:
                            best_score = geodes_from_ngb
                            print(f"{best_score=} after visiting {visited:>6,} vertices. "
                                  f"Queue size is {len(queue):>8,}", end="")
                            queue = deque([*filter(upper_bound_check, queue)])
                            print(f"-> {len(queue):>8,}")  # path: {vertex}")
                            print(f"\tPath: {neighbour}")
                            print(f"\tLBP: {lower_bound_path}")
    print(geode_ceil_memo)
    return best_score


def day_19_path_score(path: {}, minute: int) -> int:
    """score a path based on combination of current resources and production rates"""
    current_resources = day_19_current_resources(path, minute)
    minutes_remaining = [24, 32][day_19_doing_part_two]
    if all(current_resources[res] >= day_19_costs[GEODE][res] * minutes_remaining
           for res in (ORE, OBSIDIAN)):
        """This never happens"""
        print(f"maxed out on {path}")
        print(10 / 0)
        return 10 ** 24
    resources_score = sum(10 ** (k * 3) * v for k, v in current_resources.items())
    rates_score = sum(10 ** ((robot + 4) * 3) *
                      (len([*filter(lambda v: v == robot, path.values())]))
                      for robot in range(max(path.values()) + 1))
    return resources_score + rates_score


def day_19_get_path_neighbours(path: {}) -> [{}]:
    out = []
    final_minute = [24, 32][day_19_doing_part_two]
    current_minute = max(path.keys())
    best_next_robot = min(GEODE, max(path.values()) + 1)
    for new_robot in range(best_next_robot + 1):
        if day_19_robot_cap_check(path, new_robot):
            ready_minute = current_minute + \
                           day_19_minutes_required_to_produce(new_robot, path)
            if ready_minute < final_minute:
                new_path = copy.deepcopy(path)
                new_path[ready_minute] = new_robot
                out.append(new_path)
    return out


def day_19_robot_cap_check(path: {}, proposed_robot: int) -> bool:
    if proposed_robot == GEODE:
        return True
    return [*path.values()].count(proposed_robot) < \
           [ore_robots_cap, clay_cap, obsidian_cap][proposed_robot]


def day_19_geodes_found_on_random_walks(time_available: int) -> int:
    day_19_random_walk({0: 0}, end_minute=time_available, n_walks=500)
    best_paths = [day_19_productive_paths[t]
                  for t in sorted(day_19_productive_paths.keys(),
                                  reverse=True)][:3]
    cutoff = 3
    best_stems = [{k: v for k, v in p.items() if k < cutoff} for p in best_paths]
    while cutoff < time_available:
        for st in best_stems:
            day_19_random_walk(st, end_minute=time_available, n_walks=1000)
            best_paths = [day_19_productive_paths[t]
                          for t in sorted(day_19_productive_paths.keys(),
                                          reverse=True)][:3]
            best_stems = [{k: v for k, v in p.items() if k < cutoff} for p in best_paths]
        cutoff += 1
        print(f"{cutoff=} best: {max(day_19_productive_paths.keys())}")
        # pprint.pprint(aoc_2022.day_19_productive_paths)
    return max(day_19_productive_paths.keys())


def day_19_random_walk(path_stem: {}, end_minute: int = 24, n_walks: int = 100) -> [{}]:
    global day_19_productive_paths
    ore_robots_cap = max(day_19_costs[r][ORE] for r in day_19_costs)
    clay_cap, obsidian_cap = (day_19_costs[r + 1][r] for r in (1, 2))

    def cap_check(path: {}, proposed_robot: int) -> bool:
        if proposed_robot == GEODE:
            return True
        return [*path.values()].count(proposed_robot) < \
               [ore_robots_cap, clay_cap, obsidian_cap][proposed_robot]
    walks = []
    for _ in range(n_walks):
        walk = copy.deepcopy(path_stem)
        next_minute = max(path_stem.keys())
        while next_minute < end_minute:
            best_next_robot = min(GEODE, max(walk.values()) + 1)
            next_robot = random.randrange(best_next_robot + 1)
            if cap_check(walk, next_robot):
                next_minute += day_19_minutes_required_to_produce(next_robot, walk)
                if next_minute < end_minute:
                    walk[next_minute] = next_robot
        walks.append(walk)
        geode_count = day_19_geode_count(walk, part_two=(end_minute == 32))
        if geode_count:
            day_19_productive_paths[geode_count] = walk
    return walks


def day_19_evaluate_complete_paths(paths: [{}], part_two: bool = False) -> int:
    print(f"Taking these to the end gives us {len(paths)} paths altogether")
    best_path = max(paths, key=lambda p: day_19_geode_count(p)) if paths else {}
    print(f"The best path is:")
    print(best_path)
    open_geodes = day_19_geode_count(best_path)
    print(f"This produces {open_geodes} open geodes")
    print(f"There are "
          f"{len([*filter(lambda p: day_19_geode_count(p) == open_geodes, paths)])}"
          f" paths that achieve this result")
    return open_geodes


def day_19_next_possible_paths(path_so_far: dict, stop_at_minute: int = 22,
                               part_two: bool = False) -> [dict]:
    possible_new_paths = [path_so_far]
    most_advanced_robot = max(path_so_far.values())
    ore_robots_cap = max(day_19_costs[r][ORE] for r in day_19_costs)
    clay_cap, obsidian_cap = (day_19_costs[r + 1][r] for r in (1, 2))
    def cap_check(proposed_robot: int) -> bool:
        if proposed_robot == GEODE:
            return True
        return [*path_so_far.values()].count(proposed_robot) < \
               [ore_robots_cap, clay_cap, obsidian_cap][proposed_robot]

    for robot in range(most_advanced_robot +
                       (2 if most_advanced_robot < GEODE else 1)):
        minute_ready = max(path_so_far.keys()) + \
                       day_19_minutes_required_to_produce(robot, path_so_far)
        if (minute_ready < stop_at_minute + 2): # and \
                # day_19_production_is_worthwhile(robot, minute_ready):
            if cap_check(robot):
                new_path = copy.deepcopy(path_so_far)
                new_path[minute_ready] = robot
                possible_new_paths += day_19_next_possible_paths(
                    new_path, stop_at_minute, part_two)
                if path_so_far in possible_new_paths:
                    possible_new_paths.remove(path_so_far)
    return possible_new_paths


def day_19_flexible_branch_and_bound(paths_so_far: [{}] = [{0: 0}],
                                     extend_by_mins: int = 14) -> [{}]:
    print(f"Flexible B&B on {extend_by_mins}-minute run starts with "
          f"{len(paths_so_far)} paths")
    all_next_paths = []
    for path in paths_so_far:
        all_next_paths += day_19_next_possible_paths(
            path, max(path.keys()) + extend_by_mins, True)
    scored_paths = [(day_19_pseudo_score_partial_path(path), path)
                    for path in all_next_paths]
    distribution = {score: len([*filter(lambda sp: sp[0] == score, scored_paths)])
                    for score in set(v[0] for v in scored_paths)}

    no_of_new_paths, max_score = 0, 0
    for sc in sorted(distribution):
        max_score = sc
        no_of_new_paths += distribution[sc]
        if no_of_new_paths > 300:
            break
    print(f"{distribution} -> {max_score=}")
    paths_to_proceed_with = [fp[1] for fp in
                             filter(lambda sp: sp[0] <= max_score, scored_paths)]

    print(f"-> {len(paths_to_proceed_with)} paths (from {len(all_next_paths)})")
    return paths_to_proceed_with


def day_19_branch_and_bound_best_paths() -> [dict]:
    initial_paths = day_19_next_possible_paths({0: 0}, 14)
    day_19_evaluate_complete_paths(initial_paths)
    scored_paths = [(day_19_pseudo_score_partial_path(path), path)
                    for path in initial_paths]
    best_score = min(t[0] for t in scored_paths)
    distribution = {score: len([*filter(lambda sp: sp[0] == score, scored_paths)])
                    for score in set(v[0] for v in scored_paths)}
    no_of_top_two_scores = sum(distribution[sc] for sc in sorted(distribution)[:2])
    tolerance = 1 if no_of_top_two_scores > 10 else 2
    print(f"{distribution} -> {tolerance=}")
    paths_to_proceed_with = [fp[1] for fp in filter(
        lambda sp: sp[0] <= best_score + tolerance, scored_paths)]
    final_paths = []
    for ppw in paths_to_proceed_with:
        final_paths += day_19_next_possible_paths(ppw)
    return final_paths


def day_19_pseudo_score_partial_path(pp: dict) -> int:
    """favour paths that have more robots producing costlier resources"""
    weightings = {res: day_19_costs[res + 1][res] for res in range(1, 3)}
    part_two = any(minute > 23 for minute in pp.keys())
    if any(v == GEODE for v in pp.values()):
        return -100 * len([*filter(lambda v: v == GEODE, pp.values())])
        # tried increasing GEODE weighting to 1000, but later on the majority of paths
        # have geode robots, so too many are being evaluated
        # without any geode weighting, productive paths are missed
    return - sum(w * [*pp.values()].count(r)
                 for r, w in weightings.items())#\
           # - (4 * day_19_geode_count(pp, part_two))
    # return - len([*filter(lambda v: v == GEODE, pp.values())])


def day_19_old_pseudo_score_partial_path(pp: dict) -> int:
    """upper bound assumption: produce the next robot up asap.
        then assume each more advanced robot will be created
        on a minute-by-minute basis, with geode robots all the way to the end.
        Result is the minute at which geode robot production would start,
        the earliest being the most desirable"""
    start_minute = max(pp.keys())
    next_robot = max(pp.values()) + 1
    part_two = any(minute > 23 for minute in pp.keys())
    return start_minute + GEODE - next_robot \
           - day_19_geode_count(pp, part_two)
           # - len([*filter(lambda v: v == GEODE, pp.values())])


def day_19_production_is_worthwhile(robot: int, minute: int) -> bool:
    """could producing such a robot actually lead to increased geode production?"""
    final_minute = [24, 32][day_19_doing_part_two]
    if robot == GEODE:
        return True
    # if minute >= final_minute - 3: #21:
    #     return robot != CLAY
    return True


def day_19_minutes_required_to_produce(robot: int, path: {}) -> int:
    current_resources = day_19_current_resources(path, max(path.keys()))
    time_costs_by_resource = []
    for commodity, cost in day_19_costs[robot].items():
        production_minutes = 0
        amount_needed = max(cost - current_resources[commodity], 0)
        if amount_needed:
            production_rate = len([*filter(lambda rbt: rbt == commodity, path.values())])
            time, remainder = divmod(amount_needed, production_rate)
            production_minutes = time + (1 if remainder else 0)
        time_costs_by_resource.append(production_minutes)
    return max(*time_costs_by_resource, 0) + 1  # +1 is the minute taken to build robot


def day_19_current_resources(path: dict, minute: int) -> dict:
    current_resources = defaultdict(int)
    for m, robot in path.items():
        if m < 32:
            current_resources[robot] += minute - m
            if m > 0:
                for cost_item, amount in day_19_costs[robot].items():
                    current_resources[cost_item] -= amount
        else:
            print(f"32-min plus path found: {path=}")
    return current_resources


def day_19_purchasing_options(resources_available: dict) -> [int]:
    options = []
    for robot in day_19_costs:
        if all(
                (resource in resources_available) and
                (resources_available[resource] >= required_quantity)
                for resource, required_quantity in day_19_costs[robot].items()
        ):
            options.append(robot)
    return options


def day_19_geode_count(path: {}) -> int:
    end_time = 32 if day_19_doing_part_two else 24
    return sum(end_time - minute for minute, robot in path.items()
               if robot == GEODE and minute < end_time)


def day_19_get_total_score(quality_ratings: {}) -> int:
    return sum(id_no * max_geodes for id_no, max_geodes in quality_ratings.items())


def day_18_part_one() -> int:
    cubes_input = Puzzle22(18).get_text_input()
    return day_18_total_surface_area(day_18_get_cube_set(cubes_input))


def day_18_get_cube_set(raw_cubes: str) -> set:
    return {tuple(int(d) for d in desc.split(","))
            for desc in raw_cubes.strip().split("\n")}


def day_18_total_surface_area(cubes: {}) -> int:
    cubes = list(cubes)
    exposed_faces = 6 * len(cubes)
    for i, cb in enumerate(cubes):
        for other in cubes[i + 1:]:
            if day_18_cubes_are_adjacent(cb, other):
                exposed_faces -= 2
    return exposed_faces


def day_18_cubes_are_adjacent(cube_1: (int, int, int), cube_2: (int, int, int)) -> bool:
    touch_score = sum(abs(dim - cube_2[j]) for j, dim in enumerate(cube_1))
    return touch_score == 1


def day_18_part_two(raw_cube_list: str = None, part_1_solution: int = 4340) -> int:
    if not raw_cube_list:
        raw_cube_list = Puzzle22(18).get_text_input()
    all_cubes = day_18_get_cube_set(raw_cube_list)
    container_side = day_18_calculate_containing_cube_size(all_cubes)
    all_empty_areas = day_18_find_all_empty_regions(container_side, all_cubes)
    empty_space_locations = (container_side ** 3) - len(all_cubes)
    assert sum(len(s) for s in all_empty_areas) == empty_space_locations
    return day_18_subtract_hidden_surface_area(all_empty_areas, part_1_solution)


def day_18_find_all_empty_regions(container_size: int, cubes: set) -> [{}]:
    all_empty_areas = []
    for z in range(container_size):
        for y in range(container_size):
            for x in range(container_size):
                point = x, y, z
                done = False
                if point not in cubes:
                    for area in all_empty_areas:
                        if day_18_is_adjacent_to_space(point, area):
                            area.add(point)
                            done = True
                            break
                    if not done:
                        all_empty_areas.append({point})
        print(f"z={z}")
        all_empty_areas = day_18_merge_adjoining_spaces(all_empty_areas)
        if z < 4:
            print(f"{len(all_empty_areas)} empty areas, "
                  f"of sizes {', '.join(str(len(a)) for a in all_empty_areas)}")
            for a in all_empty_areas:
                print(a)
    return all_empty_areas


def day_18_calculate_containing_cube_size(cubes: set) -> int:
    field = [max(c[dim] for c in cubes) for dim in range(3)]
    # print(f"The space is {' x '.join(str(s) for s in field)}")
    return max(field) + 1


def day_18_merge_adjoining_spaces(empty_spaces: [{}]) -> [{}]:
    total_points = sum(len(sp) for sp in empty_spaces)
    print(f"Total points before: {total_points}, "
          f"spaces sizes: {', '.join(str(len(a)) for a in empty_spaces)}")
    adjacent_pairs = []
    for i, region in enumerate(empty_spaces):
        for j, other_area in enumerate(empty_spaces[i + 1:]):
            for point in region:
                if day_18_is_adjacent_to_space(point, other_area):
                    adjacent_pairs.append((i, i + 1 + j))
                    break
    print(f"Adjacent pairs by index: {adjacent_pairs}")
    merge_cascade = sorted(adjacent_pairs, key=lambda pr: (pr[1], pr[0]), reverse=True)
    print(f"Cascade: {merge_cascade}")
    kill_list = []
    for master_index, subset_index in merge_cascade:
        empty_spaces[master_index].update(empty_spaces[subset_index])
        kill_list.append(subset_index)

    empty_spaces = [ea for i, ea in enumerate(empty_spaces) if i not in kill_list]
    print(f"Total points after: {sum(len(a) for a in empty_spaces)}, "
          f"spaces sizes: {', '.join(str(len(a)) for a in empty_spaces)}")
    assert sum(len(sp) for sp in empty_spaces) == total_points
    return empty_spaces


def day_18_is_adjacent_to_space(point: tuple, space: {}) -> bool:
    x, y, z = point
    adjacent_points = set()
    for disp in (-1, 1):
        adjacent_points.add((x + disp, y, z))
        adjacent_points.add((x, y + disp, z))
        adjacent_points.add((x, y, z + disp))
    return any(pt in space for pt in adjacent_points)


def day_18_subtract_hidden_surface_area(empty_spaces: [{}],
                                        total_surface: int = 4340) -> int:
    surfaces_by_volume = {1: 6, 2: 10}
    containing_volume = max(len(space) for space in empty_spaces)
    deductions = 0
    for sp in empty_spaces:
        if len(sp) != containing_volume:
            if len(sp) < 3:
                deductions += surfaces_by_volume[len(sp)]
            else:
                deductions += day_18_total_surface_area(sp)
    external_surface_area = total_surface - deductions
    assert external_surface_area < total_surface
    return external_surface_area


day_17_rocks = """####

.#.
###
.#.

..#
..#
###

#
#
#
#

##
##"""


day_17_cavern = ["+-------+"]


def day_17_part_one() -> int:
    return day_17_run_simulation()


def day_17_part_two() -> int:
    return day_17_height_of_trillion_rock_tower()


def day_17_height_of_trillion_rock_tower(test_jets: str = None) -> int:
    repeated_rows = 53 if test_jets else 2_783
    trillion = 1_000_000_000_000
    """how many rocks need to fall to get to this height?
        multiply that by the big_number // repeated_rows
        Try this:
            1. drop 3000 rocks and get the height
            2. count the number of rocks that increase height by repeated_rows
            3. divide the big number - 3000 by (2)
            4. run the simulation the remainder number of times to
                add on the additional height gained by dropping big number of rocks"""
    incoming_rocks = day_17_load_rocks()
    jet_cycle = day_17_load_jets(test_jets)
    print("")
    for _ in range(5_000):
        day_17_rock_falls(jet_cycle, incoming_rocks)
    height_after_5_000 = len(day_17_cavern) - 1
    print(f"Height after 5,000 rocks: {height_after_5_000}")
    rocks_per_repeat = 0
    while len(day_17_cavern) - 1 < height_after_5_000 + repeated_rows:
        day_17_rock_falls(jet_cycle, incoming_rocks)
        rocks_per_repeat += 1
    print(f"Rocks required to gain {repeated_rows} rows = {rocks_per_repeat}")
    assert len(day_17_cavern) - 1 == height_after_5_000 + repeated_rows
    repeats_required = (trillion - 5_000 - rocks_per_repeat) // rocks_per_repeat
    print(f"Need to repeat {repeats_required} times")
    rocks_dropped = 5_000 + rocks_per_repeat + (rocks_per_repeat * repeats_required)
    height_before_final_push = len(day_17_cavern) - 1
    for r in range(trillion - rocks_dropped):
        day_17_rock_falls(jet_cycle, incoming_rocks)
    return height_after_5_000 + (repeated_rows * (repeats_required + 1)) + \
           len(day_17_cavern) - 1 - height_before_final_push


def day_17_run_simulation(test_jets: str = None, how_many_times: int = 2022) -> int:
    incoming_rocks = day_17_load_rocks()
    jet_cycle = day_17_load_jets(test_jets)
    print("")
    for _ in range(how_many_times):
        day_17_rock_falls(jet_cycle, incoming_rocks)
    return len(day_17_cavern) - 1


def day_17_rock_falls(jet_cycle: (), incoming_rocks: ()):
    initial_hashes = day_17_testing_count_hashes()
    falling_rock = deepcopy(next(incoming_rocks))
    falling_rock.bottom_y = len(day_17_cavern) + 3
    can_move = True
    while can_move:
        jet = next(jet_cycle)
        # print(jet, end="")
        falling_rock.move_horizontally(jet)
        can_move = falling_rock.move_down()
    falling_rock.settle()
    assert day_17_testing_count_hashes() == initial_hashes + \
           sum(row.count('#') for row in falling_rock.inverted_shape)


def day_17_testing_count_hashes() -> int:
    return sum(row.count("#") for row in day_17_cavern)


class Day17Rock:
    def __init__(self, representation: str):
        self.inverted_shape = representation.split("\n")[::-1]
        self.width = len(self.inverted_shape[0])
        self.bottom_y = 0           # current vertical y-coordinate of the bottom row
        self.left_x = 3             # current x-coordinate of bottom left corner

    def move_horizontally(self, direction: str):
        displacement = 1 if direction == ">" else -1
        new_left_x = self.left_x + displacement
        if 0 < new_left_x and new_left_x + self.width - 1 < 8:
            self.left_x = new_left_x
            if self.collides_with_any_rock():
                self.left_x -= displacement

    def move_down(self) -> bool:
        """Move down one row if possible.  Return value is success or failure"""
        if self.bottom_y > 1:  # not hit rock bottom
            self.bottom_y -= 1
            if self.collides_with_any_rock():
                self.bottom_y += 1
                return False
            return True
        return False

    def settle(self):
        global day_17_cavern

        def modify_row(existing_row: str, new_rock_row: str) -> str:
            modified_row = existing_row[:self.left_x]
            overlay = ""
            for j, c in enumerate(new_rock_row):
                if new_rock_row[j] == "#":
                    overlay += "#"
                else:
                    overlay += existing_row[self.left_x + j]
            modified_row += overlay + existing_row[self.left_x + self.width:]
            return modified_row

        for i, row in enumerate(self.inverted_shape):
            if self.bottom_y + i >= len(day_17_cavern):
                day_17_cavern.append("|.......|")
            day_17_cavern[self.bottom_y + i] = modify_row(day_17_cavern[self.bottom_y + i],
                                                          row)

    def get_top_y(self) -> int:
        return self.bottom_y + len(self.inverted_shape) - 1

    def collides_with_any_rock(self):
        # print(f"I am a rock at ({self.left_x}, {self.bottom_y})")
        for y in range(self.bottom_y, self.get_top_y() + 1):
            for x in range(self.left_x, self.left_x + self.width):
                if y in range(len(day_17_cavern)):
                    my_sign = self.inverted_shape[y - self.bottom_y][x - self.left_x]
                    if my_sign == "#" and day_17_cavern[y][x] == "#":
                        return True
        return False


def day_17_load_rocks(reset_cavern: bool = True) -> ():
    if reset_cavern:
        global day_17_cavern
        day_17_cavern = ["+-------+"]
    return cycle([Day17Rock(r) for r in day_17_rocks.split("\n\n")])


def day_17_load_jets(jets: str = None) -> ():
    if not jets:
        jets = Puzzle22(17).get_text_input().strip()
        print(f"Length of jets = {len(jets)}")
    return cycle(jets)


day_16_route_table = {}


def day_16_part_one() -> int:
    global day_16_route_table
    text = Puzzle22(16).get_text_input()
    valve_data = day_16_load_valve_data(text)
    day_16_route_table = day_16_build_distances_table(valve_data)
    # return day_16_by_traversal_of_all_routes_between_worthwhile_points(text)
    tuple_routes = day_16_best_journey_by_taking_it_one_step_at_a_time(valve_data)
    print(f"{len(tuple_routes):,} routes found")
    return max([day_16_score_tuple_journey(r, valve_data) for r in tuple_routes])


def day_16_part_two() -> int:
    global day_16_route_table
    text = Puzzle22(16).get_text_input()
    day_16_route_table = day_16_build_distances_table(day_16_load_valve_data(text))
    return day_16_breadth_first_search(text)


def day_16_by_traversal_of_all_routes_between_worthwhile_points(input_text: str) -> int:
    valve_data = day_16_load_valve_data(input_text)
    routes = day_16_get_all_valid_routes([])
    print(f"{len(routes):,} routes found")
    return max([day_16_score_journey(r, valve_data) for r in routes])


def day_16_breadth_first_search(text: str) -> int:
    def upper_bound_check(path: ((str,),)) -> bool:
        """assume each remaining valve, in descending order of flow rate,
         can be reached in one step"""
        ub_score = day_16_score_double_headed_tuple_journey(path, valve_data)
        me, elephant = path
        route_times = {ident: day_16_journey_time(pp)
                       for ident, pp in zip(("Me", "Elephant"), path)}
        remaining_valves = sorted([v[0] for k, v in valve_data.items()
                                   if v[0] > 0 and k not in [*me] + [*elephant]])
        while remaining_valves and ub_score < best and \
                any(rt < 24 for rt in route_times.values()):
            next_best_valve = remaining_valves.pop()
            most_effective_player = min(route_times.keys(), key=lambda k: route_times[k])
            route_times[most_effective_player] += 2
            ub_score += next_best_valve * (26 - route_times[most_effective_player])
        return ub_score >= best

    def lower_bound_score(path: ((str,),)) -> int:
        """Each player goes to optimal next valve at the time, until the end"""
        lb_score = day_16_score_double_headed_tuple_journey(path, valve_data)
        itineraries = {ind: (day_16_journey_time(path[ind]), path[ind][-1])
                       for ind in range(2)}
        available = [k for k, v in valve_data.items()
                     if v[0] > 0 and k not in [*path[0]] + [*path[1]]]
        while available and any(t[0] < 24 for t in itineraries.values()):
            for pi in itineraries:
                t_player, last_visited = itineraries[pi]
                if t_player > 23:
                    continue
                reachable_valves = [*filter(lambda item: (item[0] in available) and
                                                         (item[1] + 1 + t_player < 26),
                                            day_16_route_table[last_visited].items())]
                if reachable_valves:
                    def achievable_flow(valve_id: str, dist: int) -> int:
                        elapsed_time = t_player + dist + 1
                        if elapsed_time > 25:
                            return 0
                        return valve_data[valve_id][0] * (26 - elapsed_time)
                    best_valve, distance = max(reachable_valves,
                                               key=lambda i: achievable_flow(*i))
                    lb_score += achievable_flow(best_valve, distance)
                    t_player += distance + 1
                    itineraries[pi] = t_player, best_valve
                    available.remove(best_valve)
                else:
                    itineraries[pi] = 26, ""
            # if max(v[0] for v in itineraries.values()) < 21:
            #     break
        return lb_score

    def remove_duplicates(queue: deque) -> deque:
        if not queue:
            return queue
        new_list = []
        while queue:
            vertex = queue.popleft()
            if vertex not in new_list:
                reverse_vertex = vertex[1], vertex[0]
                if reverse_vertex not in new_list:
                    new_list.append(vertex)
        return deque(new_list)

    global day_16_route_table
    best = 0
    valve_data = day_16_load_valve_data(text)
    day_16_route_table = day_16_build_distances_table(valve_data)
    vertices = deque([tuple(("AA",) for _ in range(2))])
    while vertices:
        route = vertices.popleft()
        neighbours = [*filter(upper_bound_check, day_16_get_route_neighbours(route))]
        for ngb in neighbours:
            lwr_bnd_score = lower_bound_score(ngb)
            if lwr_bnd_score > best:
                if lwr_bnd_score > 2495:
                    print("debug")
                print(f"Best score is now {lwr_bnd_score}.  Deque size: {len(vertices)} -> ", end="")
                best = lwr_bnd_score
                vertices = deque([*filter(upper_bound_check, vertices)])
                vertices = remove_duplicates(vertices)
                vertices.append(ngb)
                vertices = deque(sorted(vertices, key=lower_bound_score, reverse=True))
                print(len(vertices))
            vertices.append(ngb)
        if len(vertices) > 10000:
            print(f"Vertices {len(vertices)} -> ", end="")
            vertices = remove_duplicates(vertices)
            vertices = deque(sorted(vertices, key=lower_bound_score, reverse=True)[:5])
            print(len(vertices))
    return best


def day_16_get_route_neighbours(route: ((str,),)) -> [((str,),)]:
    neighbours = []
    me, elephant = route
    for p_id, player in enumerate(route):
        time_used = day_16_journey_time(player)
        distances = day_16_route_table[player[-1]]
        for point in distances:
            if time_used + distances[point] < 25 and (point not in [*me] + [*elephant]):
                if p_id == 0:
                    # if len(player) <= len(elephant) + 1:
                    neighbours.append(((*player, point), elephant))
                else: #if len(player) <= len(me) + 1:   # do something here to stop duplication?
                    neighbours.append((me, (*player, point)))
    return neighbours


def day_16_by_teaming_up_with_elephant(input_text: str) -> int:
    valves = day_16_load_valve_data(input_text)
    best_routes = {"no route": 0}
    useful_valves = [*filter(lambda vd: valves[vd][0] > 0, valves)]
    routes = [(("AA",), ("AA",))]
    completed_routes = []
    # TODO: could there be duplication of effort?  I.e. two pairs of routes
    #   that are the inverse of each other?
    #       Also times where the human finishes early but the elephant keeps going?
    for step in range(len(useful_valves)):
        print(f"{step=:>2}, have to look at {len(routes):>8} routes")
        next_routes = []
        for route in routes:
            my_route, elephant_route = route
            unvisited_valves = set(useful_valves) - set(my_route) - set(elephant_route)
            if len(unvisited_valves) == 1:
                uv = [*unvisited_valves][0]
                new_route = tuple([*my_route] + [uv]), tuple([*elephant_route] + [uv])
                completed_routes.append(new_route)
            else:
                uv_pairs = [*itertools.permutations(unvisited_valves, 2)]
                for v1, v2 in uv_pairs:
                    new_route = tuple([*my_route] + [v1]), tuple([*elephant_route] + [v2])
                    new_rm, new_re = new_route
                    # There are some scenarios where someone finishes first
                    if day_16_journey_time(new_rm) > 25:
                        new_route = my_route, new_re
                    if day_16_journey_time(new_re) > 25:
                        new_route = new_rm, elephant_route
                    if len(unvisited_valves) == 2:
                        completed_routes.append(new_route)
                    elif day_16_score_double_headed_tuple_journey(new_route, valves) > \
                            max(best_routes.values()) + 30:
                        next_routes.append(new_route)
        # if step % 2 == 0:
        for r in routes:
            best_routes[r] = day_16_score_double_headed_tuple_journey(r, valves)
        routes = next_routes
    return max(day_16_score_double_headed_tuple_journey(j, valves) for j in completed_routes)
    return 0    # old code follows:
    routes = day_16_get_all_valid_team_routes([[], []])
    print(f"{len(routes):,} routes found")
    return max([day_16_score_double_headed_journey(r, valves) for r in routes])


def day_16_get_all_valid_team_routes(routes_so_far: [[str]]) -> [[[str]]]:
    """as day_16_get_all_valid_routes() except:
            - total time available is only 26 minutes (so must get to tap in <25 mins)
            - must rule out already-visited taps in EITHER of two lists per route"""
    def extend_route(existing_route: [str], next_step: str) -> [str]:
        if next_step == "":
            return existing_route
        location = existing_route[-1] if existing_route else "AA"
        distance = day_16_route_table[location][next_step]
        return existing_route + ([""] * (distance - 1)) + ([next_step] * 2)

    current_locations = [r[-1] if r else "AA" for r in routes_so_far]
    possible_next_steps = []
    for i, player_route in enumerate(routes_so_far):
        other_route = routes_so_far[int(not i)]
        assert player_route is not other_route
        neighbours = day_16_route_table[current_locations[i]]
        options = [k for k, v in neighbours.items()
                   if len(player_route) + v < 25
                   and k != "AA"
                   and k not in player_route
                   and k not in other_route]
        if not options:
            options = [""]
        possible_next_steps.append(options)
    possible_routes = []
    valid_choices = [p for p in product(*possible_next_steps)
                     if (p[0] != p[1])]
    if valid_choices:
        for choice in valid_choices:
            next_routes = [extend_route(r, c) for r, c in zip(routes_so_far, choice)]
            possible_routes += day_16_get_all_valid_team_routes(next_routes)
        return possible_routes
    return [routes_so_far]


def day_16_get_all_valid_routes(route_so_far: [str]) -> [[str]]:
    """recursively find all routes that can turn on taps with non-zero flow
    rates that will be effective within the 30 minutes, without revisiting
    the origin, or taps that are already on"""
    # print(f"Looking at {route_so_far}")
    current_location = route_so_far[-1] if route_so_far else "AA"
    routes = day_16_route_table[current_location]
    options = [k for k, v in routes.items()
               if len(route_so_far) + v < 29
               and k not in route_so_far
               and k != "AA"]
    # print(f"Options are: {options}")
    if options:
        possible_routes = []
        for o in options:
            distance = day_16_route_table[current_location][o] - 1
            possible_routes += day_16_get_all_valid_routes(route_so_far +
                                                           [""] * distance + [o] * 2)
        return possible_routes
    return [route_so_far]


def day_16_build_distances_table(valve_data: dict) -> dict:
    table = defaultdict(defaultdict)
    useful_valves = ["AA"] + [*filter(lambda vd: valve_data[vd][0] > 0, valve_data)]
    # print(f"Useful valves are {useful_valves}")
    for i, v in enumerate(useful_valves):
        for other_valve in useful_valves[i + 1:]:
            distance = day_16_get_shortest_distance_between(v, other_valve, valve_data)
            table[v][other_valve] = distance
            table[other_valve][v] = distance
    return table


def day_16_get_shortest_distance_between(origin: str, destination: str,
                                         valves: dict) -> int:
    """uses Djikstra algorithm"""
    assert all(isinstance(loc, str) and len(loc) == 2 for loc in (origin, destination))
    all_valves = [*valves.keys()]
    distances = {valve: 1_000_000 for valve in all_valves}
    distances[origin] = 0
    while all_valves:
        this_point = min(all_valves, key=lambda v: distances[v])
        _, neighbours = valves[this_point]
        for np in neighbours:
            n_from_start = distances[this_point] + 1
            if n_from_start < distances[np]:
                distances[np] = n_from_start
        if this_point == destination:
            break
        all_valves.remove(this_point)
    # print(distances)
    return distances[destination]


def day_16_best_journey_using_permutations(valves: dict) -> int:
    global day_16_route_table
    day_16_route_table = day_16_build_distances_table(valves)
    useful_valves = [valve for valve, v_data in valves.items() if v_data[0] > 0]
    journeys = ["".join(p) for p in filter(lambda perm:
                                           day_16_journey_time(perm) + len(perm) < 30,
                                           itertools.permutations(useful_valves, 6))]
    print(f"There are {len(journeys)=}")
    # print(journeys)
    print(day_16_journey_time(journeys[1]))


def day_16_best_journey_by_taking_it_one_step_at_a_time(valves: dict):
    best_routes = {"no route": 0}
    useful_valves = [*filter(lambda vd: valves[vd][0] > 0, valves)]
    routes = [("AA",)]
    completed_routes = []
    for step in range(len(useful_valves)):
        # print(f"{step=:>2}, have to look at {len(routes):>8} routes")
        next_routes = []
        for route in routes:
            unvisited_valves = set(useful_valves) - set(route)
            for uv in unvisited_valves:
                new_route = tuple([*route] + [uv])
                if len(unvisited_valves) == 1 or min(day_16_route_table[uv][vv]
                                                     for vv in unvisited_valves
                                                     if vv != uv) + \
                        1 + day_16_journey_time(new_route) >= 30:
                    # if there are not other unvisited valves that
                    #    can be reached and opened in time
                    completed_routes.append(new_route)
                elif step % 2 or \
                        day_16_score_tuple_journey(new_route, valves) > max(best_routes.values()):
                    # only move forward with journeys that improve on the best
                    # we had two turns ago
                    next_routes.append(new_route)
        if step % 2 == 0:
            for r in routes:
                best_routes[r] = day_16_score_tuple_journey(r, valves)
        routes = next_routes
    return completed_routes


def day_16_score_tuple_journey(journey: (str,), valve_data: dict,
                               time_available: int = 30) -> int:
    total_flow, flow_start_time = 0, 0
    for i, location in enumerate(journey[1:]):
        flow_start_time += day_16_route_table[journey[i]][location] + 1
        flow_duration_minutes = time_available - flow_start_time
        flow_rate, _ = valve_data[location]
        total_flow += flow_rate * flow_duration_minutes
    return total_flow


def day_16_journey_time(journey: (str,)) -> int:
    """Time taken to make all steps of the journey FROM 'AA', including
        the time taken to turn each valve on"""
    if isinstance(journey, tuple):
        return sum(day_16_route_table[j][journey[i + 1]]
                   for i, j in enumerate(journey[:-1])) + len(journey) - 1


def day_16_score_double_headed_journey(journey: [[str]], valve_data: {}) -> int:
    return sum(day_16_score_journey(j, valve_data, time_available=26) for j in journey)


def day_16_score_double_headed_tuple_journey(journey: ((str,),), valve_data: {}) -> int:
    return sum(day_16_score_tuple_journey(j, valve_data, time_available=26) for j in journey)


def day_16_score_journey(journey: [str], valve_data: {},
                         time_available: int = 30) -> int:
    """a journey is represented as a sequence of locations visited.
    If one of these has a non-zero flow rate, it is repeated to represent
    the extra minute taken to turn on the tap.  Zero-flow locations can be
    represented as empty strings"""
    total_flow = 0
    for i, location in enumerate(journey):
        if i > 0 and location and location == journey[i - 1]:
            flow_duration_mins = time_available - (i + 1)
            flow_rate, _ = valve_data[location]
            total_flow += flow_rate * flow_duration_mins
    return total_flow


def day_16_load_valve_data(all_text: str) -> dict:
    def get_data(valve_text: str) -> (str, int):
        rate_text, _, options_text = valve_text.partition(";")
        valve_id = rate_text[6:8]
        rate = int(rate_text[rate_text.index("=") + 1:])
        options = [options_text[-2:]]
        if "," in options_text:
            options = options_text[options_text.index(",") - 2:].split(", ")
        return valve_id, tuple((rate, options))

    valve_data = {}
    for line in all_text.split("\n"):
        if line:
            valve, data = get_data(line)
            valve_data[valve] = data
    return valve_data


def day_15_part_one() -> int:
    return day_15_count_positions_without_sensor(Puzzle22(15).get_text_input(), 2_000_000)


def day_15_part_two() -> int:
    blind_spot = day_15_find_single_blind_spot(Puzzle22(15).get_text_input())
    return day_15_tuning_frequency(blind_spot)


def day_15_load_sensor_beacon_data(all_text: str) -> dict:
    data = {}
    rows = Puzzle22.convert_input(all_text, str)
    for r in rows:
        sensor, _, beacon = r.partition(": ")
        sensor_point, beacon_point = (eval(f"lib.Point({s[s.index('x'):]})")
                                      for s in (sensor, beacon))
        data[sensor_point] = beacon_point
    return data


def day_15_count_positions_without_sensor(text_input: str, row_id: int) -> int:
    positions = day_15_load_sensor_beacon_data(text_input)
    beacons_on_row = {v.x for v in positions.values() if v.y == row_id}
    known_empty_x = set()
    for sensor, nearest_beacon in positions.items():
        search_radius = lib.manhattan_distance(sensor, nearest_beacon)
        distance_to_row = abs(sensor.y - row_id)
        if distance_to_row <= search_radius:
            visible_x_on_row = [*range(sensor.x - search_radius + distance_to_row,
                                       sensor.x + search_radius - distance_to_row + 1)]
            known_empty_x.update(visible_x_on_row)
    return len(known_empty_x - beacons_on_row)


def day_15_find_single_blind_spot(all_text: str) -> lib.Point:
    from collections import Counter
    space = day_15_load_sensor_beacon_data(all_text)
    """It will be a point where there are at least four intersections
        of the lines just out of reach by the sensors"""
    all_intersections = []
    for ki, sensor in enumerate(space.keys()):
        for other_sensor in [*space.keys()][ki + 1:]:
            new_intersections = day_15_find_periphery_intersections(sensor,
                                                                    other_sensor, space)
            all_intersections += new_intersections
    print(f"There are {len(all_intersections)} intersections in total")
    winner = max(all_intersections, key=lambda i: all_intersections.count(i))
    print(f"{winner} has {all_intersections.count(winner)} intersections")

    ctr = Counter(all_intersections)
    print(f"Most common from Counter: {ctr.most_common(3)}")
    counter_winner = ctr.most_common(1)
    print(f"Counter winner: {counter_winner[0][0]}")
    if len(all_intersections) > 200:
        assert counter_winner[0][0] == lib.Point(x=527501.0, y=3570474.0)

    candidates = set(filter(lambda pt: all_intersections.count(pt) >= 3, all_intersections))
    print(f"Possible candidates: {candidates}")
    for c in candidates:
        if day_15_point_is_not_reached_by_any_sensor(space, c):
            print(f"returning a point that is not reachable: {c}")
            return c
    return max(all_intersections, key=lambda i: all_intersections.count(i))


def day_15_find_periphery_intersections(sensor_1: lib.Point, sensor_2: lib.Point,
                                        space: dict) -> [lib.Point]:
    """solve simultaneous equations for the just-unreachable lines of each of two
        points, to see at which points, if any, they intersect.  Solution could include
        a line of overlapping points if the line gradient is the same"""
    search_width = 4_000_000 if len(space) > 15 else 20
    crossings = []
    s1_params, s2_params = (day_15_get_gradients_and_intercepts(s, space[s])
                            for s in (sensor_1, sensor_2))
    visible_widths = {pt: lib.manhattan_distance(pt, space[pt])
                      for pt in (sensor_1, sensor_2)}
    smaller_sensor = min(visible_widths.keys(), key=lambda k: visible_widths[k])
    min_vis_width = visible_widths[smaller_sensor]
    for index, param in enumerate(s1_params):
        higher_intercept = index % 2
        min_x, max_x = (smaller_sensor.x for _ in range(2))
        g1, i1 = param
        if (g1 > 0 and higher_intercept) or (g1 < 0 and not higher_intercept):
            min_x = max_x - min_vis_width - 1
        if (g1 > 0 and not higher_intercept) or (g1 < 0 and higher_intercept):
            max_x = min_x + min_vis_width + 1
        assert min_x != max_x
        for s2_index, params in enumerate(s2_params):
            g2, i2 = params
            if g1 != g2:
                # assuming that an unreachable point has to be on
                # at least two perpendicular intersections
                """simultaneous equations to solve:
                        o-  y = g1x + i1
                        o-  y = g2x + i2
                        
                        (g1 -g2)x = i2 - i1
                        x = (i2 - i1) / (g1 - g2)"""
                x = (i2 - i1) / (g1 - g2)
                y = (g1 * x) + i1
                assert y == (g2 * x) + i2
                if 0 <= x <= search_width and 0 <= y <= search_width:
                    if min_x <= x <= max_x:
                        crossings.append(lib.Point(x, y))
    return crossings


def day_15_get_gradients_and_intercepts(sensor_location: lib.Point,
                                        nearest_beacon: lib.Point) -> [(int,)]:
    s_x, s_y = sensor_location
    beacon_distance = lib.manhattan_distance(sensor_location, nearest_beacon)
    return [(1, s_y - s_x - beacon_distance - 1), (1, s_y - s_x + beacon_distance + 1),
            (-1, s_x + s_y - beacon_distance - 1), (-1, s_x + s_y + beacon_distance + 1)]


def day_15_get_corners(sensor_location: lib.Point, nearest_beacon: lib.Point) -> [lib.Point]:
    beacon_distance = lib.manhattan_distance(sensor_location, nearest_beacon)
    s_x, s_y = sensor_location
    corners = [lib.Point(s_x + beacon_distance, s_y), lib.Point(s_x - beacon_distance, s_y),
               lib.Point(s_x, s_y + beacon_distance), lib.Point(s_x, s_y - beacon_distance)]
    return corners


def day_15_point_is_not_reached_by_any_sensor(space: dict, point: lib.Point) -> bool:
    for sensor in space:
        if lib.manhattan_distance(sensor, point) <= \
                lib.manhattan_distance(sensor, space[sensor]):
            return False
    return True


def day_15_tuning_frequency(point: lib.Point) -> int:
    return int((point.x * 4_000_000) + point.y)


day_14_infinite_floor_level = 0


def day_14_part_two() -> int:
    global day_14_infinite_floor_level
    blocked_points = day_14_load_all_points(Puzzle22(14).get_text_input())
    day_14_infinite_floor_level = max([pt.y for pt in blocked_points]) + 2
    units_retained = 0
    while lib.Point(500, 0) not in blocked_points:
        blocked_points = day_14_drop_particle_onto_infinite_floor(blocked_points)
        units_retained += 1
    return units_retained


def day_14_drop_particle_onto_infinite_floor(blocked_points: {lib.Point}) -> {lib.Point}:
    sand_position = lib.Point(500, 0)
    while sand_position.y < day_14_infinite_floor_level - 1:
        current_point = sand_position
        sand_position = day_14_drop_a_level_if_possible(current_point, blocked_points)
        if sand_position == current_point:
            blocked_points.add(sand_position)
            return blocked_points
    current_point = day_14_drop_a_level_if_possible(current_point, blocked_points)
    print(f"Current position is {current_point}")
    blocked_points.add(current_point)
    return blocked_points


def day_14_drop_a_level_if_possible(start: lib.Point, occupied: {lib.Point}) -> lib.Point:
    move_sequence = ("U", "UL", "UR")
    for move in move_sequence:
        desired_point = start
        for direction in move:
            desired_point = lib.point_moves[direction](desired_point)
        if desired_point not in occupied:
            return desired_point
    return start


def day_14_part_one() -> int:
    blocked_points = day_14_load_all_points(Puzzle22(14).get_text_input())
    size_of_rock = len(blocked_points)
    units_retained = 0
    while True:
        blocked_points = day_14_drop_sand_particle(blocked_points)
        if len(blocked_points) - units_retained == size_of_rock:
            break
        units_retained += 1
    return units_retained


def day_14_load_all_points(all_text: str) -> [lib.Point]:
    rock = [r for row in all_text.split("\n") if row for r in day_14_create_lines(row)]
    return set(rock)


def day_14_create_lines(input_row: str) -> [lib.Point]:
    defining_points = [lib.Point(*[int(n) for n in loc.split(",")])
                       for loc in input_row.split(" -> ")]
    line_points = [defining_points[0]]
    for index, pt in enumerate(defining_points[1:]):
        x_2, y_2 = pt
        x_1, y_1 = defining_points[index]
        if y_2 > y_1:
            line_points += [lib.Point(x_1, y_1 + n + 1) for n in range(y_2 - y_1)]
        elif y_2 < y_1:
            line_points += [lib.Point(x_1, y_1 - n - 1) for n in range(y_1 - y_2)]
        elif x_2 > x_1:
            line_points += [lib.Point(x_1 + n + 1, y_1) for n in range(x_2 - x_1)]
        elif x_2 < x_1:
            line_points += [lib.Point(x_1 - n - 1, y_1) for n in range(x_1 - x_2)]
    return line_points


def day_14_drop_sand_particle(occupied_points: {lib.Point}) -> {lib.Point}:
    """increasing y is downwards direction on screen, so use 'up' movement from library"""
    sand_position = lib.Point(500, 0)
    lowest_rock_level = max([pt.y for pt in occupied_points])
    while sand_position.y < lowest_rock_level:
        current_point = sand_position
        sand_position = day_14_drop_a_level_if_possible(current_point, occupied_points)
        if sand_position == current_point:
            occupied_points.add(sand_position)
            print(f"New point added is {sand_position}")
            break
    return occupied_points


def day_13_part_one() -> int:
    return day_13_get_sum_of_indices_of_correctly_ordered_pairs(Puzzle22(13).get_text_input())


def day_13_part_two() -> int:
    return day_13_insert_markers(Puzzle22(13).get_text_input())


def day_13_insert_markers(all_text: str) -> int:
    packets = []
    for p1, p2 in day_13_eval_load_pairs(all_text):
        packets += [p1, p2]
    total_packets = len(packets)
    packets_before_2 = len([*filter(lambda pr:
                                    day_13_order_is_correct(pr, [[2]]) == (True, True),
                                    packets)])
    packets_after_6 = len([*filter(lambda pr:
                                   day_13_order_is_correct([[6]], pr) == (True, True),
                                   packets)])
    return (packets_before_2 + 1) * (total_packets - packets_after_6 + 2)


def day_13_get_sum_of_indices_of_correctly_ordered_pairs(text_input: str) -> int:
    pairs = day_13_eval_load_pairs(text_input)

    def evaluate_pair_correctness(p) -> int:
        correctness = day_13_order_is_correct(*p)
        if correctness == (True, True):
            return 1
        return 0

    return sum((i + 1) * evaluate_pair_correctness(p) for i, p in enumerate(pairs))


def day_13_load_pairs(all_text: str) -> []:
    lines = all_text.split("\n")
    string_pairs = [tuple(day_13_comparable_list(lines[r]) for r in range(row_no, row_no + 2))
                    for row_no in range(0, len(lines), 3)]
    return string_pairs


def day_13_eval_load_pairs(all_text: str) -> []:
    lines = all_text.split("\n")
    return [tuple(eval(lines[r]) for r in range(row_no, row_no + 2))
            for row_no in range(0, len(lines), 3)]


def day_13_comparable_list(raw: str) -> []:
    if raw.startswith("[1,[2,"):
        print("")
    top_level_contents = raw[1:-1].split(",")
    if sum(len(tlc) for tlc in top_level_contents) == 0:
        return []
    if all(item.isnumeric() for item in top_level_contents):
        return [int(n) for n in top_level_contents]
    open_brackets = 0
    br_open, br_close = 0, 0
    for i, char in enumerate(raw[1:-1]):
        if char == "[":
            open_brackets += 1
            if open_brackets == 1:
                br_open = i + 1
        elif char == "]":
            open_brackets -= 1
            if open_brackets == 0:
                br_close = i + 1
                break

    return day_13_comparable_list(raw[1:br_open]) + [day_13_comparable_list(raw[br_open:br_close + 1])] +\
           day_13_comparable_list(raw[br_close + 1:])


def day_13_old_order_is_correct(left: object, right: object) -> bool:
    if all(isinstance(o, int) for o in (left, right)):
        if left == right:
            return None
        return left < right
    if isinstance(left, int):
        left = [left]
    if isinstance(right, int):
        right = [right]
    left_is_lower = True
    index = 0
    while index < len(left):
        if index >= len(right):
            return False
        left_is_lower = day_13_order_is_correct(left[index], right[index])
        if left_is_lower is not None:
            return left_is_lower
        index += 1
    if index < len(right):
        return True
    return left_is_lower


def day_13_order_is_correct(left: object, right: object) -> object:
    """return False if both sides match, otherwise (True, <result>)"""
    if all(isinstance(o, int) for o in (left, right)):
        if left == right:
            return False
        return True, left < right
    if isinstance(left, int):
        left = [left]
    if isinstance(right, int):
        right = [right]
    index = 0
    result = False
    while index < len(left):
        if index >= len(right):
            return True, False
        result = day_13_order_is_correct(left[index], right[index])
        if result:
            return result
        index += 1
    if index < len(right):
        return True, True
    return result


day_12_grid = [[]]


def day_12_part_one() -> int:
    day_12_start(Puzzle22(12).get_text_input())
    return day_12_dijkstra_shortest_distance(day_12_find_terminus())


def day_12_part_two() -> int:
    day_12_start(Puzzle22(12).get_text_input())
    first_col_as = [lib.Point(0, r) for r in range(len(day_12_grid))]
    candidates = [day_12_dijkstra_shortest_distance(pt) for pt in first_col_as]
    return min(candidates)


def day_12_start(text: str):
    global day_12_grid
    day_12_grid = day_12_build_grid(text)


def day_12_build_grid(text: str) -> [[str]]:
    return [[*r] for r in Puzzle22.convert_input(text, str)]


def day_12_find_terminus(start: bool = True) -> lib.Point:
    return day_12_find_all("S" if start else "E")[0]


def day_12_find_all(letter: str) -> [lib.Point]:
    locations = []
    termini = {"a": "S", "z": "E"}
    for y, row in enumerate(day_12_grid):
        for x, value in enumerate(row):
            if (value == letter) or (letter in termini and value == termini[letter]):
                locations.append(lib.Point(x, y))
    return locations


def day_12_get_spot_height(spot: lib.Point) -> int:
    """a = ord("a"), z = a + 25"""
    letter = day_12_grid[spot.y][spot.x]
    if letter in "SE":
        letter = "a" if letter == "S" else "z"
    return ord(letter)


def day_12_dijkstra_shortest_distance(start: lib.Point) -> int:
    all_points = [lib.Point(x, y) for y in range(len(day_12_grid))
                  for x in range(len(day_12_grid[0]))]
    distances = {pt: 1_000_000 for pt in all_points}
    end = day_12_find_terminus(False)
    distances[start] = 0
    while all_points:
        this_point = min(all_points, key=lambda p: distances[p])
        neighbours = day_12_get_valid_options(this_point)
        for np in neighbours:
            n_from_start = distances[this_point] + 1
            if n_from_start < distances[np]:
                distances[np] = n_from_start
        if this_point == end:
            break
        all_points.remove(this_point)
    return distances[end]


def day_12_get_valid_options(here: lib.Point) -> [lib.Point]:
    return [*filter(lambda loc: day_12_get_spot_height(loc) <=
                                day_12_get_spot_height(here) + 1,
                    lib.get_neighbours_in_grid(here, day_12_grid))]


day_11_monkeys = []


def day_11_part_one(text_input: str) -> int:
    day_11_initialise(text_input)
    for _ in range(20):
        day_11_play_round()
    return day_11_monkey_business()


def day_11_part_two(text_input: str) -> int:
    day_11_initialise(text_input)
    day_11_monkey_patch()
    for _ in range(10_000):
        day_11_play_round()
        if not _ % 100:
            print(f"{_:>6} rounds played")
            for m in day_11_monkeys:
                print(f"   {m.held_items}")
        if _ > 250:
            break
    return day_11_monkey_business()



def day_11_monkey_patch():
    global day_11_monkeys
    for monkey in day_11_monkeys:
        method = monkey.operation
        monkey.inspect_item = method


def day_11_initialise(all_notes: str):
    global day_11_monkeys
    day_11_monkeys = [Monkey(note) for note in all_notes.split("Monkey ")[1:]]


def day_11_play_round():
    for monkey in day_11_monkeys:
        monkey.take_turn()


def day_11_monkey_business() -> int:
    most_active = sorted([m.throw_count for m in day_11_monkeys], reverse=True)
    return most_active[0] * most_active[1]


class Monkey:
    def __init__(self, monkey_notes: str):
        lines = [ln for ln in monkey_notes.split("\n") if ln]
        self.held_items = [int(n) for n in lines[1][18:].split(", ")]
        _, _, op_text = lines[2].partition(" = ")
        _, operator, operand = (op_text.split(" "))
        self.operation = eval(f"lambda old: old {operator} {operand}")
        divisor, true_destination, false_destination = (int(line.split(" ")[-1])
                                                        for line in lines[-3:])
        self.test = lambda x: false_destination if x % divisor else true_destination
        self.throw_count = 0

    def take_turn(self):
        for item in self.held_items:
            recipient = self.throw_item(self.inspect_item(item))
            # print(f"item formerly of worry level {item} gets thrown to monkey {recipient}")
        self.held_items = []

    def inspect_item(self, worry_level: int) -> int:
        return self.operation(worry_level) // 3

    def throw_item(self, worry_level: int) -> int:
        recipient_monkey = self.test(worry_level)
        day_11_monkeys[recipient_monkey].catch_item(worry_level)
        self.throw_count += 1
        return recipient_monkey

    def catch_item(self, item: int):
        self.held_items.append(item)


def day_10_part_two() -> str:
    signals = day_10_value_after_cycle_completions(Puzzle22(10).get_text_input())
    display_text = day_10_render_image(signals)
    print(display_text)
    return display_text


def day_10_render_image(x_history: dict) -> str:
    image = ""
    for cycle in range(1, 241):
        if cycle == 10:
            print("say hi")
        new_pixel = "."
        if abs((cycle - 1) % 40 - day_10_find_value_during_cycle(cycle, x_history)) < 2:
            new_pixel = "#"
        image += new_pixel
        if not cycle % 40:
            image += "\n"
    return image


def day_10_part_one() -> int:
    return day_10_get_aggregate_signal_strength(Puzzle22(10).get_text_input())


def day_10_get_aggregate_signal_strength(text: str) -> int:
    cycles_of_interest = range(20, 240, 40)
    value_history = day_10_value_after_cycle_completions(text)
    return sum([c * day_10_find_value_during_cycle(c, value_history)
                for c in cycles_of_interest])


def day_10_value_after_cycle_completions(text: str) -> dict:
    completed_instructions, register_value = 0, 1
    status_by_cycle = {}
    for command in text.split("\n"):
        instruction, _, increment = command.partition(" ")
        completed_instructions += 1
        if instruction == "addx":
            completed_instructions += 1
            register_value += int(increment)
        status_by_cycle[completed_instructions] = register_value
    return status_by_cycle


def day_10_find_value_during_cycle(cycle_number: int, history: {}) -> int:
    if cycle_number <= min(history):
        return 1
    return history[max(filter(lambda k: k < cycle_number, history))]


day_9_points_touched_by_tail = set()


def day_9_part_one() -> int:
    return day_9_make_journey(Puzzle22(9).input_as_list(str))


def day_9_part_two() -> int:
    return day_9_make_journey_with_long_rope(Puzzle22(9).input_as_list(str))


def day_9_make_journey_with_long_rope(steps: [str]) -> int:
    return day_9_make_journey(steps, 10)


def day_9_make_journey(steps: [str], rope_length: int = 2) -> int:
    global day_9_points_touched_by_tail
    rope = tuple(lib.Point(0, 0) for _ in range(rope_length))
    day_9_points_touched_by_tail = {rope[-1]}
    for step in steps:
        rope = day_9_make_move(rope, step)
    # print(f"Points touched by tail: {day_9_points_touched_by_tail}")
    return len(day_9_points_touched_by_tail)


def day_9_make_move(rope: (lib.Point,), move: str) -> (lib.Point,):
    direction, _, distance = move.partition(" ")
    for _ in range(int(distance)):
        moved_rope = []
        new_head = day_9_move_knot_one_step(rope[0], direction)
        moved_rope.append(new_head)
        for knot in rope[1:]:
            new_head = day_9_follow_with_tail(new_head, knot)
            moved_rope.append(new_head)
        rope = tuple(moved_rope)
        day_9_points_touched_by_tail.add(rope[-1])
    # print(f"Rope configuration after {move}: {moved_rope}")
    return rope


def day_9_move_knot_one_step(current_location: lib.Point, direction: str) -> lib.Point:
    return lib.point_moves[direction](current_location)


def day_9_follow_with_tail(head: lib.Point, tail_origin: lib.Point) -> lib.Point:
    tail_position = tail_origin
    moves_needed = ""
    diffs = [tail - head for head, tail in zip(head, tail_origin)]
    if sum([abs(d) for d in diffs]) == 3:
        moves_needed += "L" if diffs[0] > 0 else "R"
        moves_needed += "D" if diffs[1] > 0 else "U"
    elif any([abs(d) == 2 for d in diffs]):
        if abs(diffs[0]) == 2:
            moves_needed += "L" if diffs[0] == 2 else "R"
        if abs(diffs[1]) == 2:
            moves_needed += "D" if diffs[1] == 2 else "U"
    for mv in moves_needed:
        tail_position = day_9_move_knot_one_step(tail_position, mv)
    return tail_position


def day_8_part_two(grid: [[int]]) -> int:
    grid_size = len(grid)
    return max([day_8_calculate_scenic_score(x, y, grid)
                for x in range(grid_size)
                for y in range(grid_size)])


def day_8_calculate_scenic_score(row_id: int, col_id: int, grid: [[int]]) -> int:
    grid_size = len(grid)
    if row_id == 0 or row_id == grid_size - 1 or col_id == 0 or col_id == grid_size - 1:
        return 0
    score = 1
    tree_height = grid[row_id][col_id]
    trees_to_left = grid[row_id][:col_id]
    trees_to_right = grid[row_id][col_id + 1:]
    trees_above = [grid[rw][col_id] for rw in range(row_id)]
    trees_below = [grid[rw][col_id] for rw in range(row_id + 1, grid_size)]
    for line_of_trees in (trees_to_left[::-1], trees_to_right, trees_above[::-1], trees_below):
        visible_trees = 0
        for intervening_tree in line_of_trees:
            visible_trees += 1
            if intervening_tree >= tree_height:
                break
        score *= visible_trees
    return score


def day_8_count_visible_trees(grid: [[int]]) -> int:
    grid_size = len(grid)

    def tree_is_visible(row_id: int, col_id: int) -> bool:
        if row_id == 0 or row_id == grid_size -1 or col_id == 0 or col_id == grid_size - 1:
            return True
        tree_height = grid[row_id][col_id]
        trees_to_left = grid[row_id][:col_id]
        trees_to_right = grid[row_id][col_id + 1:]
        trees_above = [grid[rw][col_id] for rw in range(row_id)]
        trees_below = [grid[rw][col_id] for rw in range(row_id + 1, grid_size)]
        if any([all([tr < tree_height for tr in intervening_trees])
                for intervening_trees in (trees_to_left, trees_to_right,
                                          trees_above, trees_below)]):
            return True
        return False
    return sum([sum([tree_is_visible(x, y) for x in range(grid_size)])
                for y in range(grid_size)])


def day_8_make_grid(input_text: str) -> [[int]]:
    rows = Puzzle22.convert_input(input_text, None)
    return [[int(tree) for tree in row] for row in rows]


def day_7_part_one() -> int:
    all_text = Puzzle22(7).get_text_input()
    structure = day_7_build_directory_structure(all_text)
    return day_7_get_total_size_of_small_directories(structure)


def day_7_part_two() -> int:
    all_text = Puzzle22(7).get_text_input()
    structure = day_7_build_directory_structure(all_text)
    return day_7_get_smallest_directory_to_delete(structure)


def day_7_get_smallest_directory_to_delete(structure: [int]) -> int:
    global day_7_all_directory_sizes
    day_7_all_directory_sizes = []
    space_requirement = day_7_space_needed_to_be_freed(structure)
    day_7_directory_total(structure)
    return min(filter(lambda size: size >= space_requirement, day_7_all_directory_sizes))


def day_7_space_needed_to_be_freed(structure: [int]) -> int:
    return 30_000_000 - (70_000_000 - day_7_directory_total(structure))


day_7_total_size_of_small_directories = 0
day_7_all_directory_sizes = []


def day_7_directory_total(directory: [int]) -> int:
    global day_7_total_size_of_small_directories, day_7_all_directory_sizes
    dir_total = 0
    for file in directory:
        if isinstance(file, int):
            dir_total += file
        else:
            dir_total += day_7_directory_total(file)
    if dir_total <= 100000:
        day_7_total_size_of_small_directories += dir_total
    day_7_all_directory_sizes.append(dir_total)
    return dir_total


def day_7_get_total_size_of_small_directories(structure: [int]) -> int:
    global day_7_total_size_of_small_directories
    day_7_total_size_of_small_directories = 0
    day_7_directory_total(structure)
    return day_7_total_size_of_small_directories


def day_7_build_directory_structure(text: str) -> []:
    def list_current_directory() -> []:
        line, local_tree = "", []
        while True:
            try:
                line = next(text_lines)
                if line.startswith("$ cd "):
                    destination = line.split(" ")[-1]
                    if destination == "..":
                        return local_tree
                    else:
                        local_tree.append(list_current_directory())
                elif len(line) and (not line.startswith("$")):
                    if line[0].isnumeric():
                        local_tree.append(int(line.split(" ")[0]))
            except StopIteration:
                break
        return local_tree

    text_lines = (ln for ln in text.split("\n"))
    tree = list_current_directory()[0]
    return tree


def day_6_part_one() -> int:
    return day_6_get_marker_end(Puzzle22(6).get_text_input())


def day_6_part_two() -> int:
    return day_6_get_message_start(Puzzle22(6).get_text_input())


def day_6_get_marker_end(buffer: str) -> int:
    return day_6_get_position_of_unique_n_chars(buffer, 4)


def day_6_get_message_start(buffer: str) -> int:
    return day_6_get_position_of_unique_n_chars(buffer, 14)


def day_6_get_position_of_unique_n_chars(buffer: str, n: int) -> int:
    for string_index, _ in enumerate(buffer):
        sub_string = buffer[string_index:string_index + n]
        if len(set(sub_string)) == len(sub_string):
            return string_index + n
    return 0


def day_5_part_two(raw_text: str) -> str:
    return day_5_part_one(raw_text, True)


def day_5_part_one(raw_text: str, part_two_move_style: bool = False) -> str:
    stacks = day_5_get_starting_configuration(raw_text)
    moves = filter(lambda t: t.startswith("move"), raw_text.split("\n"))
    for mv in moves:
        stacks = day_5_make_move(day_5_interpret_move(mv), stacks, part_two_move_style)
    return "".join([v[-1] for v in stacks.values()])


def day_5_get_starting_configuration(all_text: str) -> dict:
    relevant_lines = filter(lambda line: len(line) and (not line.startswith("move")),
                            all_text.split("\n"))
    items = [[ln[n] for n in range(1, len(ln), 4)] for ln in relevant_lines]
    return {int(stack_id): "".join([items[n][index]
                               for n in range(len(items) - 2, -1, -1)]).strip()
            for index, stack_id in enumerate(items[-1])}


def day_5_make_move(move: (int,), configuration: dict,
                    part_two_move: bool = False) -> dict:
    quantity, origin, destination = move
    original_stack = configuration[origin]
    if part_two_move:
        moved_boxes = original_stack[len(original_stack) - quantity:]
    else:
        moved_boxes = original_stack[::-1][:quantity]
    configuration[origin] = configuration[origin][:len(original_stack) - quantity]
    configuration[destination] = configuration[destination] + moved_boxes
    return configuration


def day_5_interpret_move(move_text: str) -> (int,):
    words = move_text.split(" ")
    return tuple([int(number) for number in [words[n] for n in range(1, 6, 2)]])


def day_4_split_function(text: str) -> ((int,),):
    return tuple(tuple(int(n) for n in t.split("-")) for t in text.split(","))


def day_4_part_one(all_pairings: [((int,),)]) -> int:
    return sum(day_4_one_wholly_contains_other(p) for p in all_pairings)


def day_4_one_wholly_contains_other(pairing: ((int,),)) -> bool:
    duties_1, duties_2 = pairing
    start_1, end_1 = duties_1
    start_2, end_2 = duties_2
    if start_1 <= start_2:
        if end_2 <= end_1:
            return True
    if start_2 <= start_1:
        if end_1 <= end_2:
            return True
    return False


def day_4_part_two(all_pairings: [((int,),)]) -> int:
    return sum(day_4_any_overlap(p) for p in all_pairings)


def day_4_any_overlap(pairing: ((int,),)) -> bool:
    duties_1, duties_2 = pairing
    start_1, end_1 = duties_1
    start_2, end_2 = duties_2
    if start_1 <= start_2:
        return end_1 >= start_2
    if start_2 <= start_1:
        return end_2 >= start_1
    return False


def day_3_part_one(rucksacks: [str]) -> int:
    return sum(day_3_get_priority_for_rucksack(r) for r in rucksacks)


def day_3_get_priority_for_rucksack(rucksack: str) -> int:
    assert len(rucksack) % 2 == 0
    halfway = len(rucksack) // 2
    compartment_1, compartment_2 = rucksack[:halfway], rucksack[halfway:]
    for item in compartment_1:
        if item in compartment_2:
            return day_3_calculate_priority(item)
    assert False
    return 0


def day_3_part_two(all_rucksacks: [str]) -> int:
    return sum(day_3_get_priority_for_group(all_rucksacks[index:index + 3])
               for index in range(0, len(all_rucksacks), 3))


def day_3_get_priority_for_group(group_of_rucksacks: [str]) -> int:
    common_letters = set(group_of_rucksacks[0]).intersection(group_of_rucksacks[1])
    common_letters = common_letters.intersection(group_of_rucksacks[2])
    assert len(common_letters) == 1
    return day_3_calculate_priority(list(common_letters)[0])


def day_3_calculate_priority(letter: str) -> int:
    priority = ord(letter) - 38
    if priority > 58:
        priority -= 58
    return priority


def day_2_part_one(round_list: [str]) -> int:
    return sum(day_2_score_round(rnd) for rnd in round_list)


def day_2_part_two(round_list: [str]) -> int:
    modified_rounds = []
    for r in round_list:
        their_move, _, outcome_letter = r.partition(" ")
        desired_points = (ord(outcome_letter) - 88) * 3
        response_letter = [*filter(lambda c:
                                 day_2_get_outcome(their_move, c) == desired_points,
                                   "XYZ")][0]
        modified_rounds.append(f"{their_move} {response_letter}")
    return day_2_part_one(modified_rounds)


def day_2_score_round(moves: str) -> int:
    opponent_move, _, my_move = moves.partition(" ")
    shape_component = {chr(n + 88): n + 1 for n in range(3)}
    return shape_component[my_move] + day_2_get_outcome(opponent_move, my_move)


def day_2_get_outcome(their_shape: str, my_shape: str) -> int:
    def shape_id(letter: str) -> int:
        single_digit = ord(letter) - 64
        if single_digit > 23:
            single_digit -= 23
        return single_digit

    opponent, me = (shape_id(char) for char in (their_shape, my_shape))
    if opponent == me:
        return 3
    if (me, opponent) == (3, 1):
        return 0
    if me > opponent or (me, opponent) == (1, 3):
        return 6
    return 0



def day_1_get_list_of_totals() -> [int]:
    puzzle = Puzzle22(1)
    strings = puzzle.convert_input(puzzle.get_text_input(), blank_lines_matter=True)
    totals = []
    current_total = 0
    for s in strings:
        if not s:
            totals.append(current_total)
            current_total = 0
        else:
            current_total += int(s)
    return totals


def day_1_part_one() -> int:
    return max(day_1_get_list_of_totals())


def day_1_part_two() -> int:
    sorted_list = sorted(day_1_get_list_of_totals(), reverse=True)
    return sum(sorted_list[:3])

# def initialise_puzzle(day: int) -> pz:
#     puzzle = pz(day)
#
#     def get_text_input(self):
#         with open(f'inputs_2022\\input{self.day}.txt', 'r') as input_file:
#             return input_file.read()
#
#     puzzle.get_text_input = get_text_input
#     return puzzle
#
#
