from main import Puzzle
import library as lib


class Puzzle22(Puzzle):
    def get_text_input(self) -> str:
        with open(f'inputs_2022\\input{self.day}.txt', 'r') as input_file:
            return input_file.read()


def day_15_part_one() -> int:
    return day_15_count_positions_without_sensor(Puzzle22(15).get_text_input(), 2_000_000)


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
        row_intersection = lib.Point(sensor.x, row_id)
        distance_to_row = lib.manhattan_distance(sensor, row_intersection)
        if distance_to_row <= search_radius:
            visible_x_on_row = [*range(sensor.x - search_radius + distance_to_row,
                                       sensor.x + search_radius - distance_to_row + 1)]
            known_empty_x.update(visible_x_on_row)
    return len(known_empty_x - beacons_on_row)


def day_15_find_single_blind_spot(all_text: str) -> lib.Point:
    space = day_15_load_sensor_beacon_data(all_text)
    """Think it will be a point where there are at least four corners or intersections
        within one or two manhattan distance, surrounding it"""
    return lib.Point(0, 0)


def day_15_get_gradients_and_intercepts(sensor_location: lib.Point,
                                        nearest_beacon: lib.Point) -> [(int,)]:
    s_x, s_y = sensor_location
    beacon_distance = lib.manhattan_distance(sensor_location, nearest_beacon)
    return [(1, s_y - s_x - beacon_distance), (1, s_y - s_x + beacon_distance),
            (-1, s_x + s_y - beacon_distance), (-1, s_x + s_y + beacon_distance)]


def day_15_get_all_just_unreachable_points(sensor_location: lib.Point,
                                           nearest_beacon: lib.Point) -> [lib.Point]:
    beacon_distance = lib.manhattan_distance(sensor_location, nearest_beacon)
    s_x, s_y = sensor_location
    return [lib.Point(s_x - beacon_distance - 1 + x, s_y + x)
            for x in range(beacon_distance + 1)] + \
           [lib.Point(s_x + x + 1, s_y - beacon_distance + x)
            for x in range(beacon_distance + 1)] + \
           [lib.Point(s_x - beacon_distance + x, s_y - 1 - x)
            for x in range(beacon_distance + 1)] + \
           [lib.Point(s_x + x, s_y + beacon_distance + 1 - x)
            for x in range(beacon_distance + 1)]


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
        next_point = min(all_points, key=lambda p: distances[p])
        neighbours = day_12_get_valid_options(next_point)
        for np in neighbours:
            n_from_start = distances[next_point] + 1
            if n_from_start < distances[np]:
                distances[np] = n_from_start
        if next_point == end:
            break
        all_points.remove(next_point)
    return distances[end]


def day_12_get_valid_options(here: lib.Point) -> [lib.Point]:
    not_too_high = lambda destination: day_12_get_spot_height(destination) <=\
                                       day_12_get_spot_height(here) + 1
    return [*filter(lambda loc: not_too_high(loc),
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
