from collections import namedtuple
from itertools import product, permutations, combinations, cycle
import random
from math import prod as product


class Puzzle:
    def __init__(self, day: int):
        self.day = day

    def get_text_input(self) -> str:
        with open(f'inputs\\input{self.day}.txt', 'r') as input_file:
            return input_file.read()

    def get_text_from_filename(self, txt_file_name: str) -> str:
        with open(f'inputs\\{txt_file_name}.txt', 'r') as input_file:
            return input_file.read()

    def input_as_list(self, conversion_func: object = int) -> []:
        return self.convert_input(self.get_text_input(), conversion_func=conversion_func)

    @staticmethod
    def convert_input(raw_input: str, conversion_func: object = int,
                      blank_lines_matter: bool = False) -> []:
        """works for lists of int, both as comma-separated or multi-line list"""
        split_on = '\n'
        if raw_input.count('\n') <= 1:
            raw_input = raw_input.strip('\n')
            split_on = ","
        raw_strings = raw_input.split(split_on)
        if not blank_lines_matter:
            raw_strings = list(filter(bool, raw_strings))
        if conversion_func == int and all([d.isnumeric() for d in raw_strings]):
            return [int(d) for d in raw_strings]
        return raw_strings


def binary_to_int(binary: []) -> int:
    if isinstance(binary, str):
        binary = [int(b) for b in binary]
    return sum([bool(b) * 2 ** i for i, b in enumerate(binary[::-1])])


def invert_binary(binary: []) -> [bool]:
    return [not b for b in binary]


# TODO: look at Barry's code
# dataclasses
day_24_add_to_x = [13, 15, 15, 11, -7, 10, 10, -5, 15, -3, 0, -5, -9, 0]
day_24_add_to_y = [6, 7, 10, 2, 15, 8, 1, 10, 5, 3, 5, 11, 12, 10]


class Day24StepperBacker:
    def __init__(self, models_and_zs: [tuple]):
        self.models_and_zs = models_and_zs

    def step_back(self, step_no: int) -> object:
        new_ms_and_zs = []
        for model, z_gen in self.models_and_zs:
            """do reverse step for each model/z combination"""
        return Day24StepperBacker(new_ms_and_zs)


def day_24_part_one():
    sb = Day24StepperBacker([])
    for step in range(13, -1, -1):
        sb = sb.step_back(step)
    return max(filter(lambda mz: next(mz[1]) == 0, sb.models_and_zs[0]),
               key=lambda mzs: int(mzs[0]))


def day_24_run_program(lines: [str], inputs: (int,),
                       initial_z: int=0) -> {str: int}:
    if isinstance(inputs, str):
        inputs = (int(num) for num in inputs)
    inputs = (ip for ip in inputs)
    ord_w = ord("w")
    memory_state = {chr(o): 0 for o in range(ord_w, ord_w + 4)}
    memory_state["z"] = initial_z
    for line in lines:
        input_value = None
        if line[:3] == "inp":
            input_value = next(inputs)
        memory_state = day_24_process_one_line(memory_state, line, input_value)
    return memory_state


def day_24_process_one_line(state: {str: int}, line: str,
                            input_value = None) -> {str: int}:
    instruction = line[:3]
    target_var = line[4]
    argument = line[6:]

    def inp(var: str, arg: int):
        state[var] = arg

    def add(var: str, arg: int):
        state[var] += arg

    def mul(var: str, arg: int):
        state[var] *= arg

    def div(var: str, arg: int):
        state[var] //= arg
        state[var] += (1 if state[var] < 0 else 0)
        
    def mod(var: str, arg: int):
        state[var] %= arg

    def eql(var: str, arg: int):
        state[var] = 1 if arg == state[var] else 0

    if input_value or input_value == 0:
        argument = input_value
    elif any([a.isnumeric() for a in argument]):
        argument = int(argument)
    else:
        argument = state[argument]
    eval(f'{instruction}')(target_var, argument)
    # print(f'{line:<10}{state}')
    return state


def day_23_load_data(raw_text: str) -> (str, [str]):
    rows = Puzzle.convert_input(raw_text, None)
    initial_hallway = "." * 11
    initial_rooms = ["" for _ in range(4)]
    for r in rows:
        letters = "".join(filter(lambda ch: ch.isalpha(), r))
        if letters:
            initial_rooms = [ltr + initial_rooms[i] for i, ltr in enumerate(letters)]
    return Configuration(initial_hallway, initial_rooms)


day_23_minimum_energy = (2 ** 31) - 1
day_23_rejection_threshold = 100000


def day_23_part_two(raw_text: str) -> int:
    return day_23_part_one(raw_text, room_size=4)


def day_23_part_one(raw_string: str, room_size: int = 2) -> int:
    global day_23_minimum_energy
    loops = 0
    all_configs = [Day23Config(day_23_load_data(raw_string), room_size=room_size)]
    while all_configs and any([not cfg.completed for cfg in all_configs]):
        loops += 1
        print(f'({loops}) Looking at {len(all_configs)} configuration{"" if len(all_configs) == 1 else "s"}')
        tuplised_set = set([cfg.tuple_ise() for cfg in all_configs])
        print(f'Length of tuplised set: {len(tuplised_set)}')
        energy_totals = [cfg.energy_usage for cfg in all_configs if cfg.completed]
        if energy_totals:
            day_23_minimum_energy = min(day_23_minimum_energy, min(energy_totals))
            print(f'Minimum energy usage is now {day_23_minimum_energy}')
        all_configs = [Day23Config.create_from_tuple(tpl, room_size=room_size)
                       for tpl in tuplised_set]
        next_configs = []
        for cfg in all_configs:
            next_configs += cfg.generate_next_configs()
        all_configs = next_configs
    return day_23_minimum_energy


class Day23Config:
    def __init__(self, config: (str, [str]), energy_usage: int = 0,
                 room_size: int = 2):
        self._config = config
        self.energy_usage = energy_usage
        self.completed = self.stalled = False
        self.room_size = room_size

    def generate_next_configs(self) -> [(str, [str])]:
        if day_23_is_completed(self._config, self.room_size):
            self.completed = True
            return [self]
        next_moves = day_23_get_next_valid_moves(self._config, self.room_size)
        if len(next_moves) == 0 \
                or self.energy_usage > min(day_23_rejection_threshold,
                                           day_23_minimum_energy):
            # think it takes at least 5 moves to get to this
            self.stalled = True
            return []
        next_configs = [Day23Config(day_23_make_move(self._config, move),
                                    self.energy_usage, self.room_size)
                        for move in next_moves]
        for config, move in zip(next_configs, next_moves):
            config.energy_usage += day_23_get_energy_usage(move, config._config,
                                                           self.room_size)
        return next_configs

    def tuple_ise(self) -> (str, int):
        hallway, rooms = self._config
        config_string = f'{hallway}' \
                        f'{"".join([rm.ljust(self.room_size, " ") for rm in rooms])}'
        return config_string, self.energy_usage

    @staticmethod
    def create_from_tuple(config_tuple: (str, int), room_size: int = 2) -> object:
        raw_string, energy = config_tuple
        hallway = raw_string[:11]
        rooms = []
        for i, raw in enumerate(raw_string[11::room_size]):
            # room_chars = raw + raw_string[11 + (i * room_size) + 1]
            start_slice = 11 + (i * room_size)
            room_chars = raw_string[start_slice:start_slice + room_size]
            room_occupants = "".join([ch for ch in room_chars if ch.isalpha()])
            rooms.append(room_occupants)
        return Day23Config(Configuration(hallway, rooms), energy, room_size)


def day_23_get_next_valid_moves(config: (str, [str]),
                                room_size: int = 2) -> [(int,)]:
    def get_room(column_id: int) -> str:
        return rooms[(column_id // 2) - 1]

    def clear_passage(start_ind: int, dest_ind: int) -> bool:
        passageway = hallway[start_ind + 1:dest_ind + 1] \
            if start_ind < dest_ind else hallway[dest_ind:start_ind]
        return all([ch == "." for ch in passageway])

    moves = []
    hallway, rooms = config
    occupied_columns = [i for i, ch in enumerate(hallway) if ch in "ABCD"]
    occupied_columns += [2 + (i * 2) for i, occupants in
                         enumerate(rooms) if occupants]
    for origin in occupied_columns:
        for destination in range(11):
            if destination == origin:
                continue
            if day_23_is_room(destination):
                letter = get_room(origin)[-1] \
                    if day_23_is_room(origin) else hallway[origin]
                only_valid_destination = ("ABCD".index(letter) + 1) * 2
                if destination == only_valid_destination and \
                        len(get_room(destination)) < room_size and \
                        (not any([ch != letter for ch in get_room(destination)])) and \
                        clear_passage(origin, destination):
                    moves.append(Move(origin, destination))
            elif day_23_is_room(origin) and clear_passage(origin, destination):
                # only move out of room if it still contains some non-native letters:
                if any([ch != "ABCD"[(origin // 2) - 1] for ch in get_room(origin)]):
                    moves.append(Move(origin, destination))
    return moves


def day_23_make_move(existing_config: (str, [str]), move: (int,)) -> (str, [str]):
    """No validation, just move"""
    hallway, rooms = existing_config
    rooms = [[*r] for r in rooms]
    orig, dest = move
    if day_23_is_room(orig):
        if day_23_is_room(dest):
            rooms[(dest // 2) - 1].append(rooms[(orig // 2) - 1].pop())
        else:
            hallway = hallway[:dest] + rooms[(orig // 2) - 1].pop() + hallway[dest + 1:]
    else:
        letter = hallway[orig]
        hallway = hallway[:orig] + "." + hallway[orig + 1:]
        rooms[(dest // 2) - 1].append(letter)
    return Configuration(hallway, ["".join(rm) for rm in rooms])


def day_23_get_energy_usage(move: (int,), final_config: (str, [str]),
                            room_size: int = 2) -> int:
    def get_room(column_id: int) -> str:
        return rooms[(column_id // 2) - 1]

    orig_col, dest_col = move
    hallway, rooms = final_config
    letter = get_room(dest_col)[-1] if day_23_is_room(dest_col) \
        else hallway[dest_col]
    rate = {ltr: 10 ** power for ltr, power in zip("ABCD", range(4))}[letter]
    steps = abs(dest_col - orig_col)
    if day_23_is_room(orig_col):
        steps += 1
        if len(get_room(orig_col)) < room_size - 1:
            steps += room_size - len(get_room(orig_col)) - 1
    if day_23_is_room(dest_col):
        steps += 1
        if len(get_room(dest_col)) < room_size:
            steps += room_size - len(get_room(dest_col))
    return rate * steps


def day_23_is_room(location_index: int) -> bool:
    return location_index in range(2, 9, 2)


def day_23_is_completed(configuration: (str, [str]), room_size: int = 2) -> bool:
    return configuration.rooms == [ltr * room_size for ltr in "ABCD"] \
           and configuration.hallway == "." * 11


def day_23_insert_additional_rows(initial_input: str, extra_text: str) -> str:
    lf = '\n'
    if extra_text[-1] != lf:
        extra_text += lf
    lines = initial_input.split(lf)
    return f'{lf.join(lines[:3])}{lf}{extra_text}{lf.join(lines[3:])}'


Configuration = namedtuple("Configuration", "hallway rooms")
Move = namedtuple("Move", "from_col to_col")


def day_22_load_data(raw_input: str, all_space: bool = False) -> [(str, [int])]:
    raw_lines = Puzzle.convert_input(raw_input, None)
    converted_data = []
    for line in raw_lines:
        on_or_off, raw_dims = tuple(line.split(" "))
        dims = day_22_get_cuboid_dimensions(raw_dims)
        if all_space or all([-50 <= d <= 50 for d in dims]):
            converted_data.append((on_or_off, dims))
    return converted_data


def day_22_part_one(raw_input: str) -> int:
    data = day_22_load_data(raw_input)
    active_points = set()
    for step in data:
        action, dims = step
        if action == "on":
            print('turning on:')
            active_points = active_points.union(day_22_create_cuboid(dims))
        else:
            print('turning off:')
            active_points = active_points.difference(day_22_create_cuboid(dims))
    return len(active_points)


def day_22_create_cuboid(dimensions: [int]) -> {object}:
    x_min, x_max, y_min, y_max, z_min, z_max = tuple(dimensions)
    cuboid = set()
    for z in range(z_min, z_max + 1):
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                cuboid.add(Point3(x, y, z))
    return cuboid


def day_22_get_cuboid_dimensions(cuboid_dims: str) -> [int]:
    numbers = []
    for dimension in cuboid_dims.split(","):
        _, _, bounds = dimension.partition("=")
        start, _, end = bounds.partition("..")
        numbers += [int(dim) for dim in (start, end)]
    return numbers


def day_22_part_two(raw_text: str) -> int:
    data = day_22_load_data(raw_text, all_space=True)
    return day_22_solve_part_two(data)


def day_22_solve_part_two(data: [(str, [int])]) -> int:
    total_on = 0
    for outer_index, data_row in enumerate(data):
        state, this_cuboid = data_row
        new_volume = day_22_get_cuboid_size(this_cuboid)
        if state == 'on':
            total_on += new_volume
        for inner_index, inner_row in enumerate(data[:outer_index]):
            inner_state, inner_dim = inner_row
            if inner_state == 'on':
                overlap_volume = day_22_calc_overlap_size(this_cuboid, inner_dim)

            total_on -= overlap_volume
    return total_on


def day_22_try_for_part_two(data: [(str, [int])]) -> int:
    no_of_cuboids = len(data)
    if no_of_cuboids == 4:
        print('hi')
    this_cuboid_state, this_cuboid_dims = data[-1]
    if no_of_cuboids == 1:
        # ASSUMING the first cuboid is never 'off'
        return day_22_get_cuboid_size(this_cuboid_dims)
    non_overlapping_volume = day_22_get_cuboid_size(this_cuboid_dims) \
        if this_cuboid_state == "on" else 0
    for simultaneous_overlaps in range(1, no_of_cuboids):
        sign = -1 if simultaneous_overlaps % 2 else 1
        combs = combinations(range(no_of_cuboids - 1), simultaneous_overlaps)
        for c in combs:
            factor = sign
            other_cuboids = [data[n] for n in c]
            cuboid_combo = other_cuboids + [data[-1]]
            overlap_dims = day_22_get_overlap_of_multiple_cuboids(cuboid_combo)
            if other_cuboids[-1][0] == "off" and this_cuboid_state == "on":
                factor = 1
                if len(cuboid_combo) > 2:
                    factor = 0
            if no_of_cuboids == 4:
                if len(overlap_dims) == 6:
                    print(f'Combo: {c} adds {factor * day_22_get_cuboid_size(overlap_dims)}')
            non_overlapping_volume += factor * day_22_get_cuboid_size(overlap_dims)
    non_overlapping_volume += day_22_try_for_part_two(data[:-1])
    return non_overlapping_volume


def day_22_get_cuboid_size(dims: [int]) -> int:
    sides = [dims[(i * 2) + 1] - d + 1 for i, d in enumerate(dims[::2])]
    if any([side_length < 0 for side_length in sides]):
        return 0
    return product(sides)


def day_22_calc_overlap_size(cuboid_1: [int], cuboid_2: [int]) -> int:
    return day_22_get_cuboid_size(day_22_get_overlap_cuboid_dims(cuboid_1, cuboid_2))


def day_22_get_overlap_of_multiple_cuboids(cuboids: [(str, [int])]) -> [int]:
    overlap = cuboids[0][1]
    for cbd in cuboids[1:]:
        overlap = day_22_get_overlap_cuboid_dims(overlap, cbd[1])
        if day_22_get_cuboid_size(overlap) == 0:
            return [0, -1]
    return overlap


def day_22_get_overlap_cuboid_dims(cuboid_1: [int], cuboid_2: [int]) -> [int]:
    zipped = zip(cuboid_1, cuboid_2)
    return [(min(zc) if i % 2 else max(zc)) for i, zc in enumerate(zipped)]


def day_21_load_data(raw_text: str) -> (int,):
    return (int(line[-1]) for line in raw_text.split('\n'))


def day_21_part_two(raw_text: str) -> int:
    p1_univs_per_turns, p2_univs_per_turns = (day_21_universes_by_no_of_turns(pp)
                                              for pp in day_21_load_data(raw_text))
    p1_routes, p2_routes = (day_21_find_all_routes_to_21(p)
                            for p in day_21_load_data(raw_text))
    combinations_dict = day_21_get_number_of_combinations_per_total()
    p1_wins = 0
    for no_of_turns, universes in p1_univs_per_turns.items():
        valid_p2_routes = filter(lambda rt: len(rt) >= no_of_turns, p2_routes)
        short_p2_routes = set([tuple(route[:no_of_turns - 1]) for route in valid_p2_routes])
        p1_wins += universes * sum([day_21_calc_universes_for_route(rt, combinations_dict,
                                                                    limit=no_of_turns - 1)
                                    for rt in short_p2_routes])

    p2_wins = 0
    for no_of_turns, universes in p2_univs_per_turns.items():
        valid_p1_routes = filter(lambda rt: len(rt) > no_of_turns, p1_routes)
        short_p1_routes = set([tuple(route[:no_of_turns]) for route in valid_p1_routes])
        p2_wins += universes * sum([day_21_calc_universes_for_route(rt, combinations_dict,
                                                                    limit=no_of_turns)
                                    for rt in short_p1_routes])
    return max(p1_wins, p2_wins)


def day_21_universes_by_no_of_turns(start_pos: int) -> dict:
    combinations_dict = day_21_get_number_of_combinations_per_total()
    universes_by_no_of_turns = {}
    all_routes = day_21_find_all_routes_to_21(start_pos)
    for route in all_routes:
        length = len(route)
        universes = day_21_calc_universes_for_route(route, combinations_dict)
        if length in universes_by_no_of_turns:
            universes_by_no_of_turns[length] += universes
        else:
            universes_by_no_of_turns[length] = universes
    return universes_by_no_of_turns


def day_21_calc_universes_for_route(route: [int], comb_dict: dict,
                                    limit: int = 20) -> int:
    total = comb_dict[route[0]]
    for dice_total in route[1:limit]:
        total *= comb_dict[dice_total]
    return total


def day_21_find_all_routes_to_21(starting_position: int) -> [[int]]:
    winning_rolls = []
    sequences_under_consideration = [[total] for total in range(3, 10)]
    while sequences_under_consideration:
        next_round = []
        for seq in sequences_under_consideration:
            for tot in range(3, 10):
                seq_with_next_roll = seq + [tot]
                if day_21_score_roll_sequence(seq_with_next_roll, starting_position) >= 21:
                    winning_rolls.append(seq_with_next_roll)
                else:
                    next_round.append(seq_with_next_roll)
        sequences_under_consideration = next_round
    return winning_rolls


def day_21_score_roll_sequence(sequence: [int], start_pos: int) -> int:
    score = 0
    position = start_pos
    for roll in sequence:
        position = (position + roll) % 10
        if position == 0:
            position = 10
        score += position
    return score


def day_21_get_number_of_combinations_per_total() -> {int: int}:
    combo_counts = {k: 0 for k in range(3, 10)}
    dice = (range(1, 4) for _ in range(3))
    three_two_one = product(*dice)
    for combo in three_two_one:
        combo_counts[sum(combo)] += 1
    return combo_counts


def day_21_part_one(starting_spaces: (int,)):
    def scores() -> (int,):
        return (scr for _, scr in data.values())

    n_rolls = 0
    die = cycle([*range(1, 101)])
    player_id = cycle([0, 1])
    positions = tuple(starting_spaces)
    data = {p_id: (st, 0) for p_id, st in zip((0, 1), positions)}
    while max(scores()) < 1000:
        pl_index = next(player_id)
        position, score = data[pl_index]
        position = (position + sum([next(die) for _ in range(3)])) % 10
        n_rolls += 3
        if position == 0:
            position = 10
        score += position
        data[pl_index] = position, score
    losing_score = min(scores())
    return losing_score * n_rolls


def day_20_part_one(raw_lines: [str]) -> int:
    algorithm, image = day_20_load_data(raw_lines)
    for n in range(2):
        image = day_20_process_image(image, algorithm,
                                     '#' if n % 2 and len(image) > 10 else '.')
        print('         Image Starts:')
        print('\n'.join(image))
    return sum([ln.count('#') for ln in image])


def day_20_part_two(raw_lines: [str]) -> int:
    algorithm, image = day_20_load_data(raw_lines)
    for n in range(50):
        image = day_20_process_image(image, algorithm, '#' if n % 2 else '.')
    print('         Image Starts:')
    print('\n'.join(image))
    with open('output.txt', 'a') as file:
        for line in image:
            file.write(line + '\n')
    return sum([ln.count('#') for ln in image])


def day_20_load_data(raw_lines: [str]) -> (str, [str]):
    algorithm, input_image = '', []
    for rl in raw_lines:
        if len(algorithm) < 512:
            algorithm += rl
        elif len(rl) > 0:
            input_image.append(rl)
    return algorithm, input_image


def day_20_process_image(image: [str], algorithm: str,
                         border_char: str = '.') -> [str]:
    width, height = len(image[0]), len(image)
    # border_char = algorithm[0] if width > 100 or height > 100 else '.'
    processed = []
    x_range, y_range = (range(dim + 2) for dim in (width, height))
    for y in y_range:
        processed_row = ''
        for x in x_range:
            found_string = day_20_surrounding_pixels(Point(x, y), image,
                                                     empty_space_char=border_char)
            binary_list = [day_20_universal_convertor(ch) for ch in found_string]
            alg_index = binary_to_int(binary_list)
            processed_row += algorithm[alg_index]
        processed.append(processed_row)
    return processed


def day_20_surrounding_pixels(loc_in_new_image: (int,), current_image: [str],
                              empty_space_char: str = '.') -> str:
    width, height = len(current_image[0]), len(current_image)
    new_x, new_y = loc_in_new_image
    output = [[empty_space_char for _ in range(3)] for _ in range(3)]
    for i_y, y in enumerate(range(new_y - 2, new_y + 1)):
        for i_x, x in enumerate(range(new_x - 2, new_x + 1)):
            if 0 <= y < height and 0 <= x < width:
                output[i_y][i_x] = current_image[y][x]
    return ''.join([''.join([ch for ch in row]) for row in output])


def day_20_get_surrounding_string(pixel_loc: (int,), whole_image: [str]) -> str:
    x, y = pixel_loc
    return ''.join([whole_image[row][x - 1:x + 2] for row in range(y - 1, y + 2)])


def day_20_universal_convertor(item_in: object) -> object:
    return {'.': False, '#': True, True: '#', False: '.'}[item_in]


def day_19_part_two(raw_lines: [str]) -> int:
    scanners = day_19_load_scanner_data(raw_lines)
    all_offsets = day_19_find_all_scanner_offsets(scanners)
    greatest_distance = 0
    for off in all_offsets:
        for other_point in all_offsets:
            greatest_distance = max(greatest_distance, day_19_manhattan_distance(off, other_point))
    return greatest_distance


def day_19_part_one(raw_lines: [str]) -> int:
    scanners = day_19_load_scanner_data(raw_lines)
    return day_19_count_beacons(scanners)


def day_19_count_beacons(scanner_obs: dict) -> int:
    scanner_known_points = {0: scanner_obs[0]}
    while len(scanner_known_points) < len(scanner_obs):
        for scanner_id, point_list in scanner_obs.items():
            if scanner_id not in scanner_known_points:
                print(f'looking at scanner id {scanner_id}:')
                for tr in day_19_transforms:
                    transformed_points = [day_19_do_transformation(pt, tr) for pt in point_list]
                    for other_sc_id in scanner_known_points.keys():
                        offsets = [day_19_point_difference(tr_pt, known_pt)
                                   for tr_pt in transformed_points
                                   for known_pt in scanner_known_points[other_sc_id]]
                        for ofs in offsets:
                            overlaps = len([*filter(lambda o: o == ofs, offsets)])
                            if overlaps >= 12:
                                print(f'There are {len(offsets)} offsets, {overlaps} overlaps')
                                print(f'Offset is {ofs}')
                                scanner_known_points[scanner_id] = \
                                    [day_19_add_offset_to_point(pt_tr, ofs) for pt_tr in
                                     transformed_points]
                                break
                        if scanner_id in scanner_known_points:
                            break
                    if scanner_id in scanner_known_points:
                        break
    known_beacons = set([pt for pt_list in scanner_known_points.values() for pt in pt_list])
    return len(known_beacons)


def day_19_find_all_scanner_offsets(scanner_obs: dict) -> [(int,)]:
    scanner_offsets = []
    scanner_known_points = {0: scanner_obs[0]}
    while len(scanner_known_points) < len(scanner_obs):
        for scanner_id, point_list in scanner_obs.items():
            if scanner_id not in scanner_known_points:
                print(f'looking at scanner id {scanner_id}:')
                for tr in day_19_transforms:
                    transformed_points = [day_19_do_transformation(pt, tr) for pt in point_list]
                    for other_sc_id in scanner_known_points.keys():
                        offsets = [day_19_point_difference(tr_pt, known_pt)
                                   for tr_pt in transformed_points
                                   for known_pt in scanner_known_points[other_sc_id]]
                        for ofs in offsets:
                            overlaps = len([*filter(lambda o: o == ofs, offsets)])
                            if overlaps >= 12:
                                print(f'There are {len(offsets)} offsets, {overlaps} overlaps')
                                print(f'Offset is {ofs}')
                                scanner_known_points[scanner_id] = \
                                    [day_19_add_offset_to_point(pt_tr, ofs) for pt_tr in
                                     transformed_points]
                                scanner_offsets.append(ofs)
                                break
                        if scanner_id in scanner_known_points:
                            break
                    if scanner_id in scanner_known_points:
                        break
    return scanner_offsets



def day_19_manhattan_distance(point_1: (int,), point_2: (int,)) -> int:
    return sum([abs(d_2 - d_1) for d_1, d_2 in zip(point_1, point_2)])


def day_19_point_difference(point_1: (int,), point_2: (int,)) -> (int,):
    x_1, y_1, z_1 = point_1
    x_2, y_2, z_2 = point_2
    return Point3(x_2 - x_1, y_2 - y_1, z_2 - z_1)


def day_19_add_offset_to_point(point: (int,), offset: (int,)) -> (int,):
    p_x, p_y, p_z = point
    o_x, o_y, o_z = offset
    return Point3(*(sum(dims) for dims in zip(point, offset)))


def day_19_load_scanner_data(raw_lines: [str]) -> dict:
    observations = {}
    scanners = 0
    for ln in raw_lines:
        if "scanner" in ln:
            observations[scanners] = []
            scanners += 1
        elif ln and (ln[0].isnumeric() or ln[0] == "-"):
            observations[scanners - 1].append(Point3(*[int(ch) for ch in ln.split(",")]))
    return observations


def day_19_do_transformation(point: (int,), transformation: str) -> (int,):
    x, y, z = point
    return eval(f"Point3{transformation}")


day_19_transforms = ["(x, y, z)", "(-y, x, z)", "(-x, -y, z)", "(y, -x, z)",
              "(-x, y, -z)", "(-y, -x, -z)", "(x, -y, -z)", "(y, x, -z)",
              "(-z, y, x)", "(-z, -x, y)", "(-z, -y, -x)", "(-z, x, -y)",
              "(z, y, -x)", "(z, -x, -y)", "(z, -y, x)", "(z, x, y)",
              "(x, -z, y)", "(y, -z, -x)", "(-x, -z, -y)", "(-y, -z, x)",
              "(x, z, -y)", "(y, z, x)", "(-x, z, y)", "(-y, z, -x)"]
Point3 = namedtuple("Point3", "x y z")


def day_18_part_one(sum_list: [str]) -> int:
    addition_result = day_18_cumulative_addition(sum_list)
    return day_18_magnitude(addition_result)


def day_18_part_two(sum_list: [str]) -> int:
    answer = 0
    perms = [*permutations(sum_list, 2)]
    print(f'There are {len(perms)} possible sums from list of {len(sum_list)} sums: ')
    for sum_1, sum_2 in perms:
        print(f'{sum_1} + {sum_2}')
        result = day_18_magnitude(day_18_addition(sum_1, sum_2))
        answer = max(answer, result)
    return answer


def day_18_cumulative_addition(terms: [str]) -> str:
    result = terms[0]
    for trm in terms[1:]:
        result = day_18_addition(result, trm)
    return result


def day_18_addition(left_term: str, right_term: str) -> str:
    print(f'ADDING {left_term} &&&&& {right_term}')
    new_pair = f'[{left_term},{right_term}]'
    return day_18_reduce(new_pair)


def day_18_reduce(expression: str) -> str:
    # TODO; Destination number could be double digits!
    #   - on either side
    #   either of the replaced numbers could be double digits
    #   there might be no number to replace in either direction
    #   make sure all explosion possibilities are exhausted before starting on splits
    def replace_adjacent_number(number_index: int, go_right=False) -> str:
        print(f'original expression: {expression}')
        lookup = day_18_get_number_positions(expression)
        adjacent_number_index, old_number, this_number = 0, -1, -1
        if go_right and max([*lookup.keys()]) > number_index:
            adjacent_number_index = min([k for k in lookup.keys() if k > number_index])
        elif (not go_right) and min([*lookup.keys()]) < number_index - 1:
            adjacent_number_index = max([k for k in lookup.keys() if k < number_index - 1])
        if adjacent_number_index:
            old_number = lookup[adjacent_number_index]
            if number_index in lookup:
                this_number = lookup[number_index]
            elif number_index - 1 in lookup:
                this_number = lookup[number_index - 1]
            print(f'This number is {this_number}')
            if old_number > -1:
                new_number = old_number + this_number
                continuation_index = adjacent_number_index + 1
                if old_number > 9:
                    continuation_index += 1
                print(f'Replacing {old_number}, to {"right" if go_right else "left"}, '
                      f'with {new_number}')
                return f'{expression[:adjacent_number_index]}{new_number}{expression[continuation_index:]}'
            return expression

            # if this_number == 14 and (not go_right):
            #     print('hi')
        print(f'Replacing {old_number if old_number > -1 else "no number"}, to {"right" if go_right else "left"}')

        replacement_offset = 0
        this_number = int(expression[number_index])
        assert this_number < 20
        expr_slice = slice(number_index + 1, len(expression)) if go_right \
            else slice(number_index - 1, 0, -1)
        # DOUBLE-DIGIT NUMBER CHECK:
        if not go_right and expression[number_index - 1].isnumeric():
            this_number = int(expression[number_index - 1:number_index + 1])
            print(f'** FOUND DOUBLE-DIGIT NUMBER TO LEFT: {this_number}')
            expr_slice = slice(number_index - 2, 0, -1)
        elif expression[number_index + 1].isnumeric():
            this_number = int(expression[number_index:number_index + 2])
            print(f'** FOUND DOUBLE-DIGIT NUMBER TO RIGHT: {this_number}')
            expr_slice = slice(number_index + 2, len(expression))

        double_digit_dest = False
        for ind, ch in enumerate(expression[expr_slice]):
            if ch.isnumeric():
                replacement_offset = ind
                double_digit_dest = expression[expr_slice][ind + 1].isnumeric()
                if this_number > 9:
                    replacement_offset += 1
                break
        if replacement_offset:
            repl_index = number_index + replacement_offset + 1 if go_right \
                else number_index - replacement_offset - 1
            # if expression[repl_index:repl_index + 1 + double_digit_dest] == "7,":
            #     print('hi')
            dest_slice = slice(repl_index - (double_digit_dest if not go_right else 0),
                               repl_index + 1 + (double_digit_dest if go_right else 0))
            new_number = int(expression[dest_slice]) + this_number
            # new_number = int(expression[repl_index:repl_index + 1 + double_digit_dest]) + this_number
            assert new_number < 100
            return f'{expression[:repl_index]}{new_number}{expression[repl_index + 1 + double_digit_dest:]}'
        return expression

    need_to_reduce = True
    can_split = False
    while need_to_reduce:
        open_bracket_count = 0
        for index, char in enumerate(expression):
            if char == '[':
                open_bracket_count += 1
            elif char == ']':
                open_bracket_count -= 1
            if open_bracket_count == 5:
                can_split = False
                if char == ",":
                    left_dd_offset = expression[index - 2].isnumeric()
                    right_dd_offset = expression[index + 2].isnumeric()
                    if right_dd_offset:
                        print(f'DOUBLE-DIGITS!!! right dd offset is {right_dd_offset}')
                    original_expr_len = len(expression)
                    expression = replace_adjacent_number(index - 1)
                    if len(expression) > original_expr_len:
                        index += 1
                    expression = replace_adjacent_number(index + 1, go_right=True)
                    expression = f'{expression[:index - 2 - left_dd_offset]}0' \
                                 f'{expression[index + 3 + right_dd_offset:]}'
                    print(f'Explosion -> {expression}')
                    assert day_18_bracket_consistency_check(expression)
                    break
            elif char.isnumeric() and expression[index - 1].isnumeric() and can_split:
                dd_number = int(expression[index - 1:index + 1])
                expression = f'{expression[:index - 1]}[{dd_number // 2},' \
                             f'{(dd_number // 2) + (1 if dd_number % 2 == 1 else 0)}]' \
                             f'{expression[index + 1:]}'
                print(f'Split -> {expression}')
                break
            if index == len(expression) - 1:
                if can_split:
                    need_to_reduce = False
                can_split = True
    return expression


def day_18_get_number_positions(expression: str) -> dict:
    number_positions = {}
    numeric_indices = [i for i, ch in enumerate(expression) if ch.isnumeric()]
    i_ni = 0
    while i_ni < len(numeric_indices):
        position = numeric_indices[i_ni]
        number_positions[position] = int(expression[position])
        if (i_ni < len(numeric_indices) - 1) and \
                numeric_indices[i_ni + 1] == numeric_indices[i_ni] + 1:
            number_positions[position] = int(expression[position:position + 2])
            i_ni += 1
        i_ni += 1
    return number_positions


def day_18_bracket_consistency_check(expression: str) -> bool:
    open_bracket_count = 0
    for index, char in enumerate(expression):
        if char == '[':
            open_bracket_count += 1
        elif char == ']':
            open_bracket_count -= 1
    return open_bracket_count == 0


def day_18_magnitude(expression: str) -> int:
    l_expression, r_expression = day_18_split_expression(expression)
    l_magnitude = 3 * (int(l_expression) if l_expression.isnumeric()
                       else day_18_magnitude(l_expression))
    r_magnitude = 2 * (int(r_expression) if r_expression.isnumeric()
                       else day_18_magnitude(r_expression))
    return l_magnitude + r_magnitude


def day_18_split_expression(expression: str) -> (str, str):
    open_bracket_count = 0
    for index, char in enumerate(expression):
        if char == '[':
            open_bracket_count += 1
        elif char == ']':
            open_bracket_count -= 1
        if open_bracket_count == 1 and char == ',':
            return expression[1:index], expression[index + 1:-1]
    return "ERROR!"


def day_17_part_two(target_area: ((int, ), )) -> int:
    x_range, y_range = target_area
    print(f'Starting from y = {min(y_range)}')
    return sum([day_17_velocity_hits_target(Point(x, y), target_area)
                for x in range(max(x_range) + 1)
                for y in range(min(y_range), 150)])
    # successes = 0
    # for x in range((max(x_range) + 1) // 2):
    #     for y in range(min(y_range), 150):
    #         successes += day_17_velocity_hits_target(Point(x, y), target_area)
    # return successes


def day_17_part_one(target_area: ((int, ), )) -> int:
    x_range, y_range = target_area
    greatest_y_velocity = 0
    for x in range((max(x_range) + 1) // 2):
        for y in range(150):
            if day_17_velocity_hits_target(Point(x, y), target_area):
                greatest_y_velocity = max(greatest_y_velocity, y)
    return day_17_peak_given_starting_y_velocity(greatest_y_velocity)


def day_17_peak_given_starting_y_velocity(y_velocity: int) -> int:
    return sum([*range(y_velocity + 1)])


def day_17_velocity_hits_target(velocity: (int, ), target: ((int,), )) -> bool:
    initial_velocity = velocity
    x, y = Point(0, 0)
    x_range, y_range = target
    while x <= max(x_range) and y >= min(y_range):
        new_point, velocity = day_17_trajectory_step(Point(x, y), velocity)
        x, y = new_point
        if min(x_range) <= x <= max(x_range) and min(y_range) <= y <= max(y_range):
            print(f'Velocity: {initial_velocity} -> ({x}, {y}).')
            return True
    # print(f'Velocity: {initial_velocity} -> Reaches ({x}, {y}) without hitting target')
    return False


def day_17_trajectory_step(start: (int, ), velocity: (int, )) -> ((int, ), (int, )):
    """Assumes x-component of velocity would never be < 0"""
    next_point = Point(start.x + velocity.x, start.y + velocity.y)
    next_velocity = Point(max(velocity.x - 1, 0), velocity.y - 1)
    return next_point, next_velocity


# TODO for Day 16:
#   string to work with is the whole remaining hex string?
#   convert it to binary
#   determine the length of the next packet (including sub-packets!)
#   for next iteration, slice the main hex string after the current packet
#

class Day16Packet:
    # TODO: extra trailing zeroes only appear after the very outer packet
    def __init__(self, data_string: str, convert_to_binary=True):
        self.binary = day_16_hex_string_to_binary(data_string.strip()) \
            if convert_to_binary else data_string
        self.version_sum, self.value = 0, 0
        self.sub_packet_values = []

    def read_packet(self):
        self.version_sum += binary_to_int(self.binary[:3])
        if day_16_packet_is_operator(self.binary):
            length_is_number_of_sub_packets = int(self.binary[6])
            operator = binary_to_int(self.binary[3:6])
            print('\n')
            print(f'Operator = {operator}')
            print(f'Binary: {self.binary}')
            if length_is_number_of_sub_packets:
                sub_packets = binary_to_int(self.binary[7:18])
                index = 18
                for _ in range(sub_packets):
                    index = self.read_next_sub_packet(index)
            else:
                bit_length = binary_to_int(self.binary[7:22])
                index = 22
                while index < 22 + bit_length:
                    index = self.read_next_sub_packet(index)
            print(f'Sub-packet values: {self.sub_packet_values}')
            if operator == 0:
                self.value = sum(self.sub_packet_values)
            elif operator == 1:
                self.value = self.sub_packet_values[0]
                if len(self.sub_packet_values) > 1:
                    for spv in self.sub_packet_values[1:]:
                        self.value *= spv
            elif operator == 2:
                self.value = min(self.sub_packet_values)
            elif operator == 3:
                self.value = max(self.sub_packet_values)
            elif operator == 5:
                self.value = 1 if self.sub_packet_values[0] > self.sub_packet_values[1] else 0
            elif operator == 6:
                self.value = 1 if self.sub_packet_values[0] < self.sub_packet_values[1] else 0
            elif operator == 7:
                self.value = 1 if self.sub_packet_values[0] == self.sub_packet_values[1] else 0
            return index
        print(f'Literal binary: {self.binary}')
        self.value = day_16_get_value_from_literal(self.binary[6:day_16_get_literal_binary_length(self.binary, True)])
        print(f'Found literal value: {self.value}')
        return day_16_get_literal_binary_length(self.binary, True)

    def read_next_sub_packet(self, index: int) -> int:
        next_packet = Day16Packet(self.binary[index:], False)
        index += next_packet.read_packet()
        self.version_sum += next_packet.get_version_sum()
        self.sub_packet_values.append(next_packet.get_value())
        return index

    def get_version_sum(self):
        return self.version_sum

    def get_value(self):
        return self.value

    def get_length_in_hex_chars(self) -> int:
        if day_16_packet_is_operator(self.binary):
            length_type_id = int(self.binary[6])
            if length_type_id:
                sub_packets = binary_to_int(self.binary[7:18])
                index = 18
                for _ in range(sub_packets):
                    new_packet = Day16Packet(self.binary[index:], False)
                    index += new_packet.get_length_in_binary_chars()
            else:
                bit_length = binary_to_int(self.binary[7:22])
                index = 22 + bit_length
            while index < len(self.binary) and not int(self.binary[index]):
                index += 1
            if index % 4 == 0:
                return index // 4
            print('Oops, non-standard termination of binary string')
            assert index % 4 == 0
        else:
            return day_16_get_literal_binary_length(self.binary)

    def get_length_in_binary_chars(self) -> int:
        """deprecated - ONLY USED IN get_length_in_hex_chars"""
        if day_16_packet_is_operator(self.binary):
            return 0
        return day_16_get_literal_binary_length(self.binary, True)


def day_16_get_value_from_literal(binary: str) -> int:
    segments = len(binary) // 5
    string = ''.join([binary[seg * 5 + 1:seg * 5 + 5] for seg in range(segments)])
    return binary_to_int(string)


def day_16_get_literal_binary_length(binary: str, is_sub_packet: bool = False) -> int:
    stop = False
    index = 6
    while not stop:
        next_number = binary[index:index + 5]
        stop = not int(next_number[0])
        index += 5

    if is_sub_packet:
        return index

    while index < len(binary) and not int(binary[index]):
        index += 1
    if index % 4 == 0:
        return index // 4
    print('Oops, non-standard termination of binary string')
    assert index % 4 == 0


def day_16_hex_string_to_binary(hexadecimals: str) -> str:
    def convert(hex_char: str) -> str:
        binary_value = eval(f'bin(0x{hex_char})[2:]')
        return binary_value.rjust(4, "0")

    return ''.join([convert(ch) for ch in hexadecimals])


def day_16_get_version_no_from_binary_string(binary: str) -> int:
    return binary_to_int(binary[:3])


def day_16_packet_is_operator(packet: str) -> bool:
    return binary_to_int(packet[3:6]) != 4


def day_15_part_one(whole_grid: [[int]]):
    x, y, total = 0, 0, 0
    edge_length = len(whole_grid)
    while x < edge_length and y < edge_length:
        print(f'Next square origin: ({x}, {y})')
        next_square = day_15_cut_square(whole_grid, Point(x, y), 5)
        end_point, section_total = day_15_get_min_path_total_across_square(next_square)
        total += section_total
        print(f'--> Section total: {section_total}')
        x_incr, y_incr = end_point
        x, y = x + x_incr, y + y_incr
        if x == edge_length - 1 or y == edge_length - 1:
            print('Edge is reached')
            break
    if x == edge_length - 1:
        total += sum([row[-1] for row in whole_grid[y + 1:]])
    if y == edge_length - 1:
        total += sum(whole_grid[y][x + 1:])
    return total


def day_15_get_min_path_total_across_square(square: [[int]]) -> ((int), int):
    all_paths = day_15_get_all_paths_across_sub_square(square)
    min_total = min([*all_paths.values()])
    best_paths = [k for k, v in all_paths.items() if v == min_total]
    if len(best_paths) > 1:
        print(f'There is more than one optimal path')
        return best_paths[random.randrange(len(best_paths))], min_total
    return [k for k, v in all_paths.items() if v == min_total][0], min_total


def day_15_get_all_paths_across_sub_square(sub_square: [[int]]) -> {}:
    possible_paths = day_15_binary_combinations((len(sub_square) - 1) * 2)
    paths_dict = {}
    for path in possible_paths:
        end, path_total = day_15_calc_path_total(sub_square, path)
        paths_dict[end] = path_total
    return paths_dict


def day_15_binary_combinations(length: int = 4) -> [bool]:
    return product(range(2), repeat=length)


def day_15_cut_square(whole_grid: [[int]], origin: (int, int), size: int) -> [[int]]:
    whole_grid_size = len(whole_grid)
    ox, oy = origin.x, origin.y
    if ox + size > whole_grid_size:
        size = whole_grid_size - ox
    if oy + size > whole_grid_size:
        size = whole_grid_size - oy
    return [[whole_grid[y][x] for x in range(ox, ox + size)]
            for y in range(oy, oy + size)]


def day_15_calc_path_total(square: [[int]], path_steps: (bool)) -> ((int), int):
    """True -> across, False -> down"""
    x, y, total = 0, 0, 0
    width = len(square)
    end_point = Point(0, 0)
    # print('\nNew Path: ', end='')
    for go_across in path_steps:
        if go_across:
            x += 1
        else:
            y += 1
        if any([dim >= width for dim in (x, y)]):
            break
        total += square[y][x]
        end_point = Point(x, y)
        # print(square[y][x], end='')
    return end_point, total


def day_14_part_one(raw_input: [str]) -> int:
    polymer, insertions = day_14_load_data(raw_input)
    new_polymer = day_14_iterate_insertion_process(polymer, insertions, 10)
    return day_14_do_final_calculation(new_polymer)


def day_14_part_two(raw_input: [str]) -> int:
    polymer, insertions = day_14_load_data(raw_input)
    last_element = polymer[-1]
    pairs = day_14_create_initial_pair_dict(polymer)
    pair_replacements = {k: (k[0] + v, v + k[1]) for k, v in insertions.items()}
    for _ in range(40):
        pairs = day_14_growth_step(pairs, pair_replacements)
    print(f'After 40 steps, length of polymer is {sum([*pairs.values()])}')
    letter_counts = day_14_count_letters(pairs, last_element)
    most_common, rarest = (func([v for v in letter_counts.values() if v > 0])
                           for func in (max, min))
    return most_common - rarest


def day_14_count_letters(pair_dict: dict, last_letter: str) -> dict:
    letter_counts = {}
    uppercase = [chr(i) for i in range(65, 91)]
    for letter in uppercase:
        pairs_recorded = [k for k in pair_dict.keys() if k[0] == letter]
        if pairs_recorded:
            letter_counts[letter] = sum([pair_dict[pair] for pair in pairs_recorded])
            if letter == last_letter:
                letter_counts[letter] += 1
            print(f'Pairs recorded for {letter}: {letter_counts[letter]}')
    return letter_counts


def day_14_growth_step(pair_dict: dict, replacements: dict) -> dict:
    next_dict = {p: 0 for p in replacements.keys()}
    for pair, occurrences in pair_dict.items():
        new_pair_1, new_pair_2 = replacements[pair]
        next_dict[new_pair_1] += occurrences
        next_dict[new_pair_2] += occurrences
    return next_dict


def day_14_create_initial_pair_dict(polymer: str):
    '''In the initial string, how many of each pair?'''
    first_dict = {}
    for ind, elem in enumerate(polymer[:-1]):
        pair = polymer[ind:ind + 2]
        if pair in first_dict:
            first_dict[pair] += 1
        else:
            first_dict[pair] = 1
    return first_dict


def day_14_do_final_calculation(long_polymer: str) -> int:
    uppercase = [chr(i) for i in range(65, 91)]
    elem_counts = {letter: long_polymer.count(letter) for letter in uppercase}
    most_common, rarest = (func([v for v in elem_counts.values() if v > 0])
                           for func in (max, min))
    return most_common - rarest


def day_14_load_data(raw_input: [str]) -> (str, dict):
    return raw_input[0], {ln[:2]: ln[-1] for ln in raw_input if "->" in ln}


def day_14_iterate_insertion_process(polymer: str, insertions: dict, iterations: int) -> str:
    for _ in range(iterations):
        polymer = day_14_run_insertion_process(polymer, insertions)
    return polymer


def day_14_run_insertion_process(polymer: str, insertions: dict) -> str:
    product_polymer = ""
    for ind, elem in enumerate(polymer[:-1]):
        product_polymer += elem + insertions[polymer[ind:ind + 2]]
    product_polymer += polymer[-1]
    return product_polymer


def day_13_part_one(raw_input: [str]) -> int:
    points, folds = day_13_load_data(raw_input)
    first_fold_axis, fold_co_ord = folds[0]
    if first_fold_axis == 'x':
        folded_page = day_13_fold_to_left(points, fold_co_ord)
        return len(folded_page)
    print(f'Failed to process instructions: unexpected fold axis came first?')
    return 0


def day_13_part_two(raw_input: [str]):
    dots, folds = day_13_load_data(raw_input)
    for fold in folds:
        # print(f'Folding: {fold[0]}, {fold[1]}')
        axis, co_ord = fold
        dots = day_13_fold_up(dots, co_ord) if axis == "y" else day_13_fold_to_left(dots, co_ord)
    day_13_print_page(dots)


def day_13_load_data(raw_lines: [str]) -> ({tuple}, [tuple]):
    points = {eval(f"Point({ln})") for ln in raw_lines if "," in ln}
    folds = []
    for line in raw_lines:
        if "=" in line:
            axis_label, _, co_ordinate = line.partition("=")
            folds.append((axis_label[-1], int(co_ordinate)))
    return points, folds


def day_13_print_page(dots: {(int, int)}):
    """NB. Calculates page size on the basis of most extreme dot positions,
    not necessarily edge of page"""
    x_max, y_max = max([d.x for d in dots]), max([d.y for d in dots])
    print(f'Page size: {x_max + 1} x {y_max + 1}')
    for y in range(y_max + 1):
        print(''.join(["#" if Point(x, y) in dots else "." for x in range(x_max + 1)]))


def day_13_fold_up(dots: [(int, int)], axis_value: int) -> {(int, int)}:
    new_dots = [d if d.y < axis_value else Point(d.x, (axis_value * 2) - d.y)
                for d in dots]
    return set(new_dots)


def day_13_fold_to_left(dots: [(int, int)], axis_value: int) -> {(int, int)}:
    new_dots = [d if d.x < axis_value else Point((axis_value * 2) - d.x, d.y)
                for d in dots]
    return set(new_dots)


def day_12_part_one(raw_input: [str]) -> int:
    all_connections = day_12_load_all_connections(raw_input)
    return len(day_12_generate_paths(["start"], all_connections))


def day_12_part_two(raw_input: [str]) -> int:
    connections = day_12_load_all_connections(raw_input)
    return len(day_12_part_two_generate_paths(["start"], connections))


def day_12_generate_paths(path_so_far: [str], connections: [(str, str)]) -> [[str]]:
    next_nodes = day_12_list_connected_nodes(path_so_far[-1], connections)
    extended_paths = []
    for node in next_nodes:
        if node == "end":
            extended_paths.append(path_so_far + ["end"])
        elif node.isupper() or node not in path_so_far:
            new_path = path_so_far + [node]
            for p in day_12_generate_paths(new_path, connections):
                extended_paths.append(p)
    return extended_paths


def day_12_part_two_generate_paths(path_so_far: [str], connections: [(str, str)]) -> [[str]]:
    next_nodes = day_12_list_connected_nodes(path_so_far[-1], connections)
    extended_paths = []

    def small_cave_is_allowed(cave: str) -> bool:
        visited_small_caves = set(filter(lambda n: n.islower(), path_so_far))
        if all([path_so_far.count(cave) == 1 for cave in visited_small_caves]):
            return True
        return cave not in path_so_far

    for node in next_nodes:
        if node == "start":
            continue
        if node == "end":
            extended_paths.append(path_so_far + ["end"])
        elif node.isupper() or small_cave_is_allowed(node):
            new_path = path_so_far + [node]
            for p in day_12_part_two_generate_paths(new_path, connections):
                extended_paths.append(p)
    return extended_paths


def day_12_load_all_connections(raw_strings: [str]) -> [(str, str)]:
    return list(map(day_12_parse_path, raw_strings))


def day_12_parse_path(dash_path: str) -> (str, str):
    a, _, b = dash_path.partition("-")
    return a, b


def day_12_list_connected_nodes(this_node: str, all_connections: [(str, str)]) -> [str]:
    starting_points = filter(lambda c: this_node in c, all_connections)
    return [day_12_the_other_one(stp, this_node) for stp in starting_points]


def day_12_the_other_one(the_tuple: (), this_one: object) -> object:
    t_1, t_2 = the_tuple
    return t_2 if t_1 == this_one else t_1


day_11_flash_count, day_11_it_happened = 0, False


def day_11_part_one(starting_layout: [[int]], iterations: int) -> int:
    global day_11_flash_count
    day_11_flash_count = 0
    # print(f'Before Part One: {day_11_print_grid(starting_layout)} Flash count: {day_11_flash_count}')
    processed_layout = starting_layout
    for it in range(iterations):
        # print("================")
        processed_layout = day_11_reset_flashers(day_11_flash_step(processed_layout))
        if day_11_it_happened:
            print(f'FFS it actually happened!! On iteration {it + 1}')
            break
        # day_11_print_grid(processed_layout)
    # print(f'After Part One: {day_11_print_grid(processed_layout)}')
    return day_11_flash_count


def day_11_global_energy_increment(all_fish: [[int]]) -> [[int]]:
    return [[e + 1 for e in row] for row in all_fish]


def day_11_flash_step(octopi: [[int]]) -> [[int]]:
    x_max, y_max = len(octopi[0]), len(octopi)
    octopi = day_11_global_energy_increment(octopi)
    iterate = True
    flashed = []
    global day_11_flash_count, day_11_it_happened
    while iterate:
        iterate = False
        for y, row in enumerate(octopi):
            for x, octopus in enumerate(row):
                location = Point(x, y)
                if octopus > 9 and location not in flashed:
                    for n_x, n_y in day_11_get_all_neighbours(location, x_max, y_max):
                        octopi[n_y][n_x] += 1
                    flashed.append(location)
                    day_11_flash_count += 1
                    iterate = True
    octopi = day_11_reset_flashers(octopi)
    if len(flashed) == x_max * y_max:
        day_11_it_happened = True
    return octopi


def day_11_reset_flashers(octopi: [[int]]) -> [[int]]:
    def reset_if_flashed(energy_level: int) -> int:
        if energy_level > 9:
            return 0
        return energy_level

    return [list(map(reset_if_flashed, o_row)) for o_row in octopi]


def day_11_get_all_neighbours(location: (int, int),
                              x_bound: int, y_bound: int) -> [(int, int)]:
    def is_valid_point(p: (int, int)) -> bool:
        if p == location:
            return False
        x_p, y_p = p
        if -1 < x_p < x_bound and -1 < y_p < y_bound:
            return True
        return False

    x, y = location
    offsets = list(product(range(-1, 2), range(-1, 2)))
    surrounds = [(x + o_x, y + o_y) for o_x, o_y in offsets]
    return list(filter(is_valid_point, surrounds))


def day_11_print_grid(octopi: [[int]]):
    string = '\n'.join([''.join([f"{i}" for i in row]) for row in octopi])
    print(string)


def day_10_part_one(raw_inputs: [str]) -> int:
    all_results = map(day_10_report_line_error, raw_inputs)
    offending_brackets = filter(bool, all_results)
    scoring = {")": 3, "]": 57, "}": 1197, ">": 25137}
    return sum([scoring[ob] for ob in offending_brackets])


def day_10_part_two(all_lines: [str]) -> int:
    incomplete_lines = filter(lambda c: not day_10_report_line_error(c), all_lines)
    scores = [day_10_score_completion_string(day_10_calc_completion_string(ln))
              for ln in incomplete_lines]
    return sorted(scores)[len(scores) // 2]


def day_10_report_line_error(snippet: str) -> str:
    return day_10_process_line(snippet)


def day_10_process_line(snippet: str, give_completion_string: bool = False) -> str:
    brackets = {"]": "[", ")": "(", "}": "{", ">": "<"}  # closer: opener
    reversed_br = {v: k for k, v in brackets.items()}
    open_brackets = ""
    for br in snippet:
        if br in [*brackets.values()]:
            open_brackets += br
            # print(f'Open brackets: {open_brackets}')
        elif br in brackets and brackets[br] == open_brackets[-1]:
            open_brackets = open_brackets[:-1]
            # print(f"We're all good!  Open brackets: {open_brackets}")
        else:
            print(f'Expected {reversed_br[open_brackets[-1]]}, '
                  f'but found {br} instead.')
            return br
    return "".join([reversed_br[ob] for ob in open_brackets[::-1]]) \
        if give_completion_string else ""


def day_10_calc_completion_string(line: str) -> str:
    return day_10_process_line(line, give_completion_string=True)


def day_10_score_completion_string(string: str) -> int:
    scoring = {")": 1, "]": 2, "}": 3, ">": 4}
    total_score = 0
    for s in string:
        total_score *= 5
        total_score += scoring[s]
    return total_score


def day_9_part_one(puzzle_input: str) -> int:
    data_rows = day_9_load_data(puzzle_input)
    return sum([1 + h for h in day_9_get_all_low_point_values(data_rows)])


def day_9_load_data(raw_input: str) -> [[int]]:
    lines = Puzzle.convert_input(raw_input, None)
    return [Puzzle.convert_input(",".join([*ln])) for ln in lines]


def day_9_get_all_low_point_values(data_rows: [[int]]) -> [int]:
    return [data_rows[y][x] for x, y in day_9_get_all_low_point_co_ordinates(data_rows)]


def day_9_is_low_point(x: int, y: int, data_rows: [[int]]):
    point_value = data_rows[y][x]
    return point_value < min(day_9_get_neighbours(x, y, data_rows))


def day_9_get_neighbours(x: int, y: int, data_rows: [[int]]) -> [int]:
    max_x, max_y = len(data_rows[0]), len(data_rows)
    neighbouring_values = []
    for x_index in (x - 1, x + 1):
        if 0 <= x_index < max_x:
            neighbouring_values.append(data_rows[y][x_index])
    for y_index in (y - 1, y + 1):
        if 0 <= y_index < max_y:
            neighbouring_values.append(data_rows[y_index][x])
    return neighbouring_values


def day_9_part_two(puzzle_input: str) -> int:
    data_rows = day_9_load_data(puzzle_input)
    all_lows = day_9_get_all_low_point_co_ordinates(data_rows)
    basin_sizes = sorted([day_9_basin_size_from_low_point(lw, data_rows)
                          for lw in all_lows], reverse=True)
    # print(basin_sizes)
    for bs in basin_sizes[:3]:
        location = [k for k in basins_and_sizes if basins_and_sizes[k] == bs][0]
        print(f'Basin of size {bs} is located at {location}')
    return basin_sizes[0] * basin_sizes[1] * basin_sizes[2]


basins_and_sizes = {}


def day_9_basin_size_from_low_point(low: (int, int), data_rows: [[int]]) -> int:
    global basins_and_sizes

    max_x, max_y = len(data_rows[0]), len(data_rows)
    lp_x, lp_y = low

    def low_line_range() -> [int]:
        first_x = last_x = lp_x
        while first_x > 0 and data_rows[lp_y][first_x - 1] < 9:
            first_x -= 1
        while last_x + 1 < max_x and data_rows[lp_y][last_x + 1] < 9:
            last_x += 1
        return list(range(first_x, last_x + 1))

    def adjacent_lines(next_y: int, x_values: [int], up=True) -> int:
        if next_y < 0 or next_y >= max_y:
            return 0
        total = 0
        # x_values = day_9_extended_x_values(x_values)
        # any CONTINUOUS ROW of values < 9, at least one of whose indices is in x_values
        # TODO: also look back in case there is an isolated one on PREVIOUS row
        valid_x_in_next_row = []
        all_groups = day_9_get_all_groups_in_row(data_rows[next_y])
        for gr in all_groups:
            if any([x in x_values for x in gr]):
                valid_x_in_next_row += gr
        # valid_x_in_next_row = [x for x in x_values
        #                        if data_rows[next_y][x] < 9]
        total += len(valid_x_in_next_row)

        # LOOK OVER YOUR SHOULDER:
        this_row_groups = day_9_get_all_groups_in_row(data_rows[next_y - 1])
        for gr in this_row_groups:
            if len(gr) == 1 and gr[0] in valid_x_in_next_row and gr[0] not in x_values:
                total += 1

        if valid_x_in_next_row and 0 < next_y < max_y - 1:
            next_y += -1 if up else 1
            total += adjacent_lines(next_y, valid_x_in_next_row, up)
        return total

    next_x_values = low_line_range()
    total_size = len(next_x_values) + adjacent_lines(lp_y - 1, next_x_values)\
           + adjacent_lines(lp_y + 1, next_x_values, up=False)
    basins_and_sizes[low] = total_size
    return total_size


def day_9_get_all_groups_in_row(row: [int]) -> [[int]]:
    current_group, all_groups = [], []
    for i in range(len(row)):
        if row[i] < 9:
            current_group.append(i)
        elif current_group:
            all_groups.append(current_group)
            current_group = []
    if current_group:
        all_groups.append(current_group)
    return all_groups


def day_9_extended_x_values(x_vals: [int]) -> [int]:
    if len(x_vals) < 2 or (len(x_vals) == 2 and x_vals[1] - x_vals[0] > 1):
        return x_vals
    first_vals, middle, last_vals = [x_vals[0] - 1], [*x_vals], [x_vals[-1] + 1]
    if x_vals[1] > x_vals[0] + 1:
        first_vals = [x_vals[0], x_vals[1] - 1]
        middle = [*x_vals[1:]]
    if x_vals[-1] > x_vals[-2] + 1:
        last_vals = [x_vals[-2] + 1, x_vals[-1]]
        middle = [*middle[:-1]]
    final_list = first_vals + middle + last_vals
    return final_list


def day_9_get_all_low_point_co_ordinates(data_rows: [[int]]) -> [(int, int)]:
    max_y, max_x = len(data_rows), len(data_rows[0])
    return [Point(x, y) for y in range(max_y) for x in range(max_x)
            if day_9_is_low_point(x, y, data_rows)]


def day_8_part_one(input_lines: [str]) -> int:
    unique_signal_lengths = [2, 4, 3, 7]
    unique_length_digits = 0
    for s in input_lines:
        _, _, outputs = s.partition(" | ")
        output_values = outputs.split()
        unique_length_digits += len([v for v in output_values
                                     if len(v) in unique_signal_lengths])
    # print(f'There are {unique_length_digits} 1s, 4s, 7s or 8s in the example')
    return unique_length_digits


def day_8_part_two(input_rows: [str]) -> int:
    return sum([day_8_decode(outputs, day_8_do_deductions(inputs))
                for inputs, outputs in day_8_get_all_inputs(input_rows)])


def day_8_identify_unique_length_strings(inputs: [str]) -> {str: int}:
    return {day_8_order_string(day_8_extract_unique_string_of_length(inputs, k)): v
            for k, v in ((2, 1), (3, 7), (4, 4), (7, 8))}


def day_8_order_string(string: str) -> str:
    return ''.join(sorted(string))


def day_8_decode(output_strings: [str], dictionary: {}) -> int:
    output_digits = [dictionary[day_8_order_string(os)] for os in output_strings]
    return sum([d * 10 ** i for i, d in enumerate(output_digits[::-1])])


def day_8_do_deductions(garbled_strs: [str]) -> {}:
    encodings = day_8_identify_unique_length_strings(garbled_strs)
    ordered_strings = [day_8_order_string(s) for s in garbled_strs]
    six_strings, five_strings = (day_8_extract_strings_of_length(ordered_strings, length)
                                 for length in (6, 5))
    reverse_lookup = {v: k for k, v in encodings.items()}

    def deduce_and_remove(from_list: [str], subset: int, target: int):
        possibles = [s for s in from_list
                     if set(reverse_lookup[subset]) < set(s)]
        if len(possibles) == 1:
            encodings[possibles[0]] = target
            from_list.remove(possibles[0])
            reverse_lookup[target] = possibles[0]
        else:
            print(f'Deduction failed: looking for {target} in {six_strings}')

    deduce_and_remove(six_strings, 4, 9)
    deduce_and_remove(six_strings, 7, 0)
    encodings[six_strings[0]] = 6
    reverse_lookup[6] = six_strings[0]

    deduce_and_remove(five_strings, 7, 3)

    # 5 is a subset of 6
    possibles = [s for s in five_strings
                 if set(s) < set(reverse_lookup[6])]
    if len(possibles) == 1:
        encodings[possibles[0]] = 5
        five_strings.remove(possibles[0])
        reverse_lookup[5] = possibles[0]
    else:
        print(f'Deduction failed: looking for {5} in {five_strings}')
    encodings[five_strings[0]] = 2
    return encodings


def day_8_get_all_inputs(all_lines: str) -> [([str], [str])]:
    return [day_8_split_line(line) for line in all_lines]


def day_8_split_line(input_line: str) -> ([str], [str]):
    inputs, _, outputs = input_line.partition(" | ")
    return tuple(data.split() for data in (inputs, outputs))


def day_8_extract_unique_string_of_length(garbled_inputs: [str], length: int) -> str:
    all_strings = day_8_extract_strings_of_length(garbled_inputs, length)
    if len(all_strings) == 1:
        return all_strings[0]
    return "Oops! These strings are not as unique as you thought."


def day_8_extract_strings_of_length(garbled_inputs: [str], length: int) -> [str]:
    return [s for s in garbled_inputs if len(s) == length]


def day_7_individual_fuel_cost_linear(origin: int, destination: int) -> int:
    return abs(origin - destination)


def day_7_individual_fuel_cost_non_linear(origin: int, destination: int) -> int:
    distance = abs(origin - destination)
    return sum(list(range(1, distance + 1)))


def day_7_group_fuel_cost(all_positions: [int], destination: int, method: object) -> int:
    return sum([method(p, destination) for p in all_positions])


def day_7_find_minimum_fuel_cost(positions: [int], calc_method: object) -> int:
    # for dest in range(len(positions) + 1):
    #     print(f'{dest:>2} --> {day_7_group_fuel_cost(positions, dest)}')
    return min([day_7_group_fuel_cost(positions, dest, calc_method)
                for dest in range(len(positions) + 1)])


# TODO: Day Six - Lanternfish
# each generation:
#   calculate how many baby fish are born
#           (number of live fish whose reproduction is scheduled for that generation)
#   update baby fish
#   current kid fish become live fish
#       - their number is effective number they would have had if they'd been alive at start
#   current baby fish become kid fish
def fish_reproduces(fish_number: int, generation: int) -> bool:
    return generation % 7 == (fish_number + 1) % 7


def day_6_part_one(starting_fish: [int], generations: int) -> int:
    previous_gen_length, previous_additions = len(starting_fish), 0

    live_fish = starting_fish
    baby_fish, kid_fish, teen_fish = [], [], []
    for g in range(1, generations + 1):
        hatches = [(g + 1) % 7] * len([fi for fi in live_fish if fish_reproduces(fi, g)])
        live_fish += teen_fish
        teen_fish = [*kid_fish]
        kid_fish = [*baby_fish]
        baby_fish = hatches

        gen_len = len(live_fish + baby_fish + kid_fish + teen_fish)
        fish_added = gen_len - previous_gen_length
        # print(f'Gen {g} += {fish_added:>6} {fish_added - previous_additions:>6}')
        # print(f'Gen {g:>2} = {gen_len:>6}')
        previous_gen_length = gen_len
        previous_additions = fish_added
        # if hatches:
        #     print(f'Generation {g} babies: {baby_fish}, kids: {kid_fish}, teens: {teen_fish}')
        # print(f'Generation {g} total = {len(live_fish + baby_fish + kid_fish + teen_fish)}')
    return len(live_fish + baby_fish + kid_fish + teen_fish)


def day_6_look_for_patterns(starting_fish: [int]):
    """DIDN'T HELP"""
    generations = 0
    results = {}
    live_fish = starting_fish
    baby_fish, kid_fish, teen_fish = [], [], []
    for g in range(1, generations + 1):
        hatches = [(g + 1) % 7] * len([fi for fi in live_fish if fish_reproduces(fi, g)])
        live_fish += teen_fish
        teen_fish = [*kid_fish]
        kid_fish = [*baby_fish]
        baby_fish = hatches
        results[g] = len(live_fish + baby_fish + kid_fish + teen_fish)

    print('\n')
    for k, v in results.items():
        double_string = ''
        if k > 80:
            diff_from_dbl = v - (results[k - 8] * 2)
            sign = '+' if diff_from_dbl > 0 else ' '
            double_string = f"Double gen-8 {sign}{diff_from_dbl:>4}"
            print(f'{k:>3} -- {v:>5} = {double_string}')
        # if k > 10:
        #     for sub_k, sub_v in results.items():
        #         if sub_k < k and v / sub_v == 2.0:
        #             print(f'gen {k} div gen {sub_k} = {v / sub_v}')

                    # does it double every eight generations??


def day_6_seed_dictionary(starting_fish: [int]) -> {int: int}:
    return {n - 8: starting_fish.count(n) for n in range(-1, 8)}


def day_6_generations_of_interest(current_generation: int) -> range:
    return range(current_generation - 9, -10, -7)


def day_6_part_two(starting_fish: [int], generations: int) -> int:
    result = len(starting_fish)
    growth_dict = day_6_seed_dictionary(starting_fish)
    for gen in range(generations + 1):
        population_increase = sum([growth_dict[gi] for gi in
                                   day_6_generations_of_interest(gen)])
        growth_dict[gen] = population_increase
        result += population_increase
        print(f'Gen {gen:>3} adds {population_increase:>5} --> {result}')
    return result


#Todo: Day Five:
# 1. get points into namedtuples
# 2. filter out diagonals using could_be_square()
# 3. expand all lines to create big list (duplicates permissible) of all points on lines
# 4. determine the size of the grid using max. x and y found in lines
# 5. iterate through every point in grid a count occurrences of that point in the list
# 6. count number of points where no. of appearances in the big list is >=2
Point = namedtuple("Point", "x y")
Line = namedtuple("Line", "a b")


def day_5_part_one(input_rows: [str]) -> int:
    straight_lines = day_5_non_diagonal_lines(day_5_points_from_input_rows(input_rows))
    all_vent_points = day_5_expand_all_lines(straight_lines)
    max_x, max_y = max([vp.x for vp in all_vent_points]), max([vp.y for vp in all_vent_points])
    print(f'Max (x, y): ({max_x}, {max_y})')
    return day_5_count_danger_points(max_x, max_y, all_vent_points)


def day_5_part_two(input_rows: [str]) -> int:
    all_lines = day_5_points_from_input_rows(input_rows)
    all_vent_points = day_5_expand_all_lines(all_lines)
    return day_5_count_danger_points(0, 0, all_vent_points)


def day_5_points_from_input_rows(input_rows: [str]) -> []:
    lines = []
    for row in input_rows:
        pt_1, _, pt_2 = row.partition(" -> ")
        points = (Point(*tuple(int(c) for c in eval(pt))) for pt in (pt_1, pt_2))
        new_line = Line(*points)
        lines.append(new_line)
    return lines


def day_5_count_danger_points(x_max: int, y_max: int, vents: {Point: int}) -> int:
    return len([v for v in vents.values() if v >= 2])


def day_5_expand_all_lines(lines: [Line]) -> {Point: int}:
    all_vent_points = {}
    for ln in lines:
        new_points = day_5_expand_line(ln)
        for np in new_points:
            if np in all_vent_points:
                all_vent_points[np] += 1
            else:
                all_vent_points[np] = 1
    return all_vent_points


def day_5_expand_line(line: Line) -> [Point]:
    x_diff, y_diff = line.b.x - line.a.x, line.b.y - line.a.y
    x_range, y_range = 0, 0
    if x_diff:
        x_range = range(x_diff + 1) if x_diff > 0 else range(0, x_diff - 1, -1)
    if y_diff:
        y_range = range(y_diff + 1) if y_diff > 0 else range(0, y_diff - 1, -1)
    if x_diff and y_diff:
        return [Point(line.a.x + x, line.a.y + y) for x, y in zip(x_range, y_range)]
    if x_diff:
        return [Point(line.a.x + x, line.a.y) for x in x_range]
    return [Point(line.a.x, line.a.y + y) for y in y_range]


# x_dim, y_dim = abs(line.b.x - line.a.x), abs(line.b.y - line.a.y)
    # x_origin, y_origin = min(line.a.x, line.b.x), min(line.a.y, line.b.y)
    # if x_dim and y_dim:
    #     assert x_dim == y_dim
    #     return [Point(line.a.x + n, line.a.y + n) for n in range(x_dim + 1)]
    # if x_dim:
    #     return [Point(x, line.a.y) for x in range(x_origin, x_origin + x_dim + 1)]
    # return [Point(line.a.x, y) for y in range(y_origin, y_origin + y_dim + 1)]


def day_5_non_diagonal_lines(all_lines: [Line]) -> [Line]:
    return list(filter(day_5_could_be_square, all_lines))


def day_5_could_be_square(line: Line) -> bool:
    p1, p2 = line
    return p1.x == p2.x or p1.y == p2.y


# todo: Day Four:
# 1. represent boards
# 1a. split input text into boards
# 2. store the 10 sets of 5 numbers for each board
# 3. iterate through list of drawn numbers
# 4. remove the last-drawn number from each board-set it's a member of
# 5. if any of the sets has become empty, the board it belongs to has won
# 6. sum the remaining numbers for all 10 sets for that board and multiply by number drawn

def day_4_part_one(input_text: str):
    draw, board_strings = day_4_parse_input(input_text)
    boards = [day_4_generate_board(b) for b in board_strings]
    for number in draw:
        print(number)
        boards = [day_4_remove_from_board(bd, number) for bd in boards]
        if any([day_4_board_wins(bd) for bd in boards]):
            for i, b in enumerate(boards):
                if day_4_board_wins(b):
                    print(f'Board {i} has won!!!')
                    print(f'The answer is {day_4_score_board(b) * number}')
                    break
            break
    return boards


def day_4_part_two(input_text: str):
    draw, board_strings = day_4_parse_input(input_text)
    boards = [day_4_generate_board(b) for b in board_strings]
    no_of_boards = len(boards)
    completed_boards = set()
    for number in draw:
        boards = [day_4_remove_from_board(bd, number) for bd in boards]
        if any([day_4_board_wins(bd) for bd in boards]):
            for i, b in enumerate(boards):
                if day_4_board_wins(b):
                    # print(f'Board {i} has won!!!')
                    completed_boards = completed_boards.union({i})
                    if len(completed_boards) == no_of_boards:
                        print(f'The answer to part two is: {day_4_score_board(b) * number}')
                        break



    return len(completed_boards)


def day_4_parse_input(raw_text: str) -> ([int], []):
    lines = Puzzle.convert_input(raw_text, conversion_func=None, blank_lines_matter=True)
    boards, new_board = [], []
    for ln in lines:
        if ln and ',' not in ln:
            new_board.append(ln)
        if len(new_board) == 5:
            boards.append('\n'.join(new_board))
            new_board = []
    return [int(i) for i in lines[0].split(',')], boards


def day_4_generate_board(raw_board: str) -> [{int}]:
    rows = [[int(n) for n in r.split()] for r in raw_board.split('\n')]
    cols = [list(z) for z in zip(*rows)]
    return [set(num_list) for num_list in rows + cols]


def day_4_remove_from_board(board: [{}], number: int) -> [{}]:
    return [s - {number} for s in board]


def day_4_board_wins(board: [{}]) -> bool:
    return any([not s for s in board])


def day_4_score_board(board: [{}]) -> bool:
    return sum(set([i for s in board for i in list(s)]))


def get_bit_value_of_interest(binaries: [bool], most_common=True) -> bool:
    if most_common:
        return sum(binaries) >= len(binaries) / 2
    return sum(binaries) < len(binaries) / 2


def zero_one_strings_to_binary(strings: [str]) -> [[int]]:
    return [[int(b) for b in i] for i in strings]


def day_three_part_one(inputs: [str]) -> int:
    no_of_rows = len(inputs)
    binary_rows = zero_one_strings_to_binary(inputs)
    modal_binary = [sum(col) > no_of_rows // 2 for col in zip(*binary_rows)]
    least_common_bin = invert_binary(modal_binary)
    return binary_to_int(modal_binary) * binary_to_int(least_common_bin)


def calc_rating(bin_inputs: [[bool]], oxygen_generator=True) -> int:
    column = 0
    while len(bin_inputs) > 1:
        columns = list(zip(*bin_inputs))
        bit_value = get_bit_value_of_interest(columns[column], oxygen_generator)
        bin_inputs = list(filter(lambda bi: bi[column] == bit_value, bin_inputs))
        column += 1
    return binary_to_int(bin_inputs[0])


def day_three_part_two(binary_lists: [[bool]]) -> int:
    return calc_rating(binary_lists) * calc_rating(binary_lists, False)



class DayTwo:
    def __init__(self, input_commands: [str]):
        self.commands = input_commands
        self.fwds = list(filter(lambda c: self.first_word(c) == 'forward', input_commands))
        self.verticals = [c for c in input_commands if c not in self.fwds]

    @staticmethod
    def first_word(command: str) -> str:
        return command.split()[0]

    @staticmethod
    def amount(command: str) -> int:
        return int(command.split()[-1])

    @staticmethod
    def vert_factor(command: str) -> int:
        return -1 if DayTwo.first_word(command) == 'up' else 1

    def get_aggregate_forward_motion(self) -> int:
        return sum([self.amount(f) for f in self.fwds])

    def part_one(self) -> int:
        depth = sum([self.vert_factor(v) * self.amount(v) for v in self.verticals])
        return self.get_aggregate_forward_motion() * depth

    def part_two(self) -> int:
        final_depth = aim = 0
        for cmd in self.commands:
            if self.first_word(cmd) == 'forward':
                final_depth += aim * self.amount(cmd)
            else:
                aim += self.vert_factor(cmd) * self.amount(cmd)
        return self.get_aggregate_forward_motion() * final_depth


def day_one_part_one(input_depths: [int]) -> int:
    depth_increases = [d > input_depths[i] for i, d in enumerate(input_depths[1:])]
    assert len(depth_increases) == len(input_depths) - 1
    return len(list(filter(bool, depth_increases)))


def day_one_part_two(input_depths: [int]) -> int:
    window_length = 3
    depth_sums = [sum(input_depths[n:n + 3]) for n in range(len(input_depths) - window_length + 1)]
    return day_one_part_one(depth_sums)


if __name__ == '__main__':
    Puzzle(6).input_as_list()