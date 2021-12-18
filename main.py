from collections import namedtuple
from itertools import product, combinations
import random


class Puzzle:
    def __init__(self, day: int):
        self.day = day

    def get_text_input(self) -> str:
        with open(f'inputs\\input{self.day}.txt', 'r') as input_file:
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
        self.version_sum = 0

    def read_packet(self):
        self.version_sum += binary_to_int(self.binary[:3])
        if day_16_packet_is_operator(self.binary):
            length_is_number_of_sub_packets = int(self.binary[6])
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
            return index
        return day_16_get_literal_binary_length(self.binary, True)

    def read_next_sub_packet(self, index: int) -> int:
        next_packet = Day16Packet(self.binary[index:], False)
        index += next_packet.read_packet()
        self.version_sum += next_packet.get_version_sum()
        return index

    def get_version_sum(self):
        return self.version_sum

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


def day_15_get_min_path_total_across_square(square: [[int]]) -> ((int), int):
    all_paths = day_15_get_all_paths_across_sub_square(square)
    min_total = min([*all_paths.values()])
    best_paths = [k for k, v in all_paths.items() if v == min_total]
    if len(best_paths) > 1:
        print(f'There is more than one optimal path')
        return best_paths[random.randrange(len(best_paths))]
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