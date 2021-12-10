from main import Puzzle, Point
import main


class TestDayTen:
    example_input = """[({(<(())[]>[[{[]{<()<>>
[(()[<>])]({[<{<<[]>>(
{([(<{}[<>[]}>{[]{[(<()>
(((({<>}<{<{<>}{[]{[]{}
[[<[([]))<([[{}[[()]]]
[{[{({}]{}}([{[{{{}}([]
{<[[]]>}<{[{[{[]{()[[[]
[<(<(<(<{}))><([]([]()
<{([([[(<>()){}]>(<<{{
<{([{{}}[<[[[<>{}]]]>[]]"""

    def test_init(self):
        snippets = Puzzle.convert_input(self.example_input)
        # print('\n'.join([f"{sn} = {len(sn)} chars" for sn in snippets]))
        assert len(snippets) == 10

    def test_first_attempt(self):
        all_snippets = Puzzle.convert_input(self.example_input)
        for i, sn in enumerate(all_snippets):
            print(f'Snippet {i + 1}:')
            main.day_10_report_line_error(sn)

    def test_part_one(self):
        all_lines = Puzzle.convert_input(self.example_input)
        assert main.day_10_part_one(all_lines) == 26397
        p1_lines = Puzzle(10).input_as_list(None)
        solution = main.day_10_part_one(p1_lines)
        print(f'The solution to Part One is {solution}')
        assert solution == 339411

    def test_completing_lines(self):
        all_lines = Puzzle.convert_input(self.example_input)
        incomplete_lines = list(filter(lambda c: not main.day_10_report_line_error(c),
                                       all_lines))
        # print('Incomplete lines are:')
        # for il in incomplete_lines:
        #     print(il)
        completion_strings = ["}}]])})]", ")}>]})", "}}>}>))))", "]]}}]}]}>", "])}>"]
        for i, v in enumerate(completion_strings):
            assert main.day_10_calc_completion_string(incomplete_lines[i]) == v
        scores = [288957, 5566, 1480781, 995444, 294]
        for j in range(len(scores)):
            assert main.day_10_score_completion_string(completion_strings[j]) == scores[j]

    def test_part_two(self):
        raw_data = Puzzle.convert_input(self.example_input)
        validation = main.day_10_part_two(raw_data)
        assert validation == 288957
        p2_lines = Puzzle(10).input_as_list(None)
        solution = main.day_10_part_two(p2_lines)
        print(f'Solution to Part Two is {solution}')


class TestDayNine:
    example_input = """2199943210
3987894921
9856789892
8767896789
9899965678"""

    def test_initial(self):
        data = main.day_9_load_data(self.example_input)
        assert len(data) == 5
        assert len(data[0]) == 10

    def test_logic(self):
        data = main.day_9_load_data(self.example_input)
        assert main.day_9_get_neighbours(3, 3, data) == [6, 8, 6, 9]
        assert main.day_9_get_neighbours(0, 0, data) == [1, 3]
        assert main.day_9_get_neighbours(9, 4, data) == [7, 9]
        assert main.day_9_get_neighbours(9, 2, data) == [9, 1, 9]
        assert main.day_9_get_neighbours(5, 4, data) == [9, 5, 9]
        assert not main.day_9_is_low_point(0, 0, data)
        assert main.day_9_is_low_point(1, 0, data)
        assert main.day_9_get_all_low_point_values(data) == [1, 0, 5, 5]

    def test_part_one(self):
        raw_data_string = self.example_input
        assert main.day_9_part_one(raw_data_string) == 15
        puzzle_data = Puzzle(9).get_text_input()
        solution = main.day_9_part_one(puzzle_data)
        print(f'The solution to Part One is {solution}')
        assert solution == 508

    def test_basin_finding(self):
        # assume all basins are bounded by 9s
        data = Puzzle(9).get_text_input()
        print(data.replace('9', '-'))
        eg_data = main.day_9_load_data(self.example_input)
        assert main.day_9_get_all_low_point_co_ordinates(eg_data) == \
               [Point(x=1, y=0), Point(x=9, y=0), Point(x=2, y=2), Point(x=6, y=4)]
        # assert main.day_9_extended_x_values([2]) == [2]
        # assert main.day_9_extended_x_values([1,2,3]) == list(range(5))
        # assert main.day_9_extended_x_values([1,2,3,5]) == list(range(5)) + [5]
        # assert main.day_9_extended_x_values([48, 50, 51, 52, 54]) == \
        #        list(range(48, 55))
        # assert main.day_9_extended_x_values([100, 102, 103, 104]) == list(range(100, 106))
        # assert main.day_9_extended_x_values([7, 9]) == [7, 9]
        # assert main.day_9_get_all_groups_in_row([1, 2, 3, 9, 9, 9, 9, 8, 8]) == \
        #        [[0, 1, 2], [7, 8]]
        # assert main.day_9_get_all_groups_in_row([9, 1, 9, 0, 9]) == [[1], [3]]
        assert main.day_9_basin_size_from_low_point(Point(x=1, y=0), eg_data) == 3
        assert main.day_9_basin_size_from_low_point(Point(x=9, y=0), eg_data) == 9
        assert main.day_9_basin_size_from_low_point(Point(x=2, y=2), eg_data) == 14
        assert main.day_9_basin_size_from_low_point(Point(x=9, y=4), eg_data) == 9
        assert main.day_9_basin_size_from_low_point(Point(x=75, y=10),
                                                    main.day_9_load_data(data)) == 127

    def test_part_two(self):
        assert main.day_9_part_two(self.example_input) == 1134
        solution = main.day_9_part_two(Puzzle(9).get_text_input())
        print(f'Part Two solution: {solution}')
        assert solution > 1526250   # answer was too low
        # TODO: debugging.  (Puzzle input -> 100 x 100 grid)
        # grab a sample of it with more complex basins and see if I can see the problem



class TestDayEight:
    example_input = """be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb |
fdgacbe cefdb cefbgd gcbe
edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec |
fcgedb cgb dgebacf gc
fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef |
cg cg fdcagb cbg
fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega |
efabcd cedba gadfec cb
aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga |
gecf egdcabf bgf bfgea
fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf |
gebdcfa ecba ca fadegcb
dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf |
cefg dcbef fcge gbcadfe
bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd |
ed bcgafe cdgba cbgef
egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg |
gbdfcae bgc cg cgb
gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc |
fgae cfgab fg bagce""".replace("|\n", "| ")

    one_line_example = """acedgfb cdfbe gcdfa fbcad dab cefabd cdfgeb eafb cagedb ab |
cdfeb fcadb cdfeb cdbaf""".replace("|\n", "| ")

    def test_initial(self):
        assert main.day_8_part_one(Puzzle.convert_input(self.example_input, None)) == 26

    def test_part_one(self):
        solution = main.day_8_part_one(Puzzle(8).input_as_list(None))
        print(f'Part One solution: {solution}')
        assert solution == 239

    def test_setup_for_part_two(self):
        test_strs = self.one_line_example.split()
        assert len(main.day_8_extract_strings_of_length(test_strs, 5)) == 7
        assert main.day_8_extract_unique_string_of_length(test_strs, 3) == "dab"
        assert 'Oops' in main.day_8_extract_unique_string_of_length(test_strs, 6)
        ins, outs = main.day_8_split_line(self.one_line_example)
        # print(main.day_8_split_line(self.one_line_example))
        assert len(ins), len(outs) == (10, 4)
        all_inputs = main.day_8_get_all_inputs(Puzzle.convert_input(self.example_input, None))
        assert len(all_inputs) == 10
        assert all([len(i) == 2 for i in all_inputs])
        assert all([(len(a), len(b)) == (10, 4) for a, b in all_inputs])

    def test_deductions(self):
        fives = [2, 3, 5]
        sixes = [0, 6, 9]
        # top bar is in 7 but not 4 or 1
        # four -> five: five is the only five-segment digit with top-left-side segment
        # bottom-left-side segment is in 8 but not 9
        # top-right-side segment is in 8 but not 6

        # USE SET ARITHMETIC - (NO GOOD)
        # generate known sets from the unique-length strings
        # key for true 'a' = set(messed-up 7) - set(messed-up 1)
        # key for true 'c' = set(messed-up 8) - set(messed-up 6)
        # key for true 'd' = set(messed-up 8) - set(messed-up 0)
        # key for true 'e' = set(messed-up 8) - set(messed-up 9)
        # key for true 'b' = set(messed-up 8) - set(messed-up 3) - set(messed-up 2)

        # IDENTIFY NUMBERS, NOT SEGMENTS:
        # 1. Split into groups by length of string:
        # 2. The four numbers of unique length are identified already (1, 4, 7, 8)
        # 3. For the strings of length six (-> [0, 6, 9]):
        #   a. 9 is the only one that 4 is a proper subset of
        #   b. of the remaining two, 0 is the only one that 7 is a proper subset of
        #   c. that leaves 6, by process of elimination
        # 4. For the strings of length five (-> [2, 3, 5]):
        #   a. 3 is the only one that 7 is a proper subset of
        #   b. of the remaining two, 5 is the only one that is a proper subset of 6
        #   c. that leaves 2, by process of elimination
        # 5. Save all these in a dictionary of ordered strings, {'abc': 7, . . . }
        # 6. Write a function to convert input strings to ordered strings for matching
        # 7. Translate all the numbers and get the final sum
        assert main.day_8_order_string('cdba') == 'abcd'
        test_strings = Puzzle.convert_input(self.one_line_example)
        test_garbled = main.day_8_get_all_inputs(test_strings)[0][0]
        print(test_garbled)
        unique_dict = main.day_8_identify_unique_length_strings(test_garbled)
        assert unique_dict == {"ab": 1, "abd": 7, "abef": 4, "abcdefg": 8}
        deductions = main.day_8_do_deductions(test_garbled)
        print(deductions)
        assert len(deductions) == 10
        assert deductions == {'ab': 1, 'abd': 7, 'abef': 4, 'abcdefg': 8,
                                    'abcdef': 9, 'abcdeg': 0, 'bcdefg': 6, 'abcdf': 3,
                                    'acdfg': 2, 'bcdef': 5}
        assert main.day_8_decode(main.day_8_get_all_inputs(test_strings)[0][1],
                                 deductions) == 5353

        # print(set('cbg') - set('gc'))

    def test_part_two(self):
        input_rows = Puzzle.convert_input(self.example_input, None)
        assert main.day_8_part_two(input_rows) == 61229
        final_input = Puzzle(8).input_as_list(None)
        solution = main.day_8_part_two(final_input)
        print(f'Part Two solution is {solution}')
        assert solution == 946346


class TestDaySeven:
    example_input = "16,1,2,0,4,2,7,1,2,14"

    def test_init(self):
        positions = Puzzle.convert_input(self.example_input)
        assert len(positions) == 10
        assert all([isinstance(pos, int) for pos in positions])
        big_list = Puzzle(7).input_as_list()
        assert all([isinstance(pos, int) for pos in big_list])

    def test_fuel_costs(self):
        assert main.day_7_individual_fuel_cost_linear(16, 2) == 14
        assert main.day_7_individual_fuel_cost_linear(1, 2) == 1
        assert main.day_7_individual_fuel_cost_linear(2, 16) == 14
        assert main.day_7_individual_fuel_cost_linear(2, 2) == 0
        # fuel costs for group move:
        positions = Puzzle.convert_input(self.example_input)
        method = main.day_7_individual_fuel_cost_linear
        assert main.day_7_group_fuel_cost(positions, 2, method) == 37
        assert main.day_7_group_fuel_cost(positions, 1, method) == 41
        assert main.day_7_group_fuel_cost(positions, 3, method) == 39
        assert main.day_7_group_fuel_cost(positions, 10, method) == 71
        # get optimal position to move to:
        assert main.day_7_find_minimum_fuel_cost(positions, method) == 37

    def test_part_one(self):
        answer = main.day_7_find_minimum_fuel_cost(Puzzle(7).input_as_list(),
                                                   main.day_7_individual_fuel_cost_linear)
        print(f'The answer to part one is {answer}')
        assert answer == 349357


    def test_non_linear_fuel_costs(self):
        assert main.day_7_individual_fuel_cost_non_linear(16, 5) == 66
        assert main.day_7_individual_fuel_cost_non_linear(5, 16) == 66
        assert main.day_7_individual_fuel_cost_non_linear(5, 1) == 10
        assert main.day_7_individual_fuel_cost_non_linear(2, 2) == 0
        assert main.day_7_individual_fuel_cost_non_linear(1, 2) == 1
        meth = main.day_7_individual_fuel_cost_non_linear
        positions = Puzzle.convert_input(self.example_input)
        assert main.day_7_group_fuel_cost(positions, 5, meth) == 168
        assert main.day_7_group_fuel_cost(positions, 2, meth) == 206

    def test_part_two(self):
        method = main.day_7_individual_fuel_cost_non_linear
        assert main.day_7_find_minimum_fuel_cost(Puzzle.convert_input(self.example_input),
                                                 method) == 168
        puzzle_positions = Puzzle(7).input_as_list()
        print(f'Part two solution is '
              f'{main.day_7_find_minimum_fuel_cost(puzzle_positions,method)}')


class TestDaySix:
    example_input = "3,4,3,1,2"

    def test_init(self):
        numbers = Puzzle.convert_input(self.example_input)
        assert len(numbers) == 5
        assert sum(numbers) == 13
        assert numbers.count(3) == 2
        numbers_2 = [*numbers]
        assert main.day_6_part_one(numbers, 18) == 26
        assert main.day_6_part_one(numbers_2, 80) == 5934

    def test_reproduction_function(self):
        assert not main.fish_reproduces(3, 1)
        assert main.fish_reproduces(3, 4)
        assert main.fish_reproduces(3, 18)
        assert main.fish_reproduces(1, 2)

    def test_part_one(self):
        original_generation = Puzzle(6).input_as_list()
        solution = main.day_6_part_one(original_generation, 80)
        assert solution == 352195
        print(f'The solution to Part One is {solution}')

    def test_pattern_serach(self):
        main.day_6_look_for_patterns(Puzzle.convert_input(self.example_input))

    def test_part_two_dictionary_solution(self):
        original_generation = Puzzle.convert_input(self.example_input)
        start_dict = main.day_6_seed_dictionary(original_generation)
        assert len(start_dict) == 9
        assert len([v for v in start_dict.values() if not v]) == 5
        assert start_dict[-7] == 1
        assert start_dict[-5] == 2
        print(start_dict)
        assert list(main.day_6_generations_of_interest(0)) == [-9]
        assert list(main.day_6_generations_of_interest(1)) == [-8]
        assert list(main.day_6_generations_of_interest(2)) == [-7]
        assert list(main.day_6_generations_of_interest(20)) == [11, 4, -3]
        first_few_gens = {0: 5, 1: 5, 2: 6, 3: 7, 4: 9, 5: 10, 8: 10, 9: 11, 18: 26,
                          80: 5934, 256: 26984457539}
        for k, v in first_few_gens.items():
            assert main.day_6_part_two(original_generation, k) == v

    def test_part_two(self):
        original_generation = Puzzle(6).input_as_list()
        assert main.day_6_part_two(original_generation, 80) == 352195
        solution = main.day_6_part_two(original_generation, 256)
        assert solution == 1600306001288
        print(f'Part Two solution is {solution}')


class TestDayFive:
    example_input = """0,9 -> 5,9
8,0 -> 0,8
9,4 -> 3,4
2,2 -> 2,1
7,0 -> 7,4
6,4 -> 2,0
0,9 -> 2,9
3,4 -> 1,4
0,0 -> 8,8
5,5 -> 8,2"""

    def test_initial(self):
        input_rows = main.Puzzle.convert_input(self.example_input, None)
        found_lines = main.day_5_points_from_input_rows(input_rows)
        assert len(found_lines) == 10
        assert found_lines[0].a.x == 0
        assert found_lines[-1].b.y == 2
        assert len(main.day_5_non_diagonal_lines(found_lines)) == 6

    def test_expand_line(self):
        input_rows = main.Puzzle.convert_input(self.example_input, None)
        found_lines = main.day_5_points_from_input_rows(input_rows)
        lines = main.day_5_non_diagonal_lines(found_lines)
        assert len(lines) == 6
        print(f'our first line is {lines[0]}')
        for ln in lines:
            print(ln)
        first_expanded = main.day_5_expand_line(lines[0])
        assert len(first_expanded) == 6
        assert len(main.day_5_expand_line(lines[1])) == 7
        assert len(main.day_5_expand_line(lines[2])) == 2
        print('All lines as points:')
        for ln in lines:
            print(main.day_5_expand_line(ln))
        all_vent_points = main.day_5_expand_all_lines(lines)
        assert len(all_vent_points) == 21
        assert all_vent_points[main.Point(0, 9)] == 2
        assert main.day_5_count_danger_points(9, 9, all_vent_points) == 5
        assert main.day_5_part_one(input_rows) == 5

    def test_part_one(self):
        input_rows = main.Puzzle.convert_input(self.example_input, None)
        assert main.day_5_part_one(input_rows) == 5
        part_one_rows = main.Puzzle(5).input_as_list(conversion_func=None)
        part_one_solution = main.day_5_part_one(part_one_rows)
        print(f'Solution to Part One is {part_one_solution}')
        assert part_one_solution == 5280

    def test_generalised_expand_line(self):
        input_rows = main.Puzzle.convert_input(self.example_input, None)
        found_lines = main.day_5_points_from_input_rows(input_rows)
        print(found_lines)
        assert len(found_lines) == 10
        first_diagonal = main.day_5_expand_line(found_lines[1])
        assert len(first_diagonal) == 9
        expected_points = [main.Point(x, y) for x, y in zip(range(8, -1, -1), range(9))]
        print(f'Expected points are {expected_points}')
        assert first_diagonal == expected_points

    def test_part_two(self):
        input_rows = main.Puzzle.convert_input(self.example_input, None)
        assert main.day_5_part_two(input_rows) == 12
        part_two_solution = main.day_5_part_two(main.Puzzle(5).input_as_list(conversion_func=None))
        print(f'The solution to Part Two is {part_two_solution}')
        assert part_two_solution == 16716


class TestDayFour:
    draw = [7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1]
    board_1 = """22 13 17 11  0
 8  2 23  4 24
21  9 14 16  7
 6 10  3 18  5
 1 12 20 15 19"""
    board_2 = """ 3 15  0  2 22
 9 18 13 17  5
19  8  7 25 23
20 11 10 24  4
14 21 16 12  6"""
    board_3 = """14 21 17 24  4
10 16 15  9 19
18  8 23 26 20
22 11 13  6  5
 2  0 12  3  7"""
    example_input = ','.join([f'{n}' for n in draw]) + '\n\n' +\
                    '\n\n'.join([board_1, board_2, board_3])


    def test_board_is_a_list_of_sets(self):
        board = """10  2
 1 11"""
        two_by_two_result = main.day_4_generate_board(board)
        assert len(two_by_two_result) == 4
        print(f'Two-by-two:\n{two_by_two_result}')
        five_by_five_result = main.day_4_generate_board(self.board_1)
        assert len(five_by_five_result) == 10
        print(f'5x5:\n{five_by_five_result}')


    def test_parsing_input(self):
        new_boards = [main.day_4_generate_board(bs) for bs in
                      (self.board_1, self.board_2, self.board_3)]
        for b in new_boards:
            assert len(b) == 10

    def test_removal_of_number(self):
        bd = main.day_4_generate_board(self.board_1)
        bd = main.day_4_remove_from_board(bd, 22)
        assert len(bd) == 10
        assert len([s for s in bd if len(s) == 5]) == 8
        assert len([s for s in bd if len(s) == 4]) == 2
        for n in (13, 17, 11, 0):
            assert not main.day_4_board_wins(bd)
            bd = main.day_4_remove_from_board(bd, n)
        assert main.day_4_board_wins(bd)

    def test_run_part_one(self):
        board = """10  2
1 11"""
        two_by_two_result = main.day_4_generate_board(board)
        assert main.day_4_score_board(two_by_two_result) == 24
        main.day_4_part_one(Puzzle(4).get_text_input())

    def test_part_two(self):
        assert main.day_4_part_two(Puzzle(4).get_text_input()) == 3


class TestDayThree:
    example_input = """00100
11110
10110
10111
10101
01111
00111
11100
10000
11001
00010
01010"""

    def test_binary_to_int(self):
        assert main.binary_to_int([False]) == 0
        assert main.binary_to_int([True]) == 1
        assert main.binary_to_int([1, 1]) == 3
        assert main.binary_to_int([True, False, False, True, True, True, False]) == 78
        assert main.binary_to_int([1, 0, 0, 5, 13, -1, 0]) == 78
        assert main.binary_to_int(['a', '', 'long string']) == 5
        assert main.binary_to_int('string') == 63

    def test_inversion(self):
        assert main.invert_binary([True]) == [False]
        assert main.invert_binary([False]) == [True]
        assert main.invert_binary([1, 0, 0, 5, 13, -1, 0]) == [False, True, True, False,
                                                               False, False, True]
        assert main.binary_to_int(main.invert_binary(['a', '', 'long string'])) == 2

    def test_part_one_example(self):
        inputs = Puzzle.convert_input(self.example_input, conversion_func=None)
        assert len(inputs) == 12
        assert all([isinstance(s, str) for s in inputs])
        assert main.day_three_part_one(inputs) == 198

    def test_part_one_solution(self):
        data = Puzzle(3).input_as_list(conversion_func=None)
        answer = main.day_three_part_one(data)
        print(f'Solution to Part One is {answer}')
        assert answer == 845186

    def test_part_two_initial(self):
        assert main.get_bit_value_of_interest([True, False], True) == True
        assert main.get_bit_value_of_interest([1, 0, 1, 0, 0], most_common=False)
        assert main.get_bit_value_of_interest([False, False, True], most_common=False)
        assert main.get_bit_value_of_interest([1, 1, 0, 0], False) == 0

        raw_input_strings = Puzzle.convert_input(self.example_input, conversion_func=None)
        binaries = main.zero_one_strings_to_binary(raw_input_strings)
        assert main.calc_rating(binaries, oxygen_generator=True) == 23
        assert main.calc_rating(binaries, oxygen_generator=False) == 10
        assert main.day_three_part_two(binaries) == 230

    def test_part_two_solution(self):
        raw_inputs = Puzzle(3).input_as_list(False)
        bins = main.zero_one_strings_to_binary(raw_inputs)
        solution = main.day_three_part_two(bins)
        assert solution == 4636702
        print(f'The solution to part two is {solution}')


class TestDayTwo:
    example_input = """forward 5
down 5
forward 8
up 3
down 8
forward 2"""

    def test_initial(self):
        commands = Puzzle.convert_input(self.example_input)
        assert len(commands) == 6

    def test_forward_motion(self):
        commands = Puzzle.convert_input(self.example_input)
        fwds = list(filter(lambda c: c[:7] == 'forward', commands))
        forward_motion = sum([int(f.split()[-1]) for f in fwds])
        assert forward_motion == 15
        verticals = [c for c in commands if c not in fwds]
        depth = sum([(-1 if v[:2] == 'up' else 1) * int(v.split()[-1]) for v in verticals])
        assert depth == 10

    def test_part_one_example(self):
        assert main.DayTwo(Puzzle.convert_input(self.example_input)).part_one() == 150

    def test_part_one_solution(self):
        assert main.DayTwo(Puzzle(2).input_as_list()).part_one() == 1762050

    def test_part_two_example(self):
        assert main.DayTwo(Puzzle.convert_input(self.example_input)).part_two() == 900

    def test_part_two_solution(self):
        solution = main.DayTwo(Puzzle(2).input_as_list()).part_two()
        print(f'\nPart Two answer: {solution}')
        assert solution == 1855892637


class TestDayOne:
    test_input = """199
200
208
210
200
207
240
269
260
263"""

    def test_run(self):
        depths = Puzzle(1).input_as_list()
        assert len(depths) == 2000

    def test_part_one(self):
        depths = Puzzle.convert_input(self.test_input)
        assert len(depths) == 10
        assert main.day_one_part_one(depths) == 7

    def test_part_one_answer(self):
        assert main.day_one_part_one(Puzzle(1).input_as_list()) == 1676

    def test_part_two(self):
        assert main.day_one_part_two(Puzzle.convert_input(self.test_input)) == 5
        assert self.test_input

    def test_part_two_answer(self):
        print(f'Part Two answer: {main.day_one_part_two(Puzzle(1).input_as_list())}')
