from main import Puzzle, Point
import main
import itertools


class TestDayTwentyFour:
    eg_1 = """inp x
mul x -1"""
    eg_2 = """inp z
inp x
mul z 3
eql z x"""
    eg_3 = """inp w
add z w
mod z 2
div w 2
add y w
mod y 2
div w 2
add x w
mod x 2
div w 2
mod w 2"""
    div_z_26 = [False] * 4 + [True, False, False, True, False] + [True] * 5

    def test_init(self):
        state = {chr(o): 0 for o in range(ord("w"), ord("w") + 4)}
        for line in self.eg_1.split('\n'):
            if 'inp' in line:
                line += " 1"
            state = main.day_24_process_one_line(state, line)
        assert state["x"] == -1

    def test_running_programs(self):
        program = Puzzle.convert_input(self.eg_1 + '\n')
        for n in range(100):
            output = main.day_24_run_program(program, (n,))
            assert output["x"] == -n
            assert sum([*output.values()]) == -n
            assert sum([v == 0 for v in output.values()]) == 3 if n > 0 else 4
        two_args_program = Puzzle.convert_input(self.eg_2, None)
        output = main.day_24_run_program(two_args_program, (3, 9))
        assert output["z"] == 1
        output = main.day_24_run_program(two_args_program, (-113, -339))
        assert output["z"]
        output = main.day_24_run_program(two_args_program, (50, 13))
        assert output["z"] == 0
        binary_convertor = Puzzle.convert_input(self.eg_3, None)
        result = main.day_24_run_program(binary_convertor, (15,))
        assert all([v == 1 for v in result.values()])
        for num in range(16):
            binary_values = main.day_24_run_program(binary_convertor, (num,))
            assert main.binary_to_int([*binary_values.values()]) == num

    def test_monad_run(self):
        arg_string = "17" * 7
        args = (int(ch) for ch in arg_string)
        monad = Puzzle(24).input_as_list(None)
        print(f'{" " * 4}{"*" * 20}')
        print(f'Running monad program:')
        validity = main.day_24_run_program(monad, args)
        print(f'Example number gives validity of {validity}')

    def test_fucking_about(self):
        print('\n')
        monad = Puzzle(24).input_as_list(None)
        unique_ops, unique_var, unique_args = set(), set(), set()
        for i, line in enumerate(monad[::18]):
            operation_sequence = " ".join([monad[(18 * i) + n][:3] for n in range(18)])
            print(operation_sequence)
            variable_sequence = " ".join([f"{monad[(18 * i) + n][4]:>3}" for n in range(18)])
            print(variable_sequence)
            argument_sequence = " ".join([f"{monad[(18 * i) + n][6:]:>3}" for n in range(18)])
            print(argument_sequence)
            unique_ops.add(operation_sequence)
            unique_var.add(variable_sequence)
            unique_args.add(argument_sequence)
        assert len(unique_ops) == len(unique_var) == 1
        example_num = "13579246899999"
        for x in range(100):
            last_digits = f'{x}'
            if len(last_digits) == 2 and "0" not in last_digits:
                candidate = f'{example_num[:12]}{last_digits}'
                final_z = main.day_24_run_program(monad, candidate)["z"]
                # print(final_z)
                if not final_z:
                    print(f'{candidate} IS VALID!')
        for z in range(962, 962 + 26):
            print(f'** INITIAL z: {z}')
            for n in range(1, 10):
                valid = self.validity_predictor(f'{n}21', initial_z=z)
                if valid:
                    print(f'{str(n)}21 is valid model number fragment')
                    assert self.model_no_fragment_is_valid(f'{n}21', initial_z=z)
                else:
                    assert not self.model_no_fragment_is_valid(f'{n}21', initial_z=z)
    # prevalence of valid model numbers is not even 1 in 10,000

        """OPERATIONS:
        w: always takes input, not touched until input step on next round
        x: (z -> x), mod 26, add random no. (-7 to 15), != w
        y: -> 25, mul x, +1, -> w (after mul z y), add random (1 to 15), mul x
        z: (div 26 or unch.), mul y, add y (last step) """

    @staticmethod
    def model_no_fragment_is_valid(model_number: str, initial_z: int=0) -> bool:
        print(f"Testing {model_number}, z={initial_z}")
        assert "0" not in model_number
        monad_prog = Puzzle(24).input_as_list(None)
        final_state = main.day_24_run_program(monad_prog[-18 * len(model_number):],
                                              model_number, initial_z)
        return final_state["z"] == 0

    def test_reverse_engineering_approach(self):
        """Start with final state, with z=0.  Get list of inputs into the
        final step that would get to z=0
        For each of those inputs, there will be a (infinite?) range
        of possible z's, which can be expressed as a generator.
        As we work backwards, there will be a growing list of model number
        fragments, each with its own generator to express the initial possible z's
        at the start of the furthest-back step
        TODO: what do you need to know at start of each step: w and z?"""
        last_step_results = self.reverse_step()
        print(f'After processing last step: {last_step_results}')
        for input_num, gen in last_step_results:
            z_value = next(gen)
            assert self.model_no_fragment_is_valid(str(input_num), z_value)
            print(f'input number: {input_num} -> z: {z_value}')
        second_last_step = self.reverse_step(final_z=9, step_no=12)
        for input_num, gen in second_last_step:
            # for _ in range(5):
            #     assert self.model_no_fragment_is_valid(str(input_num), next(gen))
            print(f'input number: {input_num} -> z: {[next(gen) for _ in range(1)]}')

    def test_go_back_two_steps(self):
        print('\n*** TWO-STEP process')
        valid_items = []
        end_step_results = self.reverse_step()
        for end_digit, z_gen in end_step_results:
            # print(end_digit, [z for z in z_gen])
            for z in z_gen:
                second_last_step = self.reverse_step(final_z=z, step_no=12)
                for dgt, gen in second_last_step:
                    for z_g in gen:
                        third_last_step = self.reverse_step(final_z=z_g, step_no=11)
                        for dgt_3, zg_3 in third_last_step:
                #     print(dgt, [g for g in gen])
                            valid_items.append((int(f"{dgt_3}{dgt}{end_digit}"), next(zg_3)))
        for vi in valid_items:
            print(f"{vi}")
            three_digits, init_z = vi
            assert self.model_no_fragment_is_valid(f"{three_digits}", init_z)
        assert len(valid_items) == 729

    def reverse_step(self, final_z: int=0, step_no: int=13) -> [(int, int)]:
        valid_w_z_combos = []
        x_increment = main.day_24_add_to_x[step_no]
        y_increment = main.day_24_add_to_y[step_no]
        w_range = range(1, 10)
        for w in w_range:
            # z_generator = itertools.count(26 + w - x_increment, 26)
            initial_z = (26 * final_z) + w - x_increment
            z_generator = iter(range(initial_z, initial_z + 1))
            valid_w_z_combos.append((w, z_generator))
        return valid_w_z_combos

    @staticmethod
    def process_readable_step(state: dict, digit: int, x_incr: int, y_incr: int) -> dict:
        initial_z = state["z"]
        w = digit
        x, y, z = TestDayTwentyFour.readable_step(initial_z, digit, x_incr, y_incr)
        for var in "wxyz":
            state[var] = eval(var)
        return state

    @staticmethod
    def readable_step(z: int, digit: int, x_incr: int, y_incr: int) -> (int, ):
        x = (digit - x_incr) != z % 26
        if x_incr <= 0:
            z //= 26
        if x:
            y = 26
            z *= y
            y = (digit + y_incr) * x
            z += y
        else:
            y = 0   # does this make any difference?
        return x, y, z

    def test_one_off(self):
        assert self.model_no_fragment_is_valid("99", 252)

    def validity_predictor(self, model_number: str, initial_z: int=0) -> bool:
        state = main.day_24_run_program([], "")
        state["z"] = initial_z
        for mn_digit, div_z, x_incr, y_incr in zip(model_number,
                                                   self.div_z_26[-len(model_number):],
                                                   main.day_24_add_to_x[-len(model_number):],
                                                   main.day_24_add_to_y[-len(model_number):]):
            number = int(mn_digit)
            # state["w"] = number
            # state["x"] = ((state["z"] % 26) + x_incr) != number
            # # left side of != could go negative, but "x" limited to 0 or 1
            # state["y"] = (25 * state["x"]) + 1
            # if div_z:
            #     state["z"] = state["z"] // 26
            #     # div_z only True when add_to_x <= 0
            # # at this point, y can only be 1 or 26
            # state["z"] *= state["y"]
            # state["y"] = (number + y_incr) * state["x"]
            # # y never goes negative
            # state["z"] += state["y"]
            state = self.process_readable_step(state, number, x_incr, y_incr)
        # print(state)
        monad_prog = Puzzle(24).input_as_list(None)
        expected = main.day_24_run_program(monad_prog[-18 * len(model_number):],
                                           model_number, initial_z)
        # print(expected)
        assert state == expected
        # assert self.model_no_fragment_is_valid(model_number)
        return state["z"] == 0

    def tst_part_one(self):
        main.day_24_part_one()

    # To get final z = 0:   (x=0, y=0, z<26 always -> final z=0 if div_z)
    #   last digit == prev. z (would be prev. z + x_incr but x_incr = 0 here)
    #   there is no x=1 solution
    # LAST DIGIT could be 1-9.  Initial z of last step = last digit
    # Second-last digit would have to be:
    #   how to get a z that's anywhere between 1 and 9?
    #   36 < z < 46 and number=z-35, e.g. 36, 1 -> z=1.  Always -> z=1.
    #   also 62 < z < 72, number=z-61 -> z=number, etc. . . .
    # Third-last digit:
    #   942 < z < 952, number=z-941 -> z=36, etc.

    def tst_silly(self):
        """doesn't get anywhere in a reasonable time"""
        for x in range(int("9" * 14), int("1" * 14), -1):
            if "0" not in f"{x}":
                if not (x % 3827):
                    print(f'Got to: {x}')
                if self.model_no_fragment_is_valid(f"{x}"):
                    print(f"{x} is valid!")
                    break

    def test_individual_model_number_validity(self):
        candidate = "99991991911111"
        validity = self.model_no_fragment_is_valid(candidate)
        print(f'{candidate} is {"VALID" if validity else "not valid"}')
        assert not validity


class TestDayTwentyThree:
    example_text = """#############
#...........#
###B#C#B#D###
  #A#D#C#A#
  #########
"""
    part_two_additional_text = """  #D#C#B#A#
  #D#B#A#C#"""

    def test_init(self):
        initial_config = main.day_23_load_data(self.example_text)
        print(initial_config)
        assert main.day_23_is_completed(main.Configuration('...........',
                                                           ["AA", "BB", "CC", "DD"]))

    def test_part_one_functions(self):
        initial_config = main.day_23_load_data(self.example_text)
        assert initial_config == main.Configuration("." * 11, ['AB', 'DC', 'CB', 'AD'])
        assert len(main.day_23_get_next_valid_moves(initial_config)) == 28
        next_config = main.day_23_make_move(initial_config, (6, 3))
        assert next_config == \
               main.Configuration("...B.......", ["AB", "DC", "C", "AD"])
        assert main.day_23_get_energy_usage((6, 3), next_config) == 40
        valid_next_moves = main.day_23_get_next_valid_moves(next_config)
        assert main.Move(4, 6) in valid_next_moves
        assert len(valid_next_moves) == 11
        next_config = main.day_23_make_move(next_config, (4, 6))
        assert next_config == \
               main.Configuration("...B.......", ["AB", "D", "CC", "AD"])
        assert main.day_23_get_energy_usage((4, 6), next_config) == 400
        valid_next_moves = main.day_23_get_next_valid_moves(next_config)
        assert len(valid_next_moves) == 10
        next_config = main.day_23_make_move(next_config, (4, 5))
        assert next_config == \
               main.Configuration("...B.D.....", ["AB", "", "CC", "AD"])
        assert main.day_23_get_energy_usage((4, 5), next_config) == 3000
        assert len(main.day_23_get_next_valid_moves(next_config)) == 6
        next_config = main.day_23_make_move(next_config, (3, 4))
        assert next_config == \
               main.Configuration(".....D.....", ["AB", "B", "CC", "AD"])
        assert main.day_23_get_energy_usage((3, 4), next_config) == 30
        next_config = main.day_23_make_move(next_config, (2, 4))
        assert next_config == \
               main.Configuration(".....D.....", ["A", "BB", "CC", "AD"])
        next_config = main.day_23_make_move(next_config, (8, 7))
        assert main.day_23_get_energy_usage((8, 7), next_config) == 2000
        next_config = main.day_23_make_move(next_config, (8, 9))
        assert main.day_23_get_energy_usage((8, 9), next_config) == 3
        assert next_config == \
               main.Configuration(".....D.D.A.", ["A", "BB", "CC", ""])
        next_config = main.day_23_make_move(next_config, (7, 8))
        assert main.day_23_get_energy_usage((7, 8), next_config) == 3000
        next_config = main.day_23_make_move(next_config, (5, 8))
        assert main.day_23_get_energy_usage((5, 8), next_config) == 4000
        assert next_config == \
               main.Configuration(".........A.", ["A", "BB", "CC", "DD"])
        next_config = main.day_23_make_move(next_config, (9, 2))
        assert main.day_23_is_completed(next_config)
        assert main.day_23_get_energy_usage((9, 2), next_config) == 8

    def test_no_move_out_of_empty_room(self):
        config = main.day_23_load_data(self.example_text)
        config = main.day_23_make_move(config, (2, 9))
        config = main.day_23_make_move(config, (2, 10))
        assert len(main.day_23_get_next_valid_moves(config)) == 15

    def test_stalled_configuration(self):
        config = main.day_23_load_data(self.example_text)
        config = main.day_23_make_move(config, (4, 1))
        config = main.day_23_make_move(config, (8, 3))
        config = main.day_23_make_move(config, (2, 9))
        config = main.day_23_make_move(config, (6, 7))
        config = main.day_23_make_move(config, (4, 5))
        print(config)
        assert main.day_23_get_next_valid_moves(config) == []
        oo_config = main.Day23Config(config)
        assert oo_config.generate_next_configs() == []
        assert oo_config.stalled
        stalls_in_3 = main.day_23_load_data(self.example_text)
        stalls_in_3 = main.day_23_make_move(stalls_in_3, (2, 3))
        stalls_in_3 = main.day_23_make_move(stalls_in_3, (8, 5))
        stalls_in_3 = main.day_23_make_move(stalls_in_3, (8, 7))
        print(stalls_in_3)
        assert main.day_23_get_next_valid_moves(stalls_in_3) == []
        s_oo_cfg = main.Day23Config(stalls_in_3)
        assert s_oo_cfg.generate_next_configs() == []
        assert s_oo_cfg.stalled

    def test_oo_approach(self):
        initial_config = main.Day23Config(main.day_23_load_data(self.example_text))
        next_configs = initial_config.generate_next_configs()
        assert len(next_configs) == 28
        d23_config = main.Day23Config.create_from_tuple(('...B.....C.A D CBAD', 28))
        print(d23_config._config)
        assert d23_config.energy_usage == 28
        hallway, rooms = d23_config._config
        assert hallway == '...B.....C.'
        assert rooms == ["A", "D", "CB", "AD"]
        d23_config = main.Day23Config.create_from_tuple(('.B.C.......A D CBAD', 28))
        print(d23_config._config)
        d23_config = main.Day23Config.create_from_tuple(('.....D.A...ABDCCB  ', 28))
        print(d23_config._config)

    def tst_part_one(self):
        assert main.day_23_part_one(self.example_text) == 12521
        # NB. Runtime: 04:25
        main.day_23_minimum_energy = (2 ** 31) - 1
        solution = main.day_23_part_one(Puzzle(23).get_text_input())
        print(f'Part One solution is {solution}')
        assert solution == 13066

    def test_set_up_for_part_two(self):
        new_rows = self.part_two_additional_text
        modified_text = main.day_23_insert_additional_rows(self.example_text, new_rows)
        assert modified_text == """#############
#...........#
###B#C#B#D###
  #D#C#B#A#
  #D#B#A#C#
  #A#D#C#A#
  #########
"""
        part_two_config = main.day_23_load_data(modified_text)
        oo_config = main.Day23Config(part_two_config, room_size=4)
        re_translated = main.Day23Config.create_from_tuple(oo_config.tuple_ise(), room_size=4)
        assert re_translated._config == oo_config._config
        moves = ((8, 10), (8, 0), (6, 9), (6, 7), (6, 1))
        for mv in moves:
            part_two_config = main.day_23_make_move(part_two_config, mv)
        config = main.Day23Config(part_two_config, room_size=4)
        cfg_tuple = config.tuple_ise()
        assert main.Day23Config.create_from_tuple(cfg_tuple, 4)._config == config._config

    def test_steps_for_part_two(self):
        start_text = main.day_23_insert_additional_rows(self.example_text,
                                                        self.part_two_additional_text)
        initial_config = main.day_23_load_data(start_text)
        next_moves = main.day_23_get_next_valid_moves(initial_config, room_size=4)
        assert len(next_moves) == 28
        config = main.day_23_make_move(initial_config, (8, 10))
        assert main.day_23_get_energy_usage((8, 10), config, room_size=4) == 3000
        next_moves = main.day_23_get_next_valid_moves(config, room_size=4)
        assert len(next_moves) == 24
        assert main.Move(8, 0) in next_moves
        config = main.day_23_make_move(config, (8, 0))
        assert main.day_23_get_energy_usage((8, 0), config, 4) == 10
        config = main.day_23_make_move(config, (6, 9))
        config = main.day_23_make_move(config, (6, 7))
        config = main.day_23_make_move(config, (6, 1))
        config = main.day_23_make_move(config, (4, 6))
        assert main.day_23_get_energy_usage((4, 6), config, 4) == 600
        next_config = initial_config
        energy_used = 0
        moves_to_completion = ((8, 10), (8, 0), (6, 9), (6, 7), (6, 1),
                               (4, 6), (4, 6), (4, 5), (4, 3), (5, 4),
                               (7, 4), (9, 4), (8, 6), (8, 9), (3, 8),
                               (2, 4), (2, 8), (2, 3), (1, 2), (0, 2),
                               (3, 8), (9, 2), (10, 8))
        for move in moves_to_completion:
            next_config = main.day_23_make_move(next_config, move)
            energy_used += main.day_23_get_energy_usage(move, next_config, 4)
        print(next_config)
        assert main.day_23_is_completed(next_config, room_size=4)
        assert energy_used == 44169

    def test_part_two(self):
        start_text = main.day_23_insert_additional_rows(self.example_text,
                                                        self.part_two_additional_text)
        # assert main.day_23_part_two(start_text) == 44169
        p2_raw_text = Puzzle(23).get_text_input()
        p2_full_text = main.day_23_insert_additional_rows(p2_raw_text,
                                                          self.part_two_additional_text)
        solution = main.day_23_part_two(p2_full_text)
        print(f'Part Two solution is {solution}')
        assert solution > 47328


class TestDayTwentyTwo:
    example_text = """on x=10..12,y=10..12,z=10..12
on x=11..13,y=11..13,z=11..13
off x=9..11,y=9..11,z=9..11
on x=10..10,y=10..10,z=10..10
"""
    larger_example = """on x=-20..26,y=-36..17,z=-47..7
on x=-20..33,y=-21..23,z=-26..28
on x=-22..28,y=-29..23,z=-38..16
on x=-46..7,y=-6..46,z=-50..-1
on x=-49..1,y=-3..46,z=-24..28
on x=2..47,y=-22..22,z=-23..27
on x=-27..23,y=-28..26,z=-21..29
on x=-39..5,y=-6..47,z=-3..44
on x=-30..21,y=-8..43,z=-13..34
on x=-22..26,y=-27..20,z=-29..19
off x=-48..-32,y=26..41,z=-47..-37
on x=-12..35,y=6..50,z=-50..-2
off x=-48..-32,y=-32..-16,z=-15..-5
on x=-18..26,y=-33..15,z=-7..46
off x=-40..-22,y=-38..-28,z=23..41
on x=-16..35,y=-41..10,z=-47..6
off x=-32..-23,y=11..30,z=-14..3
on x=-49..-5,y=-3..45,z=-29..18
off x=18..30,y=-20..-8,z=-3..13
on x=-41..9,y=-7..43,z=-33..15
on x=-54112..-39298,y=-85059..-49293,z=-27449..7877
on x=967..23432,y=45373..81175,z=27513..53682
"""

    def test_init(self):
        rows = main.day_22_load_data(self.example_text)
        assert len(rows) == 4
        assert len([*filter(lambda s: s[0] == "on", rows)]) == 3
        bigger_rows = main.day_22_load_data(self.larger_example, all_space=True)
        assert len(bigger_rows) == 22
        assert len([*filter(lambda s: s[0] == "off", bigger_rows)]) == 5
        assert len([*filter(lambda s: s[0] == "on", bigger_rows)]) == 17
        for data in (rows, bigger_rows):
            for r in data:
                numbers = r[1]
                assert all([numbers[(i * 2) + 1] >= v for i, v in enumerate(numbers[::2])])
                assert all([-50 <= n <= 50 or n < -50 or n > 50 for n in numbers])
                assert len(numbers) == 6

    def test_sets_etc(self):
        base_set = set(range(10))
        print(base_set)
        extended_set = base_set.union(range(8, 13))
        assert len(extended_set) == 13
        print(extended_set)
        reduced_set = extended_set.difference(range(6, 9))
        assert len(reduced_set) == 10
        assert 9 in reduced_set
        assert 6 not in reduced_set
        print(reduced_set)
        print(main.day_22_create_cuboid([10, 11] * 3))

    def tst_part_one(self):
        assert main.day_22_part_one(self.example_text) == 39
        assert main.day_22_part_one(self.larger_example) == 590784
        solution = main.day_22_part_one(Puzzle(22).get_text_input())
        print(f'Solution to Part One is {solution}')
        assert solution == 611176

    # TODO: Part Two
    # functions exist for: size of cuboid, do two cuboids overlap?, size of overlap
    # Traverse the list of instructions, keeping a running total of 'on' elements:
    # 1. Find all previously-considered cuboids that it overlaps
    # 2. for an 'on' instruction, add size of cuboid minus sizes of any overlaps
    #       with previous 'on' cuboids.  Ignore overlaps with 'off's?
    # 3. for an 'off' instruction, subtract sizes of any overlaps with 'on' cuboids
    #       already encountered, add for overlaps with previous 'off' cuboids, but
    #       ONLY WHERE THEY OVERLAP with previous 'on' cuboids
    # Looking at overlapping cuboids must be done in the order in which they were
    #   encountered
    # Need a function to return all previously-encountered overlapping cuboids
    #
    # RECURSIVE SOLUTION?
    #   Function gives all the overlaps(?) from a list of cuboids,
    #       eventually working for whole list
    #   End-case: one cuboid only, return its dimensions
    #   Second cuboid: +2, -2:1
    #   Third cuboid: +3, -3:1, -3:2, +3:2:1,
    #   How about only counting the 2-cuboid overlaps?
    #   Add a 4th cuboid to 1, 2, and 3, and you are ADDING ON:
    #   +4, -4:1, -4:2, +4:2:1, -4:3, +4:3:1, +4:3:2. -4:3:2:1
    #   So it's:
    #   1. Add the size of the cuboid itself
    #   2. Subtract any combinations of even-numbered length involving that cuboid
    #   3. Add any combinations of odd-numbered length involving said cuboid

    def test_part_two_functions(self):
        assert main.day_22_get_cuboid_size([1, 1, 1, 1, 1, 1]) == 1
        assert main.day_22_get_cuboid_size([-9, 10, -9, 10, -9, 10]) == 8000
        assert main.day_22_calc_overlap_size([1, 1, 1, 1, 1, 1],
                                             [-9, 10, -9, 10, -9, 10]) == 1
        assert main.day_22_calc_overlap_size([-9, -89765, -10, -89765, -9, -89765],
                                             [-9, 10, -9, 10, -9, 10]) == 0
        overlap_count = 0
        data = main.day_22_load_data(Puzzle(22).get_text_input(), all_space=True)
        for i, row in enumerate(data):
            this_cuboid = row[1]
            local_overlaps = []
            for j, other_row in enumerate(data):
                if j == i:
                    continue
                if main.day_22_calc_overlap_size(this_cuboid, other_row[1]) > 0:
                    local_overlaps.append(j)
                    overlap_count += 1
            # print(f'Cuboid {i} overlaps with: {len(local_overlaps)} cuboids')
        assert overlap_count == 3868
        cuboid_list = [[-9, 10, -9, 10, -9, 10], [-9, 10, -9, 10, -9, 10], [1, 1, 1, 1, 1, 1], [-9, -89765, -10, -89765, -9, -89765]]
        assert main.day_22_get_overlap_of_multiple_cuboids([("on", cbd)
                                                            for cbd in cuboid_list]) \
               == [0, -1]
        assert main.day_22_get_cuboid_size([0, -1]) == 0

    def test_part_two_solution_solves_part_one(self):
        data = main.day_22_load_data(self.example_text)
        assert main.day_22_try_for_part_two(data[:1]) == 27
        assert main.day_22_try_for_part_two(data[:2]) == 46
        assert main.day_22_try_for_part_two(data[:2] + [data[-1]]) == 46
        assert main.day_22_try_for_part_two(data[:3]) == 38
        assert main.day_22_try_for_part_two(data) == 39 #(46 when only treating ons)
        bigger_data = main.day_22_load_data(self.larger_example)
        assert main.day_22_solve_part_two(bigger_data) == 590784
        # just_out_of_interest = main.day_22_try_for_part_two(bigger_data)
        # print(f'number is {just_out_of_interest}')

    def tst_part_two(self):
        assert main.day_22_part_two(self.larger_example) == 2758514936282235
        # text = Puzzle(22).get_text_input()


class TestDayTwentyOne:
    example_input = """Player 1 starting position: 4
Player 2 starting position: 8"""

    def test_init(self):
        start_spots = tuple(main.day_21_load_data(self.example_input))
        assert start_spots == (4, 8)
        assert main.day_21_part_one(start_spots) == 739785

    def test_cycle(self):
        my_cycle = itertools.cycle([*range(1, 101)])
        n = 0
        while n < 100:
            dice_roll = [next(my_cycle) for _ in range(3)]
            # print(dice_roll)
            n = dice_roll[1]

    def test_part_one(self):
        text = Puzzle(21).get_text_input().strip('\n')
        solution = main.day_21_part_one(main.day_21_load_data(text))
        print(f'Part One solution is {solution}')
        assert solution == 678468

    # TODO: Part Two
    #   consider all possible journeys to 21 points (min. three turns)
    #    . . .  but it could take longer than three turns
    #   i.e. getting all valid sequences of dice totals
    #   each turn, there are 27 possible totals, ranging from 3 to 9
    #   player actions are totally independent of each other, so for each
    #   universe relating to one player's moves, all universes relating to the other
    #   player's moves need to be taken into account
    #   Only consider universes that end in a win for either player

    # Possible solution:
    #   1. get starting position
    #   Loop:
    #       2. 3 dice rolls could advance by any number in range 3-9
    #       3. advance position by each of these numbers
    #       4. increase score by new position
    #       5. record each of the 3-dice totals as a route
    #       5a.  . . . with current position/score pair
    # or can I use recursion?

    # when this list is complete, get the product of no. of permutations
    # of three dice that can get each number in each step of each item in the list

    #   have a list of routes under investigation and move to a completed list
    #   once 21 is reached?

    # player 1 / player 2: player 2 only wins in universes where they reach 21
    # without player 1 doing so.  The number of winning universes is complete
    # universes multiplied by player 1 non-winning universes evaluated up to the
    # number of turns player 2 has taken.  i.e. all universes of greater length
    # than number of turns, calculated up to player 2 turns

    # for player 1, it's winning universes multiplied by all player 2
    # universes of length > p1 turns - 1, evaluated up to p1 turns - 1

    # TODO: curtailing lists of universes -> duplication

    # think the no. of turns required can be anything from 3 to 9 (9 for player one only)

    def test_combinations(self):
        score_abundances = main.day_21_get_number_of_combinations_per_total()
        assert len(score_abundances) == 7
        assert sum(score_abundances.values()) == 27

    def test_roll_sequence_scoring(self):
        assert main.day_21_score_roll_sequence([1], 1) == 2
        assert main.day_21_score_roll_sequence([1], 10) == 1
        assert main.day_21_score_roll_sequence([1], 9) == 10
        assert main.day_21_score_roll_sequence([2, 9, 3], 8) == 21
        assert main.day_21_score_roll_sequence([3, 7, 3, 7, 3, 7, 3, 7, 3], 1) == 24
        assert main.day_21_score_roll_sequence([3, 3, 4, 3], 5) == 22
        # for start_pos in range(1, 11):
        #     all_routes = main.day_21_find_all_routes_to_21(start_pos)
        #     assert all([main.day_21_score_roll_sequence(s, start_pos) < 31 for s in all_routes])

    def test_routes_to_21(self):
        routes = main.day_21_find_all_routes_to_21(7)
        print(f'There are {len(routes)} routes to 21 from 7:')
        print(f'longest route is {max([len(r) for r in routes])}')

    def test_part_two(self):
        abundances = main.day_21_get_number_of_combinations_per_total()
        print(abundances)
        assert main.day_21_calc_universes_for_route([3, 3], abundances) == 1
        assert main.day_21_calc_universes_for_route([9] * 99, abundances) == 1
        assert main.day_21_calc_universes_for_route([6, 7, 8, 3], abundances) == 126
        assert main.day_21_part_two(self.example_input) == 444356092776315
        solution = main.day_21_part_two(Puzzle(21).get_text_input().strip('\n'))
        print(f'Solution to Part Two is {solution}')
        assert solution == 131180774190079


class TestDayTwenty:
    example_input = """..#.#..#####.#.#.#.###.##.....###.##.#..###.####..#####..#....#..#..##..##
#..######.###...####..#..#####..##..#.#####...##.#.#..#.##..#.#......#.###
.######.###.####...#.##.##..#..#..#####.....#.#....###..#.##......#.....#.
.#..#..##..#...##.######.####.####.#.#...#.......#..#.#.#...####.##.#.....
.#..#...##.#.##..#...##.#.##..###.#......#.#.......#.#.#.####.###.##...#..
...####.#..#..#.##.#....##..#.####....##...##..#...#......#.#.......#.....
..##..####..#...#.#.#...##..#.#..###..#####........#..####......#..#

#..#.
#....
##..#
..#..
..###"""

    def test_init(self):
        algorithm, image = main.day_20_load_data(Puzzle.convert_input(self.example_input, None))
        assert len(algorithm) == 512
        assert image[3][4] == '.'

    def test_get_neighbouring_pixel_values(self):
        algorithm, image = main.day_20_load_data(Puzzle.convert_input(self.example_input, None))
        print('\n')
        print('\n', image)
        assert main.day_20_surrounding_pixels(main.Point(-3, -3), image) == '.' * 9
        assert main.day_20_surrounding_pixels(main.Point(0, 0), image) == ('.' * 8) + '#'
        assert main.day_20_surrounding_pixels(main.Point(100, 100), image) == '.' * 9
        assert main.day_20_surrounding_pixels(main.Point(4, 3), image) == '.....##..'

    def test_image_conversion(self):
        algo, grid = main.day_20_load_data(Puzzle.convert_input(self.example_input, None))
        new_image = main.day_20_process_image(grid, algo)
        print('\n'.join(new_image))
        new_image = main.day_20_process_image(new_image, algo)
        print('\n'.join(new_image))

    def test_part_one(self):
        raw_data = Puzzle.convert_input(self.example_input, None)
        assert main.day_20_part_one(raw_data) == 35
        expected_image = """.......#.
.#..#.#..
#.#...###
#...##.#.
#.....#.#
.#.#####.
..#.#####
...##.##.
....###.."""
        algo, img = main.day_20_load_data(raw_data)
        for _ in range(2):
            img = main.day_20_process_image(img, algo)
        assert '\n'.join(img) == expected_image

        p1_data = Puzzle(20).input_as_list(None)
        alg, im = main.day_20_load_data(p1_data)
        assert len(alg) == 512
        assert len(im) == 100
        assert all([len(row) == 100 for row in im])
        solution = main.day_20_part_one(p1_data)
        print(f'Part One solution is {solution}')
        assert solution == 5347

    def test_part_two(self):
        p2_data = Puzzle(20).input_as_list(None)
        solution = main.day_20_part_two(p2_data)
        print(f'Part Two solution is {solution}')


class TestDayNineteen:
    # TODO:
    # 1. find a scheme to represent the 24 orientations (eg. +x-y, -z+x . . . )
    # 2. find a way to create a function that maps numbers in one set of co-ordinates to another
    #   or just manually create a dictionary of transformation per orientation
    # 3. set a scanner as having universal co-ordinates
    # 4. for all other scanners, run their numbers through each of the 24 conversion functions
    #   and see if there's one that gives at least 12 matches
    #   - match means there is a constant offset in each of the three dimensions
    # 5. while going along, maintain a growing list of beacon positions in universal co-ords

    # Consider a point (1, 2, 3) in universal co-ordinates, where +z is up, +x right, +y forwards:
    # Keeping z pointing up, rotating around four possible facings, this looks like:
    # +y:   (1, 2, 3)       = (x, y, z)
    # +x:   (-2, 1, 3)      = (-y, x, z) to convert back
    # -y:   (-1, -2, 3)     = (-x, -y, z)
    # -x:   (2, -1, 3)      = (y, -x, z)
    # Switching to -z pointing up (i.e. upside-down, mirror-image of above):
    #   NB. Facings are in YOUR WORLD, not universal co-ordinates
    # +y:   (-1, 2, -3)     = (-x, y, -z)
    # +x:   (-2, -1, -3)    = (-y, -x, -z)
    # -y:   (1, -2, -3)     = (x, -y, -z)
    # -x:   (2, 1, -3)      = (y, x, -z)
    # Switching to +x pointing up: +y remains straight ahead; +z is pointing left
    # +y:   (3, 2, -1)      = (-z, y, x)
    # +x:   (-2, 3, -1)     = (-z, -x, y)
    # -y:   (-3, -2, -1)      = (-z, -y, -x)
    # -x:   (2, -3, -1)       = (-z, x, -y)
    # Flipping so -x points up: +y remains straight ahead; +z becomes right
    # +y:   (-3, 2, 1)      = (z, y, -x)
    # +x:   (-2, -3, 1)     = (z, -x, -y)
    # -y:   (3, -2, 1)      = (z, -y, x)
    # -x:   (2, 3, 1)       = (z, x, y)
    # Switching to +y pointing up: +x remains right, +z behind
    # +y:   (1, 3, -2)      = (x, -z, y)
    # +x:   (-3, 1, -2)     = (y, -z, -x)
    # -y:   (-1, -3, -2)    = (-x, -z, -y)
    # -x:   (3, -1, -2)     = (-y, -z, x)
    # Flipping, so -y points up: +x remains to right, +z in front
    # +y:   (1, -3, 2)      = (x, z, -y)
    # +x:   (3, 1, 2)     = (y, z, x)
    # -y:   (-1, 3, 2)    = (-x, z, y)
    # -x:   (-3, -1, 2)     = (-y, z, -x)

    transforms = ["(x, y, z)", "(-y, x, z)", "(-x, -y, z)", "(y, -x, z)",
                  "(-x, y, -z)", "(-y, -x, -z)", "(x, -y, -z)", "(y, x, -z)",
                  "(-z, y, x)", "(-z, -x, y)", "(-z, -y, -x)", "(-z, x, -y)",
                  "(z, y, -x)", "(z, -x, -y)", "(z, -y, x)", "(z, x, y)",
                  "(x, -z, y)", "(y, -z, -x)", "(-x, -z, -y)", "(-y, -z, x)",
                  "(x, z, -y)", "(y, z, x)", "(-x, z, y)", "(-y, z, -x)"]

    five_scanners = """--- scanner 0 ---
-1,-1,1
-2,-2,2
-3,-3,3
-2,-3,1
5,6,-4
8,0,7

--- scanner 0 ---
1,-1,1
2,-2,2
3,-3,3
2,-1,3
-5,4,-6
-8,-7,0

--- scanner 0 ---
-1,-1,-1
-2,-2,-2
-3,-3,-3
-1,-3,-2
4,6,5
-7,0,8

--- scanner 0 ---
1,1,-1
2,2,-2
3,3,-3
1,3,-2
-4,-6,5
7,0,8

--- scanner 0 ---
1,1,1
2,2,2
3,3,3
3,1,2
-6,-4,-5
0,7,-8"""

    def test_five_scanner_example(self):
        lines = Puzzle.convert_input(self.five_scanners, None)
        observations = main.day_19_load_scanner_data(lines)
        assert len(observations) == 5
        assert all([len(data) == 6 for _, data in observations.items()])

        sc_0_points = observations[0]
        for scanner in [sc_points for sc_id, sc_points in observations.items() if sc_id > 0]:
            for tr in self.transforms:
                transformed_points = [main.day_19_do_transformation(pt, tr) for pt in scanner]
                if all([tp == op for tp, op in zip(transformed_points, sc_0_points)]):
                    print(f'There is a match! Overlapping points: {transformed_points}')
                    break

    def test_co_ordinate_transformations(self):
        test_grid = [[[0 if (x == y == layer == 0) else False for x in range(5)]
                      for y in range(5)]
                     for layer in range(5)]
        test_grid[3][2][1] = True

        new_x, new_y, new_z = main.day_19_do_transformation((3, -2, 1), "(z, -y, x)")
        assert test_grid[new_z][new_y][new_x]
        assert len(self.transforms) == len(set(self.transforms)) == 24
        transformed_points = []
        for tr in self.transforms:
            new_point = main.day_19_do_transformation((1,2,3), tr)
            transformed_points.append(new_point)
            # print(new_point)
        assert len(set(transformed_points)) == 24

    def test_play_around(self):
        axes, directions = ("z", "y", "x"), ("+", "-")
        facings = itertools.product(directions, axes, repeat=1)
        all_orientations = []
        for f in facings:
            _, fac_ax = f
            available_axes = filter(lambda a: a != fac_ax, axes)
            orientations = [f + pr for pr in itertools.product(directions, available_axes)]
            # print(["".join([*o]) for o in orientations])
            all_orientations += ["".join([*o]) for o in orientations]
        # print(all_orientations)
        assert len(all_orientations) == 24
        assert len(set(all_orientations)) == len(all_orientations)

    # def test_part_one(self):
    #     raw_eg = Puzzle.convert_input(Puzzle(19).get_text_from_filename("example19"), None)
    #     assert main.day_19_part_one(raw_eg) == 79
    #     # RUN-TIME 459.04s (0:07:39):
    #     p1_data = Puzzle(19).input_as_list(None)
    #     solution = main.day_19_part_one(p1_data)
    #     print(f'Part One solution is {solution}')
    #     assert solution == 306

    def test_manhattan_distance(self):
        point_1, point_2 = (main.Point3(*p) for p in ((1105, -1205, 1229), (-92, -2380, -20)))
        assert main.day_19_manhattan_distance(point_1, point_2) == 3621
        assert main.day_19_manhattan_distance(point_2, point_1) == 3621
        assert main.day_19_manhattan_distance(point_1, point_1) == 0

    def test_part_two(self):
        raw_eg = Puzzle.convert_input(Puzzle(19).get_text_from_filename("example19"), None)
        assert main.day_19_part_two(raw_eg) == 3621
        p2_data = Puzzle(19).input_as_list(None)
        solution = main.day_19_part_two(p2_data)
        print(f'Part Two solution is {solution}')


class TestDayEighteen:
    example_homework = """[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]
[[[5,[2,8]],4],[5,[[9,9],0]]]
[6,[[[6,2],[5,6]],[[7,6],[4,7]]]]
[[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]
[[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]
[[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]
[[[[5,4],[7,7]],8],[[8,3],8]]
[[9,3],[[9,9],[6,[4,9]]]]
[[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]
[[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]"""
    magnitude_examples = """[[1,2],[[3,4],5]] becomes 143.
[[[[0,7],4],[[7,8],[6,0]]],[8,1]] becomes 1384.
[[[[1,1],[2,2]],[3,3]],[4,4]] becomes 445.
[[[[3,0],[5,3]],[4,4]],[5,5]] becomes 791.
[[[[5,0],[7,4]],[5,5]],[6,6]] becomes 1137.
[[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]] becomes 3488."""

    def test_magnitudes(self):
        assert main.day_18_split_expression("[9,1]") == ("9", "1")
        assert main.day_18_split_expression("[[1,2],3]") == ("[1,2]", "3")
        assert main.day_18_split_expression("[9,[8,7]]") == ("9", "[8,7]")
        assert main.day_18_split_expression("[10,20]") == ("10", "20")
        assert main.day_18_magnitude("[9,1]") == 29
        input_result_pairs = {}
        for text_row in Puzzle.convert_input(self.magnitude_examples, None):
            expression, _, result = text_row.strip('.').partition(' becomes ')
            input_result_pairs[expression] = int(result)

        for k in input_result_pairs.keys():
            assert k.count('[') == k.count(']')
            assert main.day_18_magnitude(k) == input_result_pairs[k]
        print(input_result_pairs)

    def test_get_number_positions(self):
        assert main.day_18_get_number_positions("[2,]") == {1: 2}
        assert main.day_18_get_number_positions("[2,3]") == {1: 2, 3: 3}
        assert main.day_18_get_number_positions("[7,[6,[5,[4,[3,2]]]]]") == \
               {1: 7, 4: 6, 7: 5, 10: 4, 13: 3, 15: 2}
        assert main.day_18_get_number_positions("[11,2]") == {1: 11, 4: 2}
        assert main.day_18_get_number_positions("[[[[0,7],4],[15,[0,13]]],[1,1]]") == \
               {4: 0, 6: 7, 9: 4, 13: 15, 17: 0, 19: 13, 26: 1, 28: 1}

    def test_explosions(self):
        examples = {
            "[[[[[9,8],1],2],3],4]": "[[[[0,9],2],3],4]",
            "[7,[6,[5,[4,[3,2]]]]]": "[7,[6,[5,[7,0]]]]",
            "[[6,[5,[4,[3,2]]]],1]": "[[6,[5,[7,0]]],3]",
            "[[3,[2,[1,[7,3]]]],[6,[5,[4,[3,2]]]]]":
                main.day_18_reduce("[[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]]"),
            "[[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]]":
                main.day_18_reduce("[[3,[2,[8,0]]],[9,[5,[7,0]]]]"),
            # last two require two explosions, whereas the test cases are for single explosion
        }
        # reduction bug-fix test:
        # assert main.day_18_reduce("[[[[0,7],4],[7,[[8,4],9]]]]") == "[[[[0,7],4],[[7,8],[0,[6,7]]]]]"
        for test_expr, result in examples.items():
            assert main.day_18_reduce(test_expr) == result

        # explode_candidate = "[[[0,[[5,0],[[9,3]]]]]]"
        # assert main.day_18_reduce(explode_candidate) == "[[[5,[0,[[9,3]]]]]]"
        # level_6_explosion = "[[[[0,[[5,0],[[9,3]]]]]]"
        # main.day_18_reduce(level_6_explosion)

    def test_split(self):
        number = "[11,2]"
        assert main.day_18_reduce(number) == "[[5,6],2]"
        number = "[14,9]"
        assert main.day_18_reduce(number) == "[[7,7],9]"

    def test_addition(self):
        print('ADDITION')
        assert main.day_18_addition("[[[[4,3],4],4],[7,[[8,4],9]]]", "[1,1]") == \
               "[[[[0,7],4],[[7,8],[6,0]]],[8,1]]"
        number_list = [f"[{n},{n}]" for n in range(1, 5)]
        print(number_list)
        assert main.day_18_cumulative_addition(number_list) == "[[[[1,1],[2,2]],[3,3]],[4,4]]"
        num_list_5 = [f"[{n},{n}]" for n in range(1, 6)]
        assert main.day_18_cumulative_addition(num_list_5) == "[[[[3,0],[5,3]],[4,4]],[5,5]]"
        num_list_6 = [f"[{n},{n}]" for n in range(1, 7)]
        assert main.day_18_cumulative_addition(num_list_6) == "[[[[5,0],[7,4]],[5,5]],[6,6]]"
        example_list = Puzzle.convert_input(self.example_homework, None)
        # print('EXAMPLE LIST:')
        # for el in example_list:
        #     print(el)
        # assert main.day_18_cumulative_addition(example_list) ==\
        #        "[[[[6,6],[7,6]],[[7,7],[7,0]]],[[[7,7],[7,7]],[[7,8],[9,9]]]]"
        slightly_larger_eg = """[[[0,[4,5]],[0,0]],[[[4,5],[2,6]],[9,5]]]
[7,[[[3,7],[4,3]],[[6,3],[8,8]]]]
[[2,[[0,8],[3,4]]],[[[6,7],1],[7,[1,6]]]]
[[[[2,4],7],[6,[0,5]]],[[[6,8],[2,8]],[[2,1],[4,5]]]]
[7,[5,[[3,8],[1,4]]]]
[[2,[2,2]],[8,[8,1]]]
[2,9]
[1,[[[9,3],9],[[9,0],[0,7]]]]
[[[5,[7,4]],7],1]
[[[[4,2],2],6],[8,7]]"""
        sle_list = Puzzle.convert_input(slightly_larger_eg, None)
        assert main.day_18_cumulative_addition(sle_list) == \
               "[[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]]"

    def test_part_one(self):
        example_list = Puzzle.convert_input(self.example_homework, None)
        assert main.day_18_part_one(example_list) == 4140
        p1_list = Puzzle(18).input_as_list(None)
        solution = main.day_18_part_one(p1_list)
        print(f'Part One solution = {solution}')
        assert solution == 3654

    def test_part_two(self):
        example_list = Puzzle.convert_input(self.example_homework, None)
        assert main.day_18_part_two(example_list) == 3993
        p2_list = Puzzle(18).input_as_list(None)
        p2_solution = main.day_18_part_two(p2_list)
        print(f'Part Two solution = {p2_solution}')


class TestDaySeventeen:
    def test_trajectory_calcs(self):
        assert main.day_17_trajectory_step(Point(0, 0), Point(10, 10)) == ((10, 10), (9, 9))
        step_one = main.day_17_trajectory_step(Point(0, 0), Point(6, 6))
        assert main.day_17_trajectory_step(*step_one) == ((11, 11), (4, 4))
        target_area = ((20, 30), (-10, -5))
        assert main.day_17_velocity_hits_target(Point(7, 2), target_area)
        assert main.day_17_velocity_hits_target(Point(6, 3), target_area)
        assert main.day_17_velocity_hits_target(Point(9, 0), target_area)
        assert not main.day_17_velocity_hits_target(Point(17, -4), target_area)
        assert main.day_17_peak_given_starting_y_velocity(2) == 3
        assert main.day_17_peak_given_starting_y_velocity(3) == 6
        assert main.day_17_peak_given_starting_y_velocity(0) == 0

    def test_part_one(self):
        target_area = ((20, 30), (-10, -5))
        assert main.day_17_part_one(target_area) == 45
        big_target = ((119, 176), (-141, -84))
        solution = main.day_17_part_one(big_target)
        print(f'Part One solution: {solution}')
        assert solution == 9870

    def test_part_two(self):
        eg_target = ((20, 30), (-10, -5))
        assert main.day_17_part_two(eg_target) == 112
        big_target = ((119, 176), (-141, -84))
        solution = main.day_17_part_two(big_target)
        print(f'Part Two solution: {solution}')
        assert solution == 5523


class TestDaySixteen:
    part_one_examples = {
        "8A004A801A8002F478": 16,
        "620080001611562C8802118E34": 12,
        "C0015000016115A2E0802F182340": 23,
        "A0016C880162017C3686B18A3D4780": 31
    }
    literal_example = "D2FE28"

    def test_init(self):
        text = Puzzle(16).get_text_input().strip()
        # print(f'\nLength of text is {len(text)}')
        lines = text.count("\n")
        assert not lines
        # print(f'Text contains {lines} line breaks')
        # print(text)

    def test_hexadecimal_conversion(self):
        text = Puzzle(16).get_text_input().strip()
        print(bin(0x0F))
        print(main.day_16_hex_string_to_binary("17"))
        assert len(main.day_16_hex_string_to_binary(text)) == 4 * len(text)

    def test_version_number_and_type(self):
        assert main.day_16_get_version_no_from_binary_string('110100101111111000101000') == 6
        assert main.day_16_packet_is_operator("00111000000000000110111101000101001010010001001000000000")
        assert not main.day_16_packet_is_operator("110100101111111000101000")

    def test_packet_length_determination(self):
        literal_packet = main.Day16Packet(self.literal_example)
        assert literal_packet.get_length_in_hex_chars() == 6
        operator_packet = main.Day16Packet([*self.part_one_examples.keys()][0])
        # print(operator_packet.get_length_in_hex_chars())
        sub_packets_hex = "38006F45291200"
        operator_with_sub_packets = main.Day16Packet(sub_packets_hex)
        assert operator_with_sub_packets.get_length_in_hex_chars() == len(sub_packets_hex)
        packet_count_operator_hex = "EE00D40C823060"
        operator_with_packet_count = main.Day16Packet(packet_count_operator_hex)
        assert operator_with_packet_count.get_length_in_hex_chars() == len(packet_count_operator_hex)

    def test_read_packet_returns_bit_length_read(self):
        literal_packet = main.Day16Packet(self.literal_example)
        assert literal_packet.read_packet() == 21

    def test_version_sum(self):
        literal_packet = main.Day16Packet(self.literal_example)
        literal_packet.read_packet()
        assert literal_packet.get_version_sum() == 6
        packet_with_counted_sub_packets_hex = "EE00D40C823060"
        new_packet = main.Day16Packet(packet_with_counted_sub_packets_hex)
        new_packet.read_packet()
        assert new_packet.get_version_sum() == 7 + 2 + 4 + 1
        bit_length_packet_hex = "38006F45291200"
        bl_packet = main.Day16Packet(bit_length_packet_hex)
        bl_packet.read_packet()
        assert bl_packet.get_version_sum() == 1 + 6 + 2
        for raw_hex, expected_result in self.part_one_examples.items():
            pkt = main.Day16Packet(raw_hex)
            pkt.read_packet()
            print(f'{raw_hex}:')
            assert pkt.get_version_sum() == expected_result

    def test_part_one(self):
        hex_input = Puzzle(16).get_text_input()
        p1_packet = main.Day16Packet(hex_input)
        bits_read = p1_packet.read_packet()
        solution = p1_packet.get_version_sum()
        print(f'Part One solution is {solution}.  ({bits_read} bits read)')
        assert solution == 969

    def test_evaluation(self):
        encoded_binary = "101111111000101000"
        assert main.day_16_get_value_from_literal(encoded_binary) == 2021
        literal_packet = main.Day16Packet(self.literal_example)
        literal_packet.read_packet()
        assert literal_packet.get_value() == 2021
        operations_packets = {
            "C200B40A82": 3,
            "04005AC33890": 54,
            "880086C3E88112": 7,
            "CE00C43D881120": 9,
            "D8005AC2A8F0": 1,
            "F600BC2D8F": 0,
            "9C005AC2F8F0": 0,
            "9C0141080250320F1802104A08": 1,
        }
        for op_hex, expected in operations_packets.items():
            op_pkt = main.Day16Packet(op_hex)
            op_pkt.read_packet()
            assert op_pkt.get_value() == expected

    def test_part_two(self):
        p2_packet = main.Day16Packet(Puzzle(16).get_text_input())
        p2_packet.read_packet()
        solution = p2_packet.get_value()
        print(f'Part Two solution is {solution}')
        assert solution


class TestDayFifteen:
    example_input = """1163751742
1381373672
2136511328
3694931569
7463417111
1319128137
1359912421
3125421639
1293138521
2311944581"""

    def test_path_creation(self):
        example_grid = main.day_9_load_data(self.example_input)
        assert main.day_15_calc_path_total([[1, 3], [2, 1]], (True, False)) == (main.Point(1, 1), 4)
        assert main.day_15_calc_path_total([[1, 3], [2, 1]], (False,)) == (main.Point(0, 1), 2)
        assert main.day_15_cut_square(example_grid, main.Point(1, 1), 4) == \
               [[3, 8, 1, 3], [1, 3, 6, 5], [6, 9, 4, 9], [4, 6, 3, 4]]
        assert main.day_15_cut_square(example_grid, main.Point(8, 6), 4) == [[2, 1], [3, 9]]
        assert main.day_15_cut_square(example_grid, main.Point(9, 1), 4) == [[2]]
        assert main.day_15_cut_square(example_grid, main.Point(0, 7), 4) == [[3, 1, 2], [1, 2, 9], [2, 3, 1]]
        assert main.day_15_get_min_path_total_across_square([[1, 3], [2, 1]]) == (main.Point(0, 1), 2)
        sq = main.day_15_cut_square(example_grid, main.Point(0, 1), 3)
        print('3x3 square test:')
        assert main.day_15_get_min_path_total_across_square(sq) == (main.Point(0, 2), 5)
        print('last test')
        assert main.day_15_calc_path_total([[1, 3], [2, 1]], (True, True, True)) == \
               (main.Point(1, 0), 3)

    def test_part_one(self):
        example_grid = main.day_9_load_data(self.example_input)
        for _ in range(20):
            solution = min([main.day_15_part_one(example_grid) for _ in range(20)])
            print(f'Lowest path risk sum is {solution}')


class TestDayFourteen:
    example_input = """NNCB

CH -> B
HH -> N
CB -> H
NH -> C
HB -> C
HC -> B
HN -> C
NN -> C
BH -> H
NC -> B
NB -> B
BN -> B
BB -> N
BC -> B
CC -> N
CN -> C"""

    def test_init(self):
        lines = Puzzle.convert_input(self.example_input, None, True)
        template, insertions = main.day_14_load_data(lines)
        assert template == "NNCB"
        assert len(insertions) == 16
        assert insertions['BB'] == "N"
        assert insertions['CH'] == "B"

    def test_run_process(self):
        data = Puzzle.convert_input(self.example_input, None, True)
        original_polymer, insertions = main.day_14_load_data(data)
        assert main.day_14_run_insertion_process(original_polymer, insertions) == "NCNBCHB"
        assert len(main.day_14_iterate_insertion_process(original_polymer, insertions, 4)) == 49
        assert len(main.day_14_iterate_insertion_process(original_polymer, insertions, 10)) == 3073

    def test_part_one(self):
        data = Puzzle.convert_input(self.example_input, None, True)
        assert main.day_14_part_one(data) == 1588
        puzzle_data = Puzzle.convert_input(Puzzle(14).get_text_input())
        solution = main.day_14_part_one(puzzle_data)
        print(f'Part One solution: {solution}')
        assert solution == 2587

    def test_work_on_part_two(self):
        data = Puzzle.convert_input(self.example_input, None, True)
        start_polymer, insertions = main.day_14_load_data(data)

        def generate_pair_insertions(given_insertions: dict) -> dict:
            return {k: (k[0] + v, v + k[1]) for k, v in given_insertions.items()}

        assert main.day_14_create_initial_pair_dict(start_polymer) == {"NN": 1, "NC": 1, "CB": 1}
        replacement_dict = generate_pair_insertions(insertions)
        assert len(replacement_dict) == len(insertions)
        print(replacement_dict)
        for val in replacement_dict.values():
            one, two = val
            assert one in replacement_dict and two in replacement_dict

        first_run = main.day_14_growth_step({"NN": 1, "NC": 1, "CB": 1}, replacement_dict)
        assert sum([*first_run.values()]) == 6
        assert first_run["CN"] == 1
        assert first_run["HB"] == 1
        assert first_run["NN"] == 0
        second_run = main.day_14_growth_step(first_run, replacement_dict)
        assert sum([*second_run.values()]) == 12
        assert second_run["BB"] == 2
        assert second_run["BC"] == 2
        assert second_run["BH"] == 1
        assert second_run["HB"] == 0

        ten_iteration_run = main.day_14_create_initial_pair_dict(start_polymer)
        for it in range(10):
            ten_iteration_run = main.day_14_growth_step(ten_iteration_run, replacement_dict)
        assert sum([*ten_iteration_run.values()]) == 3073 - 1
        ltr_count = main.day_14_count_letters(ten_iteration_run, start_polymer[-1])
        assert ltr_count['B'] == 1749
        assert ltr_count['C'] == 298
        assert ltr_count['H'] == 161
        assert ltr_count['N'] == 865
        assert len(ltr_count) == 4

    def test_part_two(self):
        data = Puzzle.convert_input(self.example_input, None, True)
        length = 4
        for _ in range(40):
            length = (length * 2) -1
        print(f'Expected final length: {length}')
        assert main.day_14_part_two(data) == 2188189693529
        big_data = Puzzle.convert_input(Puzzle(14).get_text_input(), None, True)
        solution = main.day_14_part_two(big_data)
        print(f'Part Two solution is {solution}')
        assert solution == 3318837563123


class TestDayThirteen:
    example_input = """6,10
0,14
9,10
0,3
10,4
4,11
6,0
6,12
4,1
0,13
10,12
3,4
3,0
8,4
1,10
2,14
8,10
9,0

fold along y=7
fold along x=5"""

    def test_init(self):
        lines = Puzzle.convert_input(self.example_input, None, True)
        points, folds = main.day_13_load_data(lines)
        assert len(points) == 18
        assert len(folds) == 2
        main.day_13_print_page(points)

    def test_folding(self):
        rows = Puzzle.convert_input(self.example_input, None, True)
        dots, foldings = main.day_13_load_data(rows)
        folded = main.day_13_fold_up(dots, 7)
        main.day_13_print_page(folded)
        assert len(folded) == 17
        re_folded = main.day_13_fold_to_left(folded, 5)
        main.day_13_print_page(re_folded)
        assert len(re_folded) == 16

    def test_part_one(self):
        rows = Puzzle.convert_input(self.example_input, None, True)
        assert main.day_13_part_one(rows) == 0
        actual_input = Puzzle.convert_input(Puzzle(13).get_text_input(), None, True)
        solution = main.day_13_part_one(actual_input)
        print(f'Part One solution is {solution}')
        assert solution == 655

    def test_part_two(self):
        rows = Puzzle.convert_input(self.example_input, None, True)
        main.day_13_part_two(rows)
        actual_input = Puzzle.convert_input(Puzzle(13).get_text_input(), None, True)
        main.day_13_part_two(actual_input)


class TestDayTwelve:
    short_example = """start-A
start-b
A-c
A-b
b-d
A-end
b-end"""

    slightly_larger_example = """dc-end
HN-start
start-kj
dc-start
dc-HN
LN-dc
HN-end
kj-sa
kj-HN
kj-dc"""

    even_larger_example = """fs-end
he-DX
fs-he
start-DX
pj-DX
end-zg
zg-sl
zg-pj
pj-he
RW-he
fs-DX
pj-RW
zg-RW
start-pj
he-WI
zg-he
pj-fs
start-RW"""

    def test_init(self):
        raw_paths = Puzzle.convert_input(self.short_example, None)
        connections = main.day_12_load_all_connections(raw_paths)
        assert main.day_12_list_connected_nodes("start", connections) == ["A", "b"]
        assert main.day_12_list_connected_nodes("c", connections) == ["A"]
        assert main.day_12_list_connected_nodes("b", connections) == ["start", "A", "d", "end"]

    def test_all_valid_paths(self):
        raw = Puzzle.convert_input(self.short_example, None)
        assert main.day_12_part_one(raw) == 10
        larger = Puzzle.convert_input(self.slightly_larger_example, None)
        assert main.day_12_part_one(larger) == 19
        even_larger = Puzzle.convert_input(self.even_larger_example, None)
        assert main.day_12_part_one(even_larger) == 226

    def test_part_one(self):
        raw_inputs = Puzzle(12).input_as_list(None)
        solution = main.day_12_part_one(raw_inputs)
        print(f'Solution to Part One is {solution}')
        assert solution == 3738

    def test_more_complicated_rules(self):
        path_so_far = ['A', 'b', 'C', 'd', 'b']
        visited_small_caves = set(filter(lambda n: n.islower(), path_so_far))
        print(any([path_so_far.count(cave) > 1 for cave in visited_small_caves]))

        raw = Puzzle.convert_input(self.even_larger_example, None)
        assert main.day_12_part_two(raw) == 3509

    def test_part_two(self):
        raw_inputs = Puzzle(12).input_as_list(None)
        solution = main.day_12_part_two(raw_inputs)
        print(f'Part Two solution: {solution}')
        assert solution == 120506


class TestDayEleven:
    short_example = """11111
19991
19191
19991
11111"""
    example_input = """5483143223
2745854711
5264556173
6141336146
6357385478
4167524645
2176841721
6882881134
4846848554
5283751526"""

    def test_init(self):
        numbers = main.day_9_load_data(self.short_example)
        assert len(numbers), len(numbers[2]) == (5, 5)
        # print(numbers)
        # main.day_11_print_grid(numbers)

    def test_increment(self):
        array = [[0, 1], [8, 9]]
        assert main.day_11_global_energy_increment(array) == [[1, 2], [9, 10]]

    def test_neighbours(self):
        points = main.day_11_get_all_neighbours((0, 0), 9, 99)
        assert len(points) == 3
        central_points = main.day_11_get_all_neighbours((2, 2), 5, 5)
        assert len(central_points) == 8
        assert (2, 2) not in central_points
        bottom_corner_points = main.day_11_get_all_neighbours((4, 4), 5, 5)
        assert len(bottom_corner_points) == 3
        assert (4, 4) not in bottom_corner_points
        edge_points = main.day_11_get_all_neighbours((4, 3), 5, 5)
        assert len(edge_points) == 5
        assert (4, 3) not in edge_points

    def test_process(self):
        school = main.day_9_load_data(self.short_example)
        expected_after_step_one = main.day_9_load_data("""34543
40004
50005
40004
34543""")
        step_one = main.day_11_reset_flashers(main.day_11_flash_step(school))
        print('\nStep one, got:')
        main.day_11_print_grid(step_one)
        print('Expected:')
        main.day_11_print_grid(expected_after_step_one)
        assert step_one == expected_after_step_one
        assert main.day_11_flash_count == 9
        step_two = main.day_11_reset_flashers(main.day_11_flash_step(step_one))
        expected_after_step_two = main.day_9_load_data("""45654
51115
61116
51115
45654""")
        assert step_two == expected_after_step_two
        main.day_11_flash_count = 0
        fc = main.day_11_part_one(school, 2)
        assert fc == 9

    def test_part_one(self):
        starting_grid = main.day_9_load_data(self.example_input)
        assert main.day_11_part_one(starting_grid, 100) == 1656
        puzzle_grid = main.day_9_load_data(Puzzle(11).get_text_input())
        solution = main.day_11_part_one(puzzle_grid, 100)
        print(f'Solution to Part One is {solution}')

    def test_part_two(self):
        starting_grid = main.day_9_load_data(self.example_input)
        main.day_11_part_one(starting_grid, 200)
        main.day_11_it_happened = False
        puzzle_grid = main.day_9_load_data(Puzzle(11).get_text_input())
        main.day_11_part_one(puzzle_grid, 1000000)




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
