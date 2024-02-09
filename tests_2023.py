import pprint
import re
import aoc_2023 as a23
import library as lib
from itertools import cycle
import random


class TestDay19:
    example = """px{a<2006:qkq,m>2090:A,rfg}
pv{a>1716:R,A}
lnx{m>1548:A,A}
rfg{s<537:gd,x>2440:R,A}
qs{s>3448:A,lnx}
qkq{x<1416:A,crn}
crn{x>2662:A,R}
in{s<1351:px,qqz}
qqz{s>2770:qs,m<1801:hdj,R}
gd{a>3333:R,R}
hdj{m>838:A,pv}

{x=787,m=2655,a=1222,s=2876}
{x=1679,m=44,a=2067,s=496}
{x=2036,m=264,a=79,s=2244}
{x=2461,m=1339,a=466,s=291}
{x=2127,m=1623,a=2188,s=1013}"""

    def test_setup(self):
        eg_load = a23.day_19_load_inputs(self.example)
        pprint.pprint(eg_load)
        eg_instr, eg_dicts = eg_load
        assert len(eg_instr.split("\n")) == 11
        assert eg_dicts[0]["x"] == 787
        assert eg_dicts[1]["a"] == 2067
        assert eg_dicts[3]["s"] == 291
        main_instr, main_dicts = a23.day_19_load_inputs()
        assert len(main_instr.split("\n")) == 522
        assert all(main_instr.split("\n"))
        assert len(main_dicts) == 200
        assert main_dicts[0]["m"] == 2
        assert main_dicts[-1]["a"] == 1006
        # assert a23.day_19_part_one(self.example) == sum(sum(d.values()) for d in eg_dicts)
        # assert a23.day_19_part_one() == sum(sum(d.values()) for d in main_dicts)

    def test_following_instructions(self):
        inst, parts = a23.day_19_load_inputs(self.example)
        a23.day_19_process_instruction(parts[0], inst)
        # a23.day_19_process_instruction({}, inst, "qqz")

    def test_part_one(self):
        assert a23.day_19_part_one(self.example) == 19114
        p1_solution = a23.day_19_part_one()
        assert p1_solution < 325520
        lib.verify_solution(p1_solution)



class TestDay18:
    example = """R 6 (#70c710)
D 5 (#0dc571)
L 2 (#5713f0)
D 2 (#d2c081)
R 2 (#59c680)
D 2 (#411b91)
L 5 (#8ceee2)
U 2 (#caa173)
L 1 (#1b58a2)
U 2 (#caa171)
R 2 (#7807d2)
U 3 (#a77fa3)
L 2 (#015232)
U 2 (#7a21e3)"""
    conversion_table = """#70c710 = R 461937
#0dc571 = D 56407
#5713f0 = R 356671
#d2c081 = D 863240
#59c680 = R 367720
#411b91 = D 266681
#8ceee2 = L 577262
#caa173 = U 829975
#1b58a2 = L 112010
#caa171 = D 829975
#7807d2 = L 491645
#a77fa3 = U 686074
#015232 = L 5411
#7a21e3 = U 500254"""

    def test_setup(self):
        dug = a23.day_18_get_dug_out_points(self.example)
        assert lib.Point(0, 0) in dug
        assert len(dug) == 62

    def test_part_one(self):
        assert a23.day_18_part_one(self.example) == 62
        lib.verify_solution(a23.day_18_part_one(), 74074)

    def test_hex_conversion(self):
        for row in self.conversion_table.split("\n"):
            raw, _, exp_str = row.partition(" = ")
            drn, _, steps = exp_str.partition(" ")
            assert a23.day_18_hex_to_dig_instruction(raw) == (drn, int(steps))

    def test_finding_corners(self):
        eg_corners = a23.day_18_find_corners(self.example, for_part_two=False)
        assert len(eg_corners) == 14
        assert lib.Point(0, 0) in eg_corners
        assert lib.Point(5, 6) in eg_corners
        p2_corners_from_example = a23.day_18_find_corners(self.example)
        assert len(p2_corners_from_example) == 14
        assert lib.Point(0, 0) in eg_corners

    def test_finding_total_area_from_corners(self):
        eg_corners = a23.day_18_find_corners(self.example, for_part_two=False)
        assert a23.day_18_find_total_area(eg_corners) == 62
        """find whether the adjacency of corners in list means something . . .
            Scenarios on a new row:
            - active range is SNIPPED from NEXT row:
                <- corner 1 = start of active range, corner 2 within active range OR
                    corner 2 = end of active range, corner 1 within active range
            - active range is EXTENDED from THIS row:
                <- corner 1 < start of active range, corner 2 = start OR
                    corner 2 > end of active range, corner 1 = end
            - new active range is CREATED on THIS row:
                both corners are outside any active range
            - active range is DELETED on NEXT row:
                corners match end points of an active range
                -> """
        main_p1_corners = a23.day_18_find_corners(
            a23.Puzzle23(18).get_text_input().strip("\n"), for_part_two=False)
        assert a23.day_18_find_total_area(main_p1_corners) == a23.day_18_part_one()
        p2_corners = a23.day_18_find_corners(
            a23.Puzzle23(18).get_text_input().strip("\n"))

        a23.day_18_find_total_area(p2_corners)

    def draw_path(self, instructions: str):
        x, y = 0, 0
        path = {lib.Point(0, 0)}
        for instruction in instructions.split(","):
            direction, _, steps = instruction.partition(" ")
            for _ in range(int(steps)):
                x, y = a23.point_moves_2023[direction](lib.Point(x, y))
                path.add(lib.Point(x, y))
        for xx in range(min(p.x for p in path), max(p.x for p in path) + 1):
            print("")
            for yy in range(min(p.y for p in path), max(p.y for p in path) + 1):
                print("#" if lib.Point(xx, yy) in path else " ", end="")

    def test_more_complicated_structures(self):
        shape = "R 1,D 1,R 1,D 1,R 3,U 1,R 1,U 1,R 1,D 1,R 1,D 1,R 1,D 1,L 9,U 3"
        self.draw_path(shape)
        corners = a23.day_18_find_corners(shape.replace(",", "\n"), for_part_two=False)
        assert a23.day_18_find_total_area(corners) == 31
        shape = "R 1,D 1,R 1,D 1,R 1,D 1,R 3,U 2,L 1,U 1,L 1,U 1,R 3,D 5,L 7,U 4"
        self.draw_path(shape)
        corners = a23.day_18_find_corners(shape.replace(",", "\n"), for_part_two=False)
        assert a23.day_18_find_total_area(corners) == 38
        long_coalascence = "R 1,D 1,R 15,U 1,R 1,D 2,L 17,U 2"
        self.draw_path(long_coalascence)
        corners = a23.day_18_find_corners(long_coalascence.replace(",", "\n"), for_part_two=False)
        assert a23.day_18_find_total_area(corners) == 40
        n_shape = "R 5,D 3,L 1,U 1,L 3,D 1,L 1,U 3"
        self.draw_path(n_shape)
        corners = a23.day_18_find_corners(n_shape.replace(",", "\n"), for_part_two=False)
        assert a23.day_18_find_total_area(corners) == 22


    def test_edge_parity_method_also_works_for_day_10_p2(self):
        a23.day_10_load_map(a23.Puzzle23(10).get_text_input().strip("\n"))
        s_point = lib.Point(*[pt for pt, symbol in a23.day_10_map.items() if symbol == "S"][0])
        day_10_pipe, _ = a23.day_10_trace_pipe_from(s_point)

    def test_part_two(self):
        p2_solution = a23.day_18_part_two()
        assert p2_solution < 435094080194504
        lib.verify_solution(p2_solution, 112074045986829, part_two=True)


class TestDay17:
    example = """2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533"""

    def test_load(self):
        a23.day_17_load_city()
        city_map = a23.day_17_city
        dims = len(city_map), len(city_map[1])
        assert dims[0] == dims[1]
        print(f"City is {dims[0]} x {dims[1]} square")

    def test_neighbour_finding(self):
        a23.day_17_load_city(self.example)
        assert len(a23.day_17_find_neighbours((5, 5, "L", 1))) == 3
        assert len(a23.day_17_find_neighbours((5, 5, "L", 3))) == 3
        assert a23.day_17_find_neighbours((6, 0, "D", 2)) == [
            (6, 1, "R", 1), (7, 0, "D", 3), (6, -1, 'L', 1),
        ]

    def test_dijkstra_concept(self):
        a23.day_17_load_city(self.example)
        answer = a23.day_17_using_dijkstra()
        assert answer == 102

    def test_part_one(self):
        lib.verify_solution(a23.day_17_part_one(), 1155)
        """Takes over one minute"""

    def test_part_two(self):
        assert a23.day_17_part_two(self.example) == 94
        lib.verify_solution(a23.day_17_part_two(), correct=1283, part_two=True)
        """Takes close to 20 minutes"""


class TestDay16:
    example_grid = r""".|...\....
|.-.\.....
.....|-...
........|.
..........
.........\
..../.\\..
.-.-/..|..
.|....-|.\
..//.|...."""

    def test_load(self):
        grid = a23.day_16_load_grid(self.example_grid)
        assert len(grid) == 10
        assert all(len(rr) == 10 for rr in grid)
        big_grid = a23.day_16_load_grid()
        assert len(big_grid) == len(big_grid[1])
        assert all(len(row) == len(big_grid) for row in big_grid)
        print(f"Real grid is {len(big_grid)} x {len(big_grid)} square")

    def test_ray_tracing(self):
        a23.day_16_grid = a23.day_16_load_grid(self.example_grid)
        a23.day_16_trace_ray_until_exit(lib.Point(0, 6), "L")
        assert a23.day_16_energised == {lib.Point(0, 6), lib.Point(0, 5)}
        a23.day_16_trace_ray_until_exit(lib.Point(9, 4), "L")
        assert len(a23.day_16_energised) == 4
        assert all(pt in a23.day_16_energised for pt in (lib.Point(9, 3), lib.Point(9, 4)))
        a23.day_16_trace_ray_until_exit(lib.Point(1, 1), "U")
        assert len(a23.day_16_energised) == 6
        assert all(pt in a23.day_16_energised for pt in (lib.Point(1, 1), lib.Point(0, 1)))
        a23.day_16_trace_ray_until_exit(lib.Point(3, 0), "R")
        assert len(a23.day_16_energised) == 6 + 18
        assert all(pt in a23.day_16_energised for pt in (lib.Point(0, 8), lib.Point(9, 8)))
        assert lib.Point(3, 9) not in a23.day_16_energised
        a23.day_16_trace_ray_until_exit(lib.Point(7, 2), "L")
        assert len(a23.day_16_energised) == 6 + 18 + 3

    def test_part_one(self):
        assert a23.day_16_part_one(self.example_grid) == 46
        p1_solution = a23.day_16_part_one()
        assert p1_solution > 6840
        lib.verify_solution(p1_solution, 7951)

    def test_part_two(self):
        assert a23.day_16_part_two(self.example_grid) == 51
        p2_solution = a23.day_16_part_two()
        assert p2_solution > 7951   # expecting > part one solution
        lib.verify_solution(p2_solution, part_two=True)


class TestDay15:
    examples = """rn=1 becomes 30.
cm- becomes 253.
qp=3 becomes 97.
cm=2 becomes 47.
qp- becomes 14.
pc=4 becomes 180.
ot=9 becomes 9.
ab=5 becomes 197.
pc- becomes 48.
pc=6 becomes 214.
ot=7 becomes 231."""

    def create_examples_table(self) -> dict:
        return {
            re.search(r"\S+\s", row).group().strip():
                int(re.search(r" \d+", row).group()[1:])
            for row in self.examples.split("\n")
        }

    def test_load(self):
        pprint.pprint(self.create_examples_table())

    def test_hashing(self):
        assert a23.day_15_single_character_hash("H") == 200
        assert a23.day_15_single_character_hash("A", 200) == 153
        assert a23.day_15_single_character_hash("S", 153) == 172
        assert a23.day_15_single_character_hash("H", 172) == 52
        assert a23.day_15_string_hash("HASH") == 52
        for word, expected in self.create_examples_table().items():
            assert a23.day_15_string_hash(word) == expected

    def test_part_one(self):
        test_string = ",".join(self.create_examples_table().keys())
        print(test_string)
        assert a23.day_15_part_one(test_string) == 1320
        lib.verify_solution(a23.day_15_part_one(), 506437)

    def test_part_two(self):
        test_string = ",".join(self.create_examples_table().keys())
        assert a23.day_15_part_two(test_string) == 145
        p2_solution = a23.day_15_part_two()
        lib.verify_solution(p2_solution, 288521, part_two=True)


class TestDay14:
    example = """O....#....
O.OO#....#
.....##...
OO.#O....O
.O.....O#.
O.#..O.#.#
..O..#O..O
.......O..
#....###..
#OO..#...."""
    after_tilt_to_north = """OOOO.#.O..
OO..#....#
OO..O##..O
O..#.OO...
........#.
..#....#.#
..O..#.O.O
..O.......
#....###..
#....#...."""
    after_spin_cycles = """After 1 cycle:
.....#....
....#...O#
...OO##...
.OO#......
.....OOO#.
.O#...O#.#
....O#....
......OOOO
#...O###..
#..OO#....

After 2 cycles:
.....#....
....#...O#
.....##...
..O#......
.....OOO#.
.O#...O#.#
....O#...O
.......OOO
#..OO###..
#.OOO#...O

After 3 cycles:
.....#....
....#...O#
.....##...
..O#......
.....OOO#.
.O#...O#.#
....O#...O
.......OOO
#...O###.O
#.OOO#...O"""

    def test_load(self):
        board = a23.day_14_load_board(self.example)
        assert board[(1, 1)] == "#"
        assert board[(2, 1)] == "#"
        assert (3, 1) not in board
        assert board[(7, 10)] == "O"
        assert len(board) == 35

    def test_functions(self):
        triangular_expected = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
        assert all([a23.day_14_triangular(ii) == nn
                    for ii, nn in enumerate(triangular_expected)])
        assert a23.day_14_rock_load(1, 2) == 1
        assert a23.day_14_rock_load(2, 5) == 7
        assert a23.day_14_rock_load(4, 11) == 34

    def test_tilting(self):
        tilted = a23.day_14_load_after_tilt_board_north(a23.day_14_load_board(self.example))
        assert tilted == 136
        starting_board = a23.day_14_load_board(self.example)
        for dn in "NWSE":
            a23.day_14_tilt_board(starting_board, dn)
        n_tilted_board = a23.day_14_tilt_board(a23.day_14_load_board(self.example), "N")
        assert a23.day_14_load_board(self.after_tilt_to_north) == n_tilted_board
        assert a23.day_14_north_support_beams_loading(n_tilted_board) == 136
        board = a23.day_14_load_board(self.example)
        for new_layout in re.finditer(r":\n(.+\n){10}", self.after_spin_cycles):
            layout = new_layout.group().strip(":\n")
            expected_board = a23.day_14_load_board(layout)
            for rotation in "NWSE":
                board = a23.day_14_tilt_board(board, rotation)
            assert board == expected_board
            assert len([v for v in board.values() if v == "O"]) == 18
            assert len([v for v in board.values() if v == "#"]) == 17

    def test_part_one(self):
        p1_solution = a23.day_14_part_one()
        assert p1_solution > 96275
        lib.verify_solution(p1_solution, 107430)

    def test_p2_attempt_to_find_pattern(self):
        board = a23.day_14_load_board(self.example)
        for spin in range(1, 100):
            for rotation in "NWSE":
                board = a23.day_14_tilt_board(board, rotation)
            print(f"{spin=}\tload={a23.day_14_north_support_beams_loading(board)}")

    def test_part_two(self):
        assert a23.day_14_part_two(self.example) == 64
        p2_solution = a23.day_14_part_two()
        lib.verify_solution(p2_solution, 96317, part_two=True)



class TestDay13:
    example = """#.##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#"""

    def test_setup(self):
        both_grids = a23.day_13_load_grids(self.example)
        assert a23.day_13_find_vertical_mirror(both_grids[0]) == 5
        assert a23.day_13_find_horizontal_mirror(both_grids[0]) == 0
        assert a23.day_13_find_vertical_mirror(both_grids[1]) == 0
        assert a23.day_13_find_horizontal_mirror(both_grids[1]) == 4
        real_grids = a23.day_13_load_grids()
        assert a23.day_13_find_horizontal_mirror(real_grids[4]) == 13
        for rg in real_grids:
            v, h = a23.day_13_find_vertical_mirror(rg), a23.day_13_find_horizontal_mirror(rg)
            print(v, h, "*****" if all(val == 0 for val in (v, h)) else "")
        assert all(
            a23.day_13_find_vertical_mirror(gr) or a23.day_13_find_horizontal_mirror(gr)
            for gr in real_grids)

    def test_part_one(self):
        assert a23.day_13_part_one(self.example) == 405
        p1_solution = a23.day_13_part_one()
        assert p1_solution > 20442
        lib.verify_solution(p1_solution, 36041)

    def test_part_two(self):
        """Could easily cheat and say smudge is outside the area that is
            relevant to reflection ("old reflection line won't NECESSARILY
            continue being valid").  But assume to begin with that it's in
            that area.  Will in that case have to calculate where the line is
            before working through all points in the area, presumably"""
        assert a23.day_13_part_two(self.example) == 400
        p2_solution = a23.day_13_part_two()
        lib.verify_solution(p2_solution, part_two=True)


class TestDay12:
    example = """???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1"""

    """Are there instances in the real input where 
        there could be zero possible arrangements?
        RULES:
        - don't move so that a # would cover a known spot (.)
        - distancing: leave at least one space between every group
        - fixed ends: can you nail some of the groups down? 
                If whole length of hashes shown
                or if a .# boundary is shown?
        - work in from each end?
        - can be certain which group it is if it's within its own length + 1 from an end"""

    """January 2024 approach:
        - use social distancing first to calculate min, max starts
        - crawl through each possible start zone
        - could also reverse the string to re-use a method in opposite direction
            (rather than calculating ends using a different method)
        """

    def test_dict_solution(self):
        dictionary = {0: set(), 1: set(), 2: set(), 3: set(), 5: {0}, 9: {1}}
        assert a23.day_12_get_contiguous_range(dictionary, 0) == (0, 4)
        assert all(a23.day_12_get_contiguous_range(dictionary.keys(), n) == (0, 4)
                   for n in range(4))
        assert a23.day_12_get_contiguous_range(dictionary.keys(), 5) == (5, 6)
        a23.day_12_dictionary_based_solution(".??..??...?##.", [1, 1, 3])
        example_rows = self.example.split("\n")
        for er in example_rows:
            a23.day_12_dictionary_based_solution(*a23.day_12_get_record_details(er))

    def test_explore_real_input(self):
        assert a23.day_12_count_possible_arrangements("#?.???????#????#???.", [1, 1, 12]) == 6
        assert a23.day_12_dictionary_based_solution('??#??#?#?#?????#?.#.', [1, 1, 12, 1]) == 2
        assert a23.day_12_dictionary_based_solution("???#?????????.????.?", [11, 1, 2]) == 6
        assert a23.day_12_dictionary_based_solution(".##.?#??.#.?#", [2, 1, 1, 1]) == 1 # reddit
        data = a23.Puzzle23(12).get_text_input()
        results = {}
        for i, row in enumerate(data.strip("\n").split("\n")):
            springs, groups = a23.day_12_get_record_details(row)
            print(f"===== Input row = {i + 1} =====")
            if i == 29:
                assert a23.day_12_dictionary_based_solution(springs, groups) == 1
            if i == 77:
                assert a23.day_12_count_possible_arrangements(springs, groups) == 7
            if i == 647:
                assert a23.day_12_dictionary_based_solution(springs, groups) == 1
                assert a23.day_12_count_possible_arrangements(springs, groups) == 1
            results[i + 1] = a23.day_12_count_possible_arrangements(springs, groups)
            assert results[i + 1] == a23.day_12_count_possible_arrangements(
                f".{springs}.", groups
            )
            print(f"Returns {results[i + 1]}")
            # assert results[i + 1] == a23.day_12_dictionary_based_solution(springs, groups)
        pprint.pprint(results)
        print(f"Most possible arrangements: {max(results.items(), key=lambda it: it[1])}")
        # todo: interesting that surrounding each springs row with '.' gives slightly
        #        different result (did for a while, but went away while fixing other issues)

    def test_brute_force(self):
        assert a23.day_12_group_limits_by_social_distancing("?", [1]) == [(0, 0)]
        assert a23.day_12_group_limits_by_social_distancing("??", [1]) == [(0, 1)]
        assert a23.day_12_group_limits_by_social_distancing("??", [2]) == [(0, 0)]
        assert a23.day_12_group_limits_by_social_distancing("???", [2]) == [(0, 1)]
        assert a23.day_12_count_possible_arrangements("#", [1]) == 1
        assert a23.day_12_count_possible_arrangements("?", [1]) == 1
        assert a23.day_12_count_possible_arrangements(".", [1]) == 0
        assert a23.day_12_count_possible_arrangements("??", [1]) == 2
        assert a23.day_12_count_possible_arrangements("?#?", [1]) == 1
        assert a23.day_12_count_possible_arrangements("?#?#?", [3]) == 1
        assert a23.day_12_count_possible_arrangements("????", [1]) == 4
        assert a23.day_12_count_possible_arrangements("???", [1, 1]) == 1
        assert a23.day_12_count_possible_arrangements("????", [1, 1]) == 3
        assert a23.day_12_count_possible_arrangements("?????", [1, 1]) == 6
        assert a23.day_12_count_possible_arrangements("?????", [1, 2]) == 3
        assert a23.day_12_count_possible_arrangements("??..???", [1, 2]) == 4
        assert a23.day_12_count_possible_arrangements("??#?", [1]) == 1
        assert a23.day_12_count_possible_arrangements("?.??????#?", [1, 3, 1]) == 4

    def test_part_one(self):
        assert a23.day_12_part_one(self.example) == 21
        p1_solution = a23.day_12_part_one()
        assert 9245 > p1_solution > 6828
        # 8216 is also incorrect
        # 7949 is also incorrect
        # 7836 also . . . all based on previous attempts
        assert p1_solution not in (8216, 7949, 7836, 7931, 7748, 7569, 7317, 7321)
        lib.verify_solution(p1_solution, 7307)

    def test_part_two(self):
        assert a23.day_12_part_two(self.example) == 525152
        lib.verify_solution(a23.day_12_part_two(), part_two=True)
        """works, but takes 21 minutes"""


class TestDay11:
    example = """...#......
.......#..
#.........
..........
......#...
.#........
.........#
..........
.......#..
#...#....."""

    def test_load(self):
        gal = a23.day_11_find_galaxies(self.example)
        assert len(gal) == 9
        assert (0, 0) not in gal
        assert (0, 3) in gal
        assert gal == {(0, 3), (1, 7), (2, 0), (4, 6), (5, 1), (6, 9), (8, 7),
                       (9, 0), (9, 4)}
        pairs = [*a23.day_11_all_galaxy_pairs(gal)]
        assert len(pairs) == 36
        assert lib.manhattan_distance((11, 5), (6, 1)) == 9

    def test_method_steps(self):
        expanded_universe = a23.day_11_expand(a23.day_11_find_galaxies(self.example))
        assert len(expanded_universe) == 9
        assert (11, 5) in expanded_universe
        assert (6, 1) in expanded_universe
        assert (4, 6) not in expanded_universe
        assert expanded_universe == {(0, 4), (1, 9), (2, 0), (5, 8), (6, 1),
                                     (7, 12), (10, 9), (11, 0), (11, 5)}
        assert lib.manhattan_distance((0, 4), (10, 9)) == 15
        assert lib.manhattan_distance((2, 0), (7, 12)) == 17
        assert lib.manhattan_distance((11, 0), (11, 5)) == 5

    def test_part_one(self):
        assert a23.day_11_part_one(self.example) == 374
        lib.verify_solution(a23.day_11_part_one(), 10165598)

    def test_part_two(self):
        assert a23.day_11_part_two(self.example, 10) == 1030
        assert a23.day_11_part_two(self.example, 100) == 8410
        attempt = a23.day_11_part_two()
        assert attempt < 678729486878
        lib.verify_solution(attempt, part_two=True)


class TestDay10:
    simple_loop = """.....
.S-7.
.|.|.
.L-J.
....."""
    complex_loop = """..F7.
.FJ|.
SJ.L7
|F--J
LJ..."""
    complete_grid = """7-F7-
.FJ|7
SJLL7
|F--J
LJ.LJ"""
    part_two_examples = ["""...........
.S-------7.
.|F-----7|.
.||.....||.
.||.....||.
.|L-7.F-J|.
.|..|.|..|.
.L--J.L--J.
...........""", """..........
.S------7.
.|F----7|.
.||OOOO||.
.||OOOO||.
.|L-7F-J|.
.|..||..|.
.L--JL--J.
..........""", """.F----7F7F7F7F-7....
.|F--7||||||||FJ....
.||.FJ||||||||L7....
FJL7L7LJLJ||LJ.L-7..
L--J.L7...LJS7F-7L7.
....F-J..F7FJ|L7L7L7
....L7.F7||L7|.L7L7|
.....|FJLJ|FJ|F7|.LJ
....FJL-7.||.||||...
....L---J.LJ.LJLJ...""", """FF7FSF7F7F7F7F7F---7
L|LJ||||||||||||F--J
FL-7LJLJ||||||LJL-77
F--JF--7||LJLJ7F7FJ-
L---JF-JLJ.||-FJLJJ7
|F|F-JF---7F7-L7L|7|
|FFJF7L7F-JF7|JL---7
7-L-JL7||F7|L7F-7F7|
L.L7LFJ|||||FJL7||LJ
L7JLJL-JLJLJL--JLJ.L"""]

    def test_load(self):
        a23.day_10_load_map(self.simple_loop)
        assert a23.day_10_map[(1, 1)] == "S"
        assert a23.day_10_map[(3, 2)] == "-"
        assert (4, 1) not in a23.day_10_map

    def test_loop_finding(self):
        a23.day_10_load_map(self.simple_loop)
        assert a23.day_10_get_next_location_round((1, 2), (1, 1)) == lib.Point(1, 3)
        a23.day_10_load_map(self.complex_loop)
        assert a23.day_10_get_next_location_round((2, 3), (1, 3)) == lib.Point(2, 4)
        assert a23.day_10_get_next_location_round((4, 0), (4, 1)) == lib.Point(3, 0)
        assert a23.day_10_get_next_location_round((1, 2), (1, 1)) == lib.Point(0, 2)
        assert a23.day_10_is_connectable((0, 2), (0, 3))
        assert a23.day_10_is_connectable((0, 3), (0, 2))
        assert not a23.day_10_is_connectable((2, 2), (0, 2))

    def test_part_one(self):
        assert a23.day_10_part_one(self.simple_loop) == 4
        assert a23.day_10_part_one(self.complex_loop) == 8
        assert a23.day_10_part_one(self.complete_grid) == 8
        solution = a23.day_10_part_one()
        assert solution > 7101
        lib.verify_solution(solution, 7102)

    def test_find_method_for_p2(self):
        a23.day_10_load_map(self.part_two_examples[0])
        s_point = a23.day_10_get_starting_point()
        pipe, _ = a23.day_10_trace_pipe_from(s_point)
        assert len(pipe) == 46
        assert pipe[0] == s_point
        inside_edge_points = a23.day_10_all_inside_edge_points(pipe)
        print(inside_edge_points)
        assert len(inside_edge_points) == 4
        assert lib.Point(3, 3) not in inside_edge_points
        assert lib.Point(6, 3) in inside_edge_points
        a23.day_10_load_map(self.part_two_examples[1])
        compressed_pipe, _ = a23.day_10_trace_pipe_from(s_point)
        assert len(a23.day_10_all_inside_edge_points(compressed_pipe)) == 4
        a23.day_10_load_map(self.part_two_examples[2])
        s_point = a23.day_10_get_starting_point()
        larger_pipe, _ = a23.day_10_trace_pipe_from(s_point)
        assert larger_pipe[1] == lib.Point(4, 13)
        enclosed = a23.day_10_all_inside_edge_points(larger_pipe)
        print(enclosed)
        assert len(enclosed) == 8

    def test_part_two(self):
        assert a23.day_10_part_two(self.part_two_examples[2]) == 8
        assert a23.day_10_part_two(self.part_two_examples[3]) == 10
        p2_solution = a23.day_10_part_two()
        assert p2_solution > 229
        lib.verify_solution(p2_solution, 363, part_two=True)


class TestDay9:
    example = """0 3 6 9 12 15
1 3 6 10 15 21
10 13 16 21 30 45"""

    def validate_sequences(self, in_list: [], output: []):
        if isinstance(output[0], int):
            assert len(output) == len(in_list) - 1
        else:
            assert all([len(values) == len(output[previous_row]) - 1
                        for previous_row, values in enumerate(output[1:])])
            assert all(v == 0 for v in output[-1])

    def test_setup(self):
        for seq in a23.day_9_load_sequences(self.example):
            diff = a23.day_9_diff_sequence(seq)
            self.validate_sequences(seq, diff)
            to_zero = a23.day_9_fully_decompose(seq)
            self.validate_sequences(None, to_zero)
        expected_example_results = [18, 28, 68]
        for index, sequence in enumerate(a23.day_9_load_sequences(self.example)):
            assert a23.day_9_predict_next_number(
                a23.day_9_fully_decompose(sequence)) == expected_example_results[index]

    def test_explore_data(self):
        sequences = a23.day_9_load_sequences()
        all_numbers = {i for s in sequences for i in s}
        print(f"There are {len(sequences)} sequences in the input file")
        print(f"Numbers range from {min(all_numbers)} to {max(all_numbers)}")
        print(f"Sequence lengths from {min(len(s) for s in sequences)} to "
              f"{max(len(s) for s in sequences)}")
        most_decomposed = 0
        for seq in sequences:
            self.validate_sequences(seq, a23.day_9_diff_sequence(seq))
            decomposed = a23.day_9_fully_decompose(seq)
            self.validate_sequences(None, decomposed)
            most_decomposed = max(most_decomposed, len(decomposed))
        print(f"At most, the full decomposition has {most_decomposed} steps.")

    def test_part_one(self):
        assert a23.day_9_part_one(self.example) == 114
        lib.verify_solution(a23.day_9_part_one(), 2008960228)

    def test_part_two(self):
        assert a23.day_9_part_two(self.example) == 2
        lib.verify_solution(a23.day_9_part_two(), 1097, part_two=True)


class TestDay8:
    example_1 = """RL

AAA = (BBB, CCC)
BBB = (DDD, EEE)
CCC = (ZZZ, GGG)
DDD = (DDD, DDD)
EEE = (EEE, EEE)
GGG = (GGG, GGG)
ZZZ = (ZZZ, ZZZ)"""
    example_2 = """LLR

AAA = (BBB, BBB)
BBB = (AAA, ZZZ)
ZZZ = (ZZZ, ZZZ)"""
    p2_example = """LR

11A = (11B, XXX)
11B = (XXX, 11Z)
11Z = (11B, XXX)
22A = (22B, XXX)
22B = (22C, 22C)
22C = (22Z, 22Z)
22Z = (22B, 22B)
XXX = (XXX, XXX)"""

    def test_setup(self):
        dirs, node_map = a23.day_8_load_instructions(self.example_2)
        dir_cycle = cycle(dirs)
        assert "".join([next(dir_cycle) for _ in range(6)]) == "LLRLLR"
        assert len(node_map) == 3
        assert node_map["BBB"] == ("AAA", "ZZZ")
        assert a23.day_8_traverse(dirs, node_map) == 6
        assert a23.day_8_traverse(*a23.day_8_load_instructions(self.example_1)) == 2

    def test_part_one(self):
        lib.verify_solution(a23.day_8_part_one(), 12737)

    def test_p2_figuring_out(self):
        text = a23.Puzzle23(8).get_text_input()
        directions = text[:text.index("\n\n")]
        print(f"There are {len(directions)} L/R steps")
        _, route_map = a23.day_8_load_instructions()
        print(f"There are {len([*filter(lambda k: k[-1] == 'Z', route_map.keys())])} "
              f"points ending in Z and ", end="")
        starts = [*filter(lambda k: k[-1] == 'A', route_map.keys())]
        print(f"{len(starts)} starting points")

        def continuous_traversal(location: str, duration: int = 10_000):
            print(location)
            looping_directions = cycle(directions)
            visited_zs = set()
            good_steps = set()
            last_good, repeat = 0, 0
            for step in range(duration):
                location = route_map[location][{"L": 0, "R": 1}[next(looping_directions)]]
                if location[-1] == "Z":
                    if location in visited_zs:
                        # print(f"Revisits {location} on step {step}")
                        repeat = step - last_good
                        good_steps.add(step)
                        last_good = step
                    visited_zs.add(location)
            print(f"{repeat=}, possible first visit: {min(good_steps) - repeat}, "
                  f"{repeat % len(directions)}")
            return good_steps

        step_sets = [continuous_traversal(st, 100_000) for st in starts]
        result = step_sets[0]
        for ss in step_sets[1:]:
            result = result.intersection(ss)
        print(result)
        print(*(tuple((k, v)) for k, v in route_map.items() if k[-1] == "A"))

    def test_part_two(self):
        # assert a23.day_8_part_two(self.p2_example) == 6
        solution = a23.day_8_part_two()
        assert solution < 4913067224909339
        lib.verify_solution(solution, part_two=True)


class TestDay7:
    examples = """32T3K 765
T55J5 684
KK677 28
KTJJT 220
QQQJA 483"""

    def test_setup_and_scoring(self):
        assert len(a23.day_7_card_values) == 13
        assert a23.day_7_score_hand("AAAAA") == 61_212_121_212
        assert a23.day_7_score_hand("AA8AA") == 51_212_061_212
        assert a23.day_7_score_hand("23332") > 40_000_000_000
        assert a23.day_7_score_hand("TTT98") == 30_808_080_706
        assert a23.day_7_score_hand("23432") == 20_001_020_100
        assert a23.day_7_score_hand("A23A4") == 11_200_011_202
        assert a23.day_7_score_hand("23456") == 1_020_304
        assert a23.day_7_score_hand("33332") > a23.day_7_score_hand("2AAAA")
        assert a23.day_7_score_hand("77888") > a23.day_7_score_hand("77788")

    def test_part_one(self):
        assert a23.day_7_part_one(self.examples) == 6440
        lib.verify_solution(a23.day_7_part_one(), 249638405)

    def test_joker_scoring(self):
        assert a23.day_7_score_hand_with_jokers("QJJQ2") > 50_000_000
        print(f'{a23.day_7_score_hand_with_jokers("JKKK2"):,}')
        assert a23.day_7_score_hand_with_jokers("JKKK2") < \
               a23.day_7_score_hand_with_jokers("QQQQ2")

    def test_part_two(self):
        assert a23.day_7_part_two(self.examples) == 5905
        assert a23.day_7_part_two() < 250405856
        lib.verify_solution(a23.day_7_part_two())


class TestDay6:
    example = """Time:      7  15   30
Distance:  9  40  200"""

    def test_load(self):
        assert a23.day_6_load_race_data(self.example) == (
            (7, 15, 30), (9, 40, 200)
        )

    def test_part_one(self):
        assert a23.day_6_part_one(self.example) == 288
        lib.verify_solution(a23.day_6_part_one(), 1624896)

    def test_part_two(self):
        assert a23.day_6_part_two(self.example) == 71503
        lib.verify_solution(a23.day_6_part_two(), correct=32583852, part_two=True)


class TestDay5:
    example = """seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4"""

    def test_load_data(self):
        a23.day_5_load_data(self.example)

    def test_build_method(self):
        assert a23.day_5_range_function((50, 98, 2))(10) == -38
        assert a23.day_5_range_function((52, 50, 48))(999_999) == 1_000_001
        conversions = {0: 0, 1: 1, 49: 49, 50: 52, 51: 53, 96: 98, 97: 99, 98: 50, 99: 51}
        for conv in conversions:
            assert a23.day_5_convert([(50, 98, 2), (52, 50, 48)], conv) == conversions[conv]

    def test_part_one(self):
        assert a23.day_5_part_one(self.example) == 35
        lib.verify_solution(a23.day_5_part_one(), 196167384)

    def test_prep_p2(self):
        seeds, _ = a23.day_5_load_data()
        total = 0
        for i, s in enumerate(seeds[::2]):
            s1, s2 = (seeds[i * 2], seeds[(i * 2) + 1])
            print(s1, s2)
            total += s2
        print(f"{total=}")
        # brute force would take ~four hours
        assert a23.day_5_reverse_range_function((50, 98, 2))(51) == 99
        assert a23.day_5_reverse_range_function((52, 50, 48))(999_999) == 999_997
        conversions = {0: 0, 1: 1, 49: 49, 52: 50, 53: 51, 98: 96, 99: 97, 50: 98, 51: 99}
        for conv in conversions:
            assert a23.day_5_reverse_convert([(50, 98, 2), (52, 50, 48)], conv) == conversions[conv]

    def test_cumulative_offset_approach_for_p2(self):
        range_details, mappings = a23.day_5_load_data(self.example)
        seed_ranges = [(r, range_details[(i * 2) + 1])
                       for i, r in enumerate(range_details[::2])]
        cumulative_offsets = {sr: 0 for sr in seed_ranges}
        print(mappings)
        offsets_so_far = a23.day_5_cumulative_offsets_next_step(
            cumulative_offsets, mappings["soil"])
        assert offsets_so_far == {(79, 14): 2, (55, 13): 2}
        offsets_so_far = a23.day_5_cumulative_offsets_next_step(
            offsets_so_far, mappings["fertilizer"])
        assert offsets_so_far == {(79, 14): 2, (55, 13): 2}
        offsets_so_far = a23.day_5_cumulative_offsets_next_step(
            offsets_so_far, mappings["water"])
        assert offsets_so_far == {(79, 14): 2, (55, 4): -2, (59, 9): 2}
        offsets_so_far = a23.day_5_cumulative_offsets_next_step(
            offsets_so_far, mappings["light"])
        assert offsets_so_far == {(79, 14): -5, (55, 4): -9, (59, 9): -5}
        offsets_so_far = a23.day_5_cumulative_offsets_next_step(
            offsets_so_far, mappings["temperature"])
        assert offsets_so_far == {(79, 3): -1, (82, 11): -37, (55, 4): 27, (59, 9): 31}
        offsets_so_far = a23.day_5_cumulative_offsets_next_step(
            offsets_so_far, mappings["humidity"])
        assert offsets_so_far == {(79, 3): -1, (82, 11): -36, (55, 4): 27, (59, 9): 31}
        offsets_so_far = a23.day_5_cumulative_offsets_next_step(
            offsets_so_far, mappings["location"])
        final_expected = {(79, 3): 3, (82, 10): -36, (92, 1): -32, (55, 4): 31,
                          (59, 3): 35, (62, 4): -6, (66, 2): 31}
        assert offsets_so_far == final_expected

    def test_part_two(self):
        assert a23.day_5_part_two(self.example) == 46
        # assert a23.day_5_ranges_overlap((50, 96), (95, 161))
        # assert a23.day_5_ranges_overlap((78, 99), (47, 79))
        # assert not a23.day_5_ranges_overlap((50, 95), (95, 161))
        # assert not a23.day_5_ranges_overlap((78, 99), (47, 78))
        # assert not a23.day_5_ranges_overlap((7, 99), (547, 83778))
        # assert not a23.day_5_ranges_overlap((547, 83778), (7, 99))
        print("Part Two with real input\n=======================")
        p2_solution = a23.day_5_part_two()
        assert p2_solution > 92743746
        lib.verify_solution(p2_solution, 125742456, part_two=True)


class TestDay4:
    example = """Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11"""

    def test_part_one(self):
        assert a23.day_4_score_card(self.example.split("\n")[0]) == 8
        assert a23.day_4_part_one(self.example) == 13
        lib.verify_solution(a23.day_4_part_one(), 18619)

    def test_part_two(self):
        assert a23.day_4_part_two(self.example) == 30
        lib.verify_solution(a23.day_4_part_two(), correct=8063216, part_two=True)


class TestDay3:
    example = """467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598.."""

    def test_grid(self):
        grid = self.example.split("\n")
        # print(a23.Puzzle23(3).input_as_list(str))

    def test_tdd_for_part_two(self):
        grid = self.example.split("\n")
        assert a23.day_3_find_all_asterisks(grid) == [
            lib.Point(1, 3),
            lib.Point(4, 3),
            lib.Point(8, 5),
        ]
        real_stars = a23.day_3_find_all_asterisks(
            a23.Puzzle23(3).get_text_input().strip("\n").split("\n")
        )
        print(f"In the real thing there are {len(real_stars)} asterisks")
        assert a23.day_3_part_two(self.example) == 467835

    def test_part_one(self):
        assert a23.day_3_part_one(self.example) == 4361
        lib.verify_solution(a23.day_3_part_one(), 520135)

    def test_part_two(self):
        lib.verify_solution(a23.day_3_part_two(), part_two=True)


class TestDay2:
    example = """Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green"""

    def test_setup(self):
        games_dict = a23.day_2_load_game_text(self.example)
        assert len(games_dict) == 5
        print(games_dict)
        assert a23.day_2_part_one(self.example) == 8

    def test_part_one(self):
        lib.verify_solution(a23.day_2_part_one())

    def test_part_two(self):
        assert a23.day_2_part_two(self.example) == 2286
        lib.verify_solution(a23.day_2_part_two(), part_two=True)


class TestDay1:
    example = """1abc2
pqr3stu8vwx
a1b2c3d4e5f
treb7uchet"""
    p2_example = """two1nine
eightwothree
abcone2threexyz
xtwone3four
4nineeightseven2
zoneight234
7pqrstsixteen"""

    def test_part_one(self):
        assert a23.day_1_part_one(self.example) == 142
        lib.verify_solution(a23.day_1_part_one(), 55607)

    def test_p2_searches(self):
        all_lines = self.p2_example.split("\n")
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
        st = r"\d|" + "|".join(numbers.keys())
        print(st)
        assert a23.day_1_find_number(all_lines[0], search_term=st) == "two"
        assert a23.day_1_find_number(all_lines[0], False, search_term=st) == "nine"
        st = r"(?=(\d|" + "|".join(numbers.keys()) + r"))"
        assert a23.day_1_find_number("5eight34sckhhxrtwonem", False, st) == "one"

    def test_part_two(self):
        assert a23.day_1_part_two(self.p2_example) == 281
        proposed = a23.day_1_part_two()
        assert proposed < 55309
        lib.verify_solution(proposed, 55291, part_two=True)
