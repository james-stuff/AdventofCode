import aoc_2024 as a
import library as lib
import timeit


class TestDay20:
    eg = """###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############"""

    def test_setup(self):
        grid = a.day_20_load(self.eg)
        assert "S" in grid.values()
        assert "E" in grid.values()
        assert len(grid) == 85
        track = a.day_20_form_track(grid)
        # print(track)
        assert len(track) == 85 == len(set(track))

    def test_part_one(self):
        cheats = a.day_20_part_one(self.eg)
        # assert cheats.count(2) == 14
        # assert cheats.count(4) == 14
        # assert cheats.count(6) == 2
        # assert cheats.count(8) == 4
        # assert cheats.count(10) == 2
        # assert cheats.count(12) == 3
        # assert cheats.count(20) == 1
        # assert cheats.count(36) == 1
        # assert cheats.count(38) == 1
        # assert cheats.count(40) == 1
        # assert cheats.count(64) == 1
        assert cheats == 0
        solution = a.day_20_part_one()
        lib.verify_solution(solution, correct=1499)

    def test_part_two(self):
        assert a.day_20_part_two(self.eg) == 285
        solution = a.day_20_part_two()
        assert 611267 < solution < 16402130 # also, it took 32-38 sec
        lib.verify_solution(solution, part_two=True)


class TestDay19:
    eg = """r, wr, b, g, bwu, rb, gb, br

brwrr
bggr
gbbr
rrbgbr
ubwu
bwurrg
brgr
bbrgwb"""

    def test_part_one(self):
        assert a.day_19_part_one(self.eg) == 6
        # a.day_19_known_possible = set()
        solution = a.day_19_part_one()
        assert solution < 304
        lib.verify_solution(solution, correct=300)

    def test_part_two(self):
        assert a.day_19_part_two(self.eg) == 16
        p2_solution = a.day_19_part_two()
        assert p2_solution < 960732611151017
        lib.verify_solution(p2_solution, correct=624802218898092, part_two=True)

    def test_gorgeous(self):
        #https://www.reddit.com/r/adventofcode/comments/1hhlb8g/2024_day_19_solutions/
        from functools import cache

        @cache
        def possible(design):
            if not design:
                return 1
            return sum(possible(design[len(towel):])
                       for towel in towels if design.startswith(towel))

        towels,  designs = a.day_19_load()
        pos = [possible(design) for design in designs]
        print(sum(map(bool, pos)))
        print(sum(pos))


class TestDay18:
    eg = """5,4
4,2
4,5
3,0
2,1
6,3
2,4
1,5
0,6
3,3
2,6
5,1
1,2
5,5
2,5
6,5
1,4
0,4
6,4
1,1
6,1
1,0
0,5
1,6
2,0"""

    def test_part_one(self):
        assert a.day_18_part_one(self.eg) == 22
        lib.verify_solution(a.day_18_part_one(), 250)

    def test_part_two(self):
        # assert a.day_18_part_two(self.eg) == "6,1"
        lib.verify_solution(a.day_18_part_two(), "56,8", part_two=True)


class TestDay17:
    eg = """Register A: 729
Register B: 0
Register C: 0

Program: 0,1,5,4,3,0"""
    p2_eg = """Register A: 2024
Register B: 0
Register C: 0

Program: 0,3,5,4,3,0"""

    def test_part_one(self):
        assert a.day_17_part_one(self.eg) == "4,6,3,5,6,3,5,2,1,0"
        lib.verify_solution(a.day_17_part_one())

    def test_example_test_cases(self):
        reg = {"A": 0, "B": 0, "C": 9}
        prog = [2, 6]
        a.day_17_execute(reg, prog)
        assert reg["B"] == 1

        reg["A"] = 10
        assert a.day_17_execute(reg, [5,0,5,1,5,4]) == "0,1,2"

        reg["A"] = 2024
        assert a.day_17_execute(reg, [0,1,5,4,3,0]) == "4,2,5,6,7,7,7,7,3,1,0"
        assert reg["A"] == 0

        reg["B"] = 29
        a.day_17_execute(reg, [1, 7])
        assert reg["B"] == 26

        reg["B"] = 2024
        reg["C"] = 43690
        a.day_17_execute(reg, [4, 0])
        assert reg["B"] == 44354

    def p2_alt_exec(self, reg: {}) -> str:
        output = ""
        while reg["A"] > 0:
            reg["B"] = reg["A"] % 8 ^ 5
            reg["C"] = reg["A"] // 2 ** reg["B"]
            reg["B"] = reg["B"] ^ 6 ^ reg["C"]
            output += f"{reg['B'] % 8}"
            reg["A"] //= 8
        return ",".join(output)

    def test_p2_dev(self):
        r, p = a.day_17_load()
        saved_r = {**r}
        assert self.p2_alt_exec(r) == a.day_17_execute(saved_r, p)
        assert a.day_17_step_deconstruct(2) == 33940147 % 8
        """Key insight: puzzle is asking for the LOWEST POSITIVE
            initial value for register A"""

    def test_part_two(self):
        r, p = a.day_17_load(self.p2_eg)
        r["A"] = 117440
        assert a.day_17_execute(r, p) == ",".join(f"{n}" for n in p)
        _, p2p = a.day_17_load()
        # a.day_17_predict_register_a(p2p)
        # assert a.day_17_predict_register_a(p) == 117440
        # assert a.day_17_part_two(self.p2_eg) == 117440
        solution = a.day_17_part_two()
        reg, prog = a.day_17_load()
        reg["A"] = solution
        print(f"{solution=}")
        assert solution > 107408875647753
        assert a.day_17_execute(reg, prog) == ",".join(f"{nn}" for nn in prog)
        lib.verify_solution(solution, part_two=True)

        """The 3,0 at end of program causes it to loop back to start,
            until register A reaches zero and the program terminates.
            Each 0 opcode divides register A value by 2^operand.
            Each 5 opcode appends to output (5, 5) gives %8 value of B
            There is one output and one division of Register A by 8
            per loop"""


class TestDay16:
    small_eg = """###############
#.......#....E#
#.#.###.#.###.#
#.....#.#...#.#
#.###.#####.#.#
#.#.#.......#.#
#.#.#####.###.#
#...........#.#
###.#.#####.#.#
#...#.....#.#.#
#.#.#.###.#.#.#
#.....#...#.#.#
#.###.#.#.#.#.#
#S..#.....#...#
###############"""
    big_eg = """#################
#...#...#...#..E#
#.#.#.#.#.#.#.#.#
#.#.#.#...#...#.#
#.#.#.#.###.#.#.#
#...#.#.#.....#.#
#.#.#.#.#.#####.#
#.#...#.#.#.....#
#.#.#####.#.###.#
#.#.#.......#...#
#.#.###.#####.###
#.#.#...#.....#.#
#.#.#.#####.###.#
#.#.#.........#.#
#.#.#.#########.#
#S#.............#
#################"""

    def test_p1_dev(self):
        m = a.day_16_load_maze(self.small_eg)
        assert lib.Point(0, 0) not in m
        assert lib.Point(2, 2) not in m
        assert lib.Point(12, 1) in m
        assert len(m) > 20
        print(f"{len(m)=}")
        assert a.day_16_start == lib.Point(13, 1)
        assert a.day_16_end == lib.Point(1, 13)

    def test_part_one(self):
        assert a.day_16_part_one(self.small_eg) == 7036
        a.day_16_distances_table = {}
        assert a.day_16_part_one(self.big_eg) == 11048
        a.day_16_distances_table = {}
        lib.verify_solution(a.day_16_part_one(), 115500)

    def test_part_two(self):
        a.day_16_part_one(self.small_eg)
        assert a.day_16_part_two(self.small_eg) == 45
        a.day_16_distances_table = {}
        a.day_16_part_one(self.big_eg)
        assert a.day_16_part_two(self.big_eg) == 64
        a.day_16_distances_table = {}
        a.day_16_part_one()
        solution = a.day_16_part_two()
        assert solution > 545
        lib.verify_solution(solution, 679, part_two=True)


class TestDay15:
    small_eg = """########
#..O.O.#
##@.O..#
#...O..#
#.#.O..#
#...O..#
#......#
########

<^^>>>vv<v>>v<<"""
    large_eg = """##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v
><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<
<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^
^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><
^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^
>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^
<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>
^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^"""
    widened_large_eg = """####################
##....[]....[]..[]##
##............[]..##
##..[][]....[]..[]##
##....[]@.....[]..##
##[]##....[]......##
##[]....[]....[]..##
##..[][]..[]..[][]##
##........[]......##
####################"""

    def test_p1_dev(self):
        wh = a.day_15_load_warehouse(self.small_eg[:self.small_eg.index("\n\n")])
        dims = (min(filter(lambda k: wh[k] == "\n", wh)) + 1,
                  self.small_eg[:self.small_eg.index("\n\n")].count("\n") + 1)
        expected = {(20, "^"): 11, (20, "<"): 20, (11, ">"): 13}
        for args, result in expected.items():
            assert a.day_15_look_ahead(wh, dims, *args) == result

    def test_part_one(self):
        assert a.day_15_part_one(self.small_eg) == 2028
        assert a.day_15_part_one(self.large_eg) == 10092
        lib.verify_solution(a.day_15_part_one(), 1486930)

    def test_p2_dev(self):
        assert a.day_15_p2_widen_warehouse(
            a.day_15_load_warehouse(self.large_eg[:self.large_eg.index("\n\n")])
        ) == a.day_15_load_warehouse(self.widened_large_eg)
        gps_eg = """##########
##...[]...
##........"""
        assert a.day_15_gps_score(
            a.day_15_load_warehouse(gps_eg), gps_eg.index("\n") + 1
        ) == 105
        test_moves = "^^<<<vvvvvvvvvvv"
        a.day_15_part_two(
            self.large_eg[:self.large_eg.index("\n\n")] + "\n\n" + test_moves,
            start_pos=92
        )

    def test_part_two(self):
        assert a.day_15_part_two(self.large_eg) == 9021
        lib.verify_solution(a.day_15_part_two(), 1492011, part_two=True)


class TestDay14:
    eg = """p=0,4 v=3,-3
p=6,3 v=-1,-3
p=10,3 v=-1,2
p=2,0 v=2,-1
p=0,0 v=1,3
p=3,0 v=-2,-2
p=7,6 v=-1,-3
p=3,0 v=-1,-2
p=9,3 v=2,3
p=7,3 v=-1,2
p=2,4 v=2,-3
p=9,5 v=-3,-3"""

    def test_part_one(self):
        data = a.day_14_load_data(self.eg)
        print(data[-2])
        i, v = data[-2]
        assert a.day_14_robot_position(
            i, v, (11, 7), 0) == tuple((2, 4))
        assert a.day_14_robot_position(
            i, v, (11, 7), 1) == tuple((4, 1))
        assert a.day_14_robot_position(
            i, v, (11, 7), 2) == tuple((6, 5))
        assert a.day_14_robot_position(
            i, v, (11, 7), 3) == tuple((8, 2))
        assert a.day_14_robot_position(
            i, v, (11, 7), 4) == tuple((10, 6))
        assert a.day_14_robot_position(
            i, v, (11, 7), 5) == tuple((1, 3))
        assert a.day_14_part_one(self.eg) == 12
        assert a.day_14_part_one() > 223036638
        lib.verify_solution(a.day_14_part_one(), 224357412)

    def test_part_two(self):
        lib.verify_solution(a.day_14_part_two(), part_two=True)


class TestDay13:
    eg = """Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400

Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176

Button A: X+17, Y+86
Button B: X+84, Y+37
Prize: X=7870, Y=6450

Button A: X+69, Y+23
Button B: X+27, Y+71
Prize: X=18641, Y=10279"""

    def test_prize_grab(self):
        assert a.day_13_prize_cost({
            "a": (94, 34),
            "b": (22, 67),
            "p": (8400, 5400),
        }) == 280

    def test_part_one(self):
        assert a.day_13_part_one(self.eg) == 480
        lib.verify_solution(a.day_13_part_one(), 34787)

    def test_part_two(self):
        solution = a.day_13_part_two()
        assert solution < 158668878553983
        lib.verify_solution(solution, part_two=True)


class TestDay12:
    def test_part_one(self):
        # print(a.day_12_load_map_data())
        # print(a.day_12_all_contiguous(1, a.day_12_load_map_data()))
        # print(a.day_12_find_regions(a.day_12_load_map_data()))
        lib.verify_solution(a.day_12_part_one())

    def test_part_two(self):
        lib.verify_solution(a.day_12_part_two(), part_two=True)


class TestDay11:
    eg_stones = "0 1 10 99 999"
    main_eg = "125 17"

    def test_p1_dev(self):
        assert a.day_11_load_stones(self.eg_stones) == [0, 1, 10, 99, 999]
        assert all(isinstance(i, int) for i in a.day_11_load_stones())
        assert a.day_11_load_stones()[-1] == 91
        after_1 = "1 2024 1 0 9 9 2021976"
        st = a.day_11_load_stones(self.eg_stones)
        blinked = a.day_11_blink(st)
        assert len(blinked) >= len(st)
        assert all(isinstance(i, int) for i in blinked)
        assert blinked == a.day_11_load_stones(after_1)
        main_st = a.day_11_load_stones(self.main_eg)
        cumulative_expected = [
            "253000 1 7",
            "253 0 2024 14168",
            "512072 1 20 24 28676032",
            "512 72 2024 2 0 2 4 2867 6032",
            "1036288 7 2 20 24 4048 1 4048 8096 28 67 60 32",
            "2097446912 14168 4048 2 0 2 4 40 48 2024 40 48 80 96 2 8 6 7 6 0 3 2",
        ]
        for expected in cumulative_expected:
            main_st = a.day_11_blink(main_st)
            assert main_st == a.day_11_load_stones(expected)

    def test_part_one(self):
        assert a.day_11_part_one(self.main_eg) == 55312
        lib.verify_solution(a.day_11_part_one(), 193899)

    def test_p2_exploration(self):
        list_1 = a.day_11_load_stones(self.main_eg)
        assert a.day_11_part_one(self.main_eg) == sum([
            a.day_11_part_one(s) for s in self.main_eg.split(" ")
        ])
        """Each stone's behaviour is completely independent"""
        """Could I get a count for each number if it's blinked 35
            times, then run for 40 times, counting how many of each
            number there are in the list, and just sum their products?"""

    def test_part_two(self):
        lib.verify_solution(a.day_11_part_two(), 229682160383225, part_two=True)


class TestDay10:
    eg_map = """89010123
78121874
87430965
96549874
45678903
32019012
01329801
10456732"""

    def test_part_one(self):
        assert a.day_10_part_one(self.eg_map) == 36
        lib.verify_solution(a.day_10_part_one(), 482)

    def test_part_two(self):
        assert a.day_10_part_two(self.eg_map) == 81
        lib.verify_solution(a.day_10_part_two(), 1094, part_two=True)


class TestDay9:
    trivial_eg = "12345"
    eg_disk_map = "2333133121414131402"

    def test_part_one(self):
        dm = a.day_9_load_raw_map(self.eg_disk_map)
        assert dm[0] == (0, 2)
        assert dm[9] == (40, 2)
        assert a.day_9_part_one(self.eg_disk_map) == 1928
        solution = a.day_9_part_one()
        assert solution < 15617128862801
        # edge case is free space sections of zero length.
        # only one of these in eg. file, whose position is
        # second-to-last in the input, so it becomes irrelevant,
        # whereas in real text there are many zeroes.  Solved by:
        # adding check and only adding to the free space dict
        # if it's of non-zero length.
        lib.verify_solution(solution, 6291146824486)

    def test_part_two(self):
        assert a.day_9_part_two(self.eg_disk_map) == 2858
        lib.verify_solution(a.day_9_part_two(),
                            6307279963620, part_two=True)


class TestDay8:
    eg = """............
........0...
.....0......
.......0....
....0.......
......A.....
............
............
........A...
.........A..
............
............"""
    p2eg = """T....#....
...T......
.T....#...
.........#
..#.......
..........
...#......
..........
....#.....
.........."""

    def test_part_one(self):
        a.day_8_part_one("..A.....A.\n")
        assert a.day_8_part_one(self.eg) == 14
        lib.verify_solution(a.day_8_part_one(), 357)

    def test_part_two(self):
        assert a.day_8_part_two(self.p2eg) == 9
        assert a.day_8_part_two(self.eg) == 34
        solution = a.day_8_part_two()
        lib.verify_solution(solution, part_two=True)


class TestDay7:
    eg = """190: 10 19
3267: 81 40 27
83: 17 5
156: 15 6
7290: 6 8 6 15
161011: 16 10 13
192: 17 8 14
21037: 9 7 18 13
292: 11 6 16 20"""

    def test_part_one(self):
        assert a.day_7_part_one(self.eg) == 3749
        lib.verify_solution(a.day_7_part_one(), 8401132154762)

    def test_part_two(self):
        assert a.day_7_part_two(self.eg) == 11387
        solution = a.day_7_part_two()
        assert solution > 31813124622469
        lib.verify_solution(solution, 95297119227552, part_two=True)
        # Works (having understood the question properly,
        # see https://www.reddit.com/r/adventofcode/comments/1h8o1gh/2024_day_7_part_2_typo_in_problem_statement/
        # instead of concatenating before calculating.  But takes 27 sec


class TestDay6:
    eg_map = """....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#..."""

    def test_part_one(self):
        assert a.day_6_part_one(self.eg_map) == 41
        lib.verify_solution(a.day_6_part_one(), 5080)

    def test_p2_dev(self):
        room = a.day_6_load_map(self.eg_map)
        assert a.day_6_distance_to_next_blocker(lib.Point(2, 2), "D", room) == 1
        assert a.day_6_distance_to_next_blocker(lib.Point(0, 1), "R", room) == 3
        assert a.day_6_distance_to_next_blocker(lib.Point(2, 0), "R", room) == 1_000_000
        assert a.day_6_distance_to_next_blocker(lib.Point(3, 9), "L", room) == 7
        assert a.day_6_distance_to_next_blocker(lib.Point(9, 0), "U", room) == 1
        assert a.day_6_distance_to_next_blocker(
            lib.Point(1, 9), "L", room) == 1_000_000
        bu = {
            "U": lib.Point(11, 10),
            "R": lib.Point(10, 9),
            "D": lib.Point(9, 10),
            "L": lib.Point(10, 11),
        }
        for facing, expected in bu.items():
            assert a.day_6_back_up(lib.Point(10, 10), facing) == expected
        tr = {
            "U": "R",
            "R": "D",
            "D": "L",
            "L": "U",
        }
        for facing, expected in tr.items():
            assert a.day_6_turn_to_right(facing) == expected

    def test_part_two(self):
        assert a.day_6_part_two_simple(self.eg_map) == 6
        lib.verify_solution(a.day_6_part_two_simple(), 1919, part_two=True)

        assert a.day_6_part_two(self.eg_map) == 6
        p2_solution = a.day_6_part_two()
        assert p2_solution > 1884
        lib.verify_solution(p2_solution, 1919, part_two=True)
        # still takes over 3 minutes

    def test_timings(self):
        timeit.timeit(
            "a.day_6_part_one()", "import aoc_2024 as a",
            number=10
        )


class TestDay5:
    eg_input = """47|53
97|13
97|61
97|47
75|29
61|13
75|53
29|13
97|29
53|29
61|53
97|53
61|29
47|13
75|47
97|75
47|61
75|61
47|29
75|13
53|13

75,47,61,53,29
97,61,53,29,13
75,29,13
75,97,47,61,53
61,13,29
97,13,75,29,47"""

    def test_validation(self):
        r, u = a.day_5_load(self.eg_input)
        assert a.day_5_is_valid(u[0], r)
        assert a.day_5_is_valid(u[1], r)
        assert a.day_5_is_valid(u[2], r)
        assert not a.day_5_is_valid(u[3], r)
        assert not a.day_5_is_valid(u[4], r)
        assert not a.day_5_is_valid(u[5], r)

    def test_part_one(self):
        assert a.day_5_part_one(self.eg_input) == 143
        lib.verify_solution(a.day_5_part_one(), 4185)

    def test_part_two(self):
        r, u = a.day_5_load()
        lefts, rights = (set(v[i] for v in r) for i in (0, 1))
        print(len(lefts))
        print(len(rights))
        print(rights - lefts)
        # Interesting: in example, left hand and right hand
        #   numbers each lack one number the other side has;
        #   in the real thing, there is 100% overlap
        assert a.day_5_part_two(self.eg_input) == 123
        lib.verify_solution(a.day_5_part_two(), 4480, part_two=True)


class TestDay4:
    eg = """MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX"""

    def test_part_one(self):
        assert a.day_4_part_one(self.eg) == 18
        lib.verify_solution(a.day_4_part_one(), 2569)

    def test_part_two(self):
        assert a.day_4_part_two(self.eg) == 9
        lib.verify_solution(a.day_4_part_two(), 1998, part_two=True)


class TestDay3:
    eg = "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))"
    p2_eg = "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))"

    def test_part_one(self):
        assert a.day_3_sum_of_muls(self.eg) == 161
        assert a.day_3_part_one(self.eg) == 161
        lib.verify_solution(a.day_3_part_one(), 171183089)

    def test_part_two(self):
        assert a.day_3_part_two(self.p2_eg) == 48
        too_low, too_high = 2486652, 96020947
        assert too_low < a.day_3_part_two() < too_high
        lib.verify_solution(a.day_3_part_two(), 63866497, part_two=True)


class TestDay2:
    eg_reports = """7 6 4 2 1
1 2 7 8 9
9 7 6 2 1
1 3 2 4 5
8 6 4 4 1
1 3 6 7 9"""

    def test_p1_method(self):
        print(a.day_2_load_reports(self.eg_reports))

    def test_part_one(self):
        assert a.day_2_part_one(self.eg_reports) == 2
        lib.verify_solution(a.day_2_part_one(), correct=379)

    def test_part_two(self):
        assert a.day_2_part_two(self.eg_reports) == 4
        lib.verify_solution(a.day_2_part_two(), correct=430, part_two=True)


class TestDay1:
    eg = """3   4
4   3
2   5
1   3
3   9
3   3
"""

    def test_part_one(self):
        assert a.day_1_part_one(self.eg) == 11
        lib.verify_solution(a.day_1_part_one(), 1660292)

    def test_part_two(self):
        assert a.day_1_part_two(self.eg) == 31
        lib.verify_solution(a.day_1_part_two(), 22776016, part_two=True)
