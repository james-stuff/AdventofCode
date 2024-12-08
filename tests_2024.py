import aoc_2024 as a
import library as lib


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

    def test_part_two(self):
        # assert a.day_6_part_two(self.eg_map) == 6
        p2_solution = a.day_6_part_two()
        assert p2_solution > 1884
        lib.verify_solution(a.day_6_part_two(), part_two=True)


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
