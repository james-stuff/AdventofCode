import aoc_2025 as a
import library as lib


class TestDay4:
    eg = """..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@."""

    def test_part_one(self):
        assert a.day_4_part_one(self.eg) == 13
        lib.verify_solution(a.day_4_part_one())

    def test_part_two(self):
        assert a.day_4_part_two(self.eg) == 43
        lib.verify_solution(a.day_4_part_two(), part_two=True)


class TestDay3:
    eg = """987654321111111
811111111111119
234234234234278
818181911112111"""

    def test_part_one(self):
        assert a.day_3_part_one(self.eg) == 357
        solution = a.day_3_part_one()
        assert solution > 16322
        lib.verify_solution(solution, 17095)

    def test_part_two(self):
        assert a.day_3_part_two(self.eg) == 3121910778619
        proposed = a.day_3_part_two()
        assert proposed > 168734619529744
        lib.verify_solution(proposed, 168794698570517, part_two=True)


class TestDay2:
    def test_part_one(self):
        lib.verify_solution(a.day_2_part_one(), 12599655151)


    def test_part_two(self):
        solution = a.day_2_part_two()
        assert solution < 20942028300
        lib.verify_solution(solution, 20942028255, part_two=True)


class TestDay1:
    eg = """L68
L30
R48
L5
R60
L55
L1
L99
R14
L82"""

    def test_part_one(self):
        assert a.day_1_part_one(self.eg) == 3
        lib.verify_solution(a.day_1_part_one(), 1081)

    def test_part_two(self):
        assert a.day_1_part_two(self.eg) == 6
        p2_solution = a.day_1_part_two()
        assert 6179 < p2_solution < 6707
        lib.verify_solution(a.day_1_part_two(), part_two=True)
