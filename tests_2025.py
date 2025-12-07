import aoc_2025 as a
import library as lib


class TestDay12:
    def test(self):
        lib.verify_2025(a)


class TestDay11:
    def test(self):
        lib.verify_2025(a)


class TestDay10:
    def test(self):
        lib.verify_2025(a)


class TestDay9:
    def test(self):
        lib.verify_2025(a)


class TestDay8:
    def test(self):
        lib.verify_2025(a)


class TestDay7:
    def test_p1(self):
        solution = a.day_7_part_one()
        assert solution > 917

    def test(self):
        lib.verify_2025(a, 1524, 32982105837605)


class TestDay6:
    def test_p2(self):
        eg = """123 328  51 64 
 45 64  387 23 
  6 98  215 314
*   +   *   +  """
        assert a.day_6_part_two(eg) == 3263827
        assert a.day_6_part_two() > 10121317

    def test(self):
        lib.verify_2025(a, 5060053676136, 9695042567249)


class TestDay5:
    def test(self):
        lib.verify_2025(a, 681, 348820208020395)
    #
    # def test_part_one(self):
    #     lib.verify_solution(a.day_5_part_one(), 681)
    #
    # def test_part_two(self):
    #     lib.verify_solution(a.day_5_part_two(), 348820208020395, part_two=True)


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
