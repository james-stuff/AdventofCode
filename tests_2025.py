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
    def test_part_two_dev(self):
        eg = """7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3"""
        reds = a.day_9_load_red_tiles()
        print(f"{max(reds)=}")
        assert a.day_9_part_two(eg) == 24
        assert a.day_9_part_two() < 4634026886

    def test(self):
        lib.verify_2025(a, 4763040296)


class TestDay8:
    eg = """162,817,812
    57,618,57
    906,360,560
    592,479,940
    352,342,300
    466,668,158
    542,29,236
    431,825,988
    739,650,466
    52,470,668
    216,146,977
    819,987,18
    117,168,530
    805,96,715
    346,949,466
    970,615,88
    941,993,340
    862,61,35
    984,92,344
    425,690,689"""

    def test_p1(self):
        assert a.day_8_part_one(self.eg) == 40
        solution = a.day_8_part_one()
        assert solution > 15939

    def test_p2(self):
        assert a.day_8_part_two(self.eg) == 25272
        p2s = a.day_8_part_two()
        assert p2s < 1387328957

    def test(self):
        lib.verify_2025(a, 123420, 673096646)


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

    def test(self):
        lib.verify_2025(a, 1363, 8184)

    def test_part_one(self):
        assert a.day_4_part_one(self.eg) == 13

    def test_part_two(self):
        assert a.day_4_part_two(self.eg) == 43


class TestDay3:
    eg = """987654321111111
811111111111119
234234234234278
818181911112111"""

    def test(self):
        lib.verify_2025(a, 17095, 168794698570517)

    def test_part_one(self):
        assert a.day_3_part_one(self.eg) == 357
        solution = a.day_3_part_one()
        assert solution > 16322

    def test_part_two(self):
        assert a.day_3_part_two(self.eg) == 3121910778619
        proposed = a.day_3_part_two()
        assert proposed > 168734619529744


class TestDay2:
    def test(self):
        lib.verify_2025(a, 12599655151, 20942028255)

    def test_part_two(self):
        solution = a.day_2_part_two()
        assert solution < 20942028300


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

    def test(self):
        lib.verify_2025(a, 1081, 6689)

    def test_part_one(self):
        assert a.day_1_part_one(self.eg) == 3

    def test_part_two(self):
        assert a.day_1_part_two(self.eg) == 6
        p2_solution = a.day_1_part_two()
        assert 6179 < p2_solution < 6707
