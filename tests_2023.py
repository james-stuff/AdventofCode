import collections
import math
import pprint
import re
import aoc_2023 as a23
import library as lib
from itertools import cycle


class TestDay23:
    eg_map = """#.#####################
#.......#########...###
#######.#########.#.###
###.....#.>.>.###.#.###
###v#####.#v#.###.#.###
###.>...#.#.#.....#...#
###v###.#.#.#########.#
###...#.#.#.......#...#
#####.#.#.#######.#.###
#.....#.#.#.......#...#
#.#####.#.#.#########v#
#.#...#...#...###...>.#
#.#.#v#######v###.###v#
#...#.>.#...>.>.#.###.#
#####v#.#.###v#.#.###.#
#.....#...#...#.#.#...#
#.#########.###.#.#.###
#...###...#...#...#.###
###.###.#.###v#####v###
#...#...#.#.>.>.#.>.###
#.###.###.#.###.#.#v###
#.....###...###...#...#
#####################.#"""

    def clear_globals(self):
        a23.day_23_walkable, a23.day_23_slopes = ({} for _ in range(2))

    def test_part_one(self):
        assert a23.day_23_part_one(self.eg_map) == 94
        self.clear_globals()
        lib.verify_solution(a23.day_23_part_one(), 2278)

    def test_no_choice_routes(self):
        a23.day_23_load_scene()
        # NB. 34 points where path can split, only 2 where there are
        #   no options (i.e. the end points)
        turning_points = [
            pt for pt in a23.day_23_walkable
            if len(a23.day_23_walkable_neighbours(pt)) > 2
        ]
        print(f"In real map, there are {len(turning_points)} points where path can split")
        ncps = a23.day_23_no_choice_paths()
        print(f"{len(ncps)=}")
        # for ncp in ncps:
        #     print(f"\t{ncp[0]} to {ncp[-1]} of length {len(ncp)}")
        print(f"All walkable points: {len(a23.day_23_walkable)}")
        all_ncp_points = set()
        for ncp in ncps:
            all_ncp_points.update(set(ncp))
        print(f"All points on ncps: {len(all_ncp_points)}")
        self.clear_globals()
        a23.day_23_load_scene(self.eg_map)
        assert len(a23.day_23_walkable_neighbours(lib.Point(0, 1))) == 1
        assert len(a23.day_23_walkable_neighbours(lib.Point(1, 1))) == 2
        assert len(a23.day_23_walkable_neighbours(lib.Point(3, 7))) == 2
        assert len(a23.day_23_walkable_neighbours(lib.Point(3, 11))) == 3
        eg_ncps = a23.day_23_no_choice_paths()

    def test_part_two(self):
        a23.day_23_load_scene(self.eg_map)
        ncp = a23.day_23_ncp_as_dict()
        # self.clear_globals()
        assert a23.day_23_part_two(self.eg_map) == 154
        print("")
        self.clear_globals()
        print(f"No. of walkable points: {len(a23.day_23_walkable)}")
        solution = a23.day_23_part_two()
        assert solution > 5478
        lib.verify_solution(solution, 6734, part_two=True)
        # Correct answer but took 08:25 minutes



class TestDay22:
    eg_input = """1,0,1~1,2,1
0,0,2~2,0,2
0,2,3~2,2,3
0,0,4~0,2,4
2,0,5~2,2,5
0,1,6~2,1,6
1,1,8~1,1,9"""

    """bricks = {starting co-ordinate: dimensions}?
        Then collapse the stack
        Need a dict of {brick: {bricks it supports}}
        Starting co-ordinate will always be unique id"""

    def test_load_and_explore(self):
        real_text = a23.Puzzle23(22).get_text_input()

        def load(t: str) -> {}:
            t = t.strip("\n")
            def co_ord_str_to_tuple(csv: str) -> (int,):
                return tuple(int(v) for v in csv.split(","))

            bricks = {}
            for row in t.split("\n"):
                end_1, end_2 = row.split("~")
                bricks[co_ord_str_to_tuple(end_1)] = co_ord_str_to_tuple(end_2)
            return bricks

        br = load(real_text)
        for bk, bv in br.items():
            assert all(bk[n] <= bv[n] for n in range(3))
        assert min(k[2] for k in br.keys()) >= 1
        assert min(k[0] for k in br.keys()) >= 0
        assert min(k[1] for k in br.keys()) >= 0
        print(f"Extent of snapshot: "
              f"x={max(v[0] for v in br.values())}, "
              f"y={max(v[1] for v in br.values())}, "
              f"z={max(v[2] for v in br.values())}, ")

        dimmed_bricks = a23.day_22_load_bricks(self.eg_input)
        assert dimmed_bricks[(0, 2, 3)] == (3, 1, 1)
        assert dimmed_bricks[(1, 1, 8)] == (1, 1, 2)

    def test_collapse(self):
        new_state = a23.day_22_collapse(
            a23.day_22_load_bricks(self.eg_input)
        )
        assert isinstance(new_state, dict)
        assert new_state[(1, 0, 1)] == (1, 3, 1)    # A
        # print(new_state)
        assert new_state[(0, 0, 2)] == (3, 1, 1)    # B
        assert new_state[(0, 2, 2)] == (3, 1, 1)    # C
        assert new_state[(0, 0, 3)] == (1, 3, 1)    # D
        assert new_state[(2, 0, 3)] == (1, 3, 1)    # E
        assert new_state[(0, 1, 4)] == (3, 1, 1)    # F
        assert new_state[(1, 1, 5)] == (1, 1, 2)    # G

        real_bricks = a23.day_22_load_bricks()
        real_collapse = a23.day_22_collapse(real_bricks)
        assert len(real_collapse) == len(real_bricks)
        # print(f"{len(real_collapse)=}\n{real_collapse}")
        assert all(0 <= b[0] <= 9 for b in real_collapse)
        assert all(0 <= b[1] <= 9 for b in real_collapse)
        assert all(0 <= b[2] < 299 for b in real_collapse)
        print(f"Highest brick now suspended at z={max(b[2] for b in real_collapse)}")

    def test_process(self):
        collapsed = a23.day_22_collapse(
            a23.day_22_load_bricks(self.eg_input)
        )
        rel = a23.day_22_get_supporting_relationships(collapsed)
        # print(rel)

    def test_part_one(self):
        assert a23.day_22_part_one(self.eg_input) == 5
        lib.verify_solution(a23.day_22_part_one(), correct=421)

    def test_chain_reaction(self):
        collapsed = a23.day_22_collapse(
            a23.day_22_load_bricks(self.eg_input)
        )
        rel = a23.day_22_get_supporting_relationships(collapsed)
        assert a23.day_22_chain_reaction(rel, (0, 1, 4)) == 1
        assert a23.day_22_chain_reaction(rel, (1, 0, 1)) == 6

    def test_part_two(self):
        assert a23.day_22_part_two(self.eg_input) == 7
        lib.verify_solution(a23.day_22_part_two(), correct=39247, part_two=True)


class TestDay21:
    eg_map = """...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
..........."""
    part_two_expected = """In exactly 6 steps, he can still reach 16 garden plots.
In exactly 10 steps, he can reach any of 50 garden plots.
In exactly 50 steps, he can reach 1594 garden plots.
In exactly 100 steps, he can reach 6536 garden plots.
In exactly 500 steps, he can reach 167004 garden plots.
In exactly 1000 steps, he can reach 668697 garden plots.
In exactly 5000 steps, he can reach 16733044 garden plots."""

    def test_load(self):
        gdn = a23.day_21_load_garden(self.eg_map)
        assert len([*filter(lambda v: v, gdn.values())]) == 1
        assert gdn[(5, 5)] is True
        assert (9, 1) not in gdn
        assert all([not gdn[(10, n)] for n in range(11)])
        print(f"{len(gdn)=}, {len(self.eg_map.split()[0])=}")

    def test_with_example(self):
        # func = lambda steps: (
        #     a23.day_21_count_reachable_plots(
        #     a23.day_21_load_garden(self.eg_map), steps
        #     )
        # )
        func = lambda steps: (
            a23.day_21_run_for_n_steps(
                a23.day_21_load_garden(self.eg_map), steps
            )
        )
        assert func(0) == 1
        assert func(1) == 2
        assert func(2) == 4
        assert func(3) == 6
        assert func(6) == 16

    def test_part_one(self):
        garden = a23.day_21_load_garden(a23.Puzzle23(21).get_text_input())
        lib.verify_solution(
            # a23.day_21_count_reachable_plots(garden, 64),
            a23.day_21_run_for_n_steps(garden, 64),
            3617
        )

    def extract_p2_expected(self) -> {}:
        return {
            steps: expected
            for steps, expected in zip(
                *(
                    [
                        int(m.group().split()[0])
                        for m in
                        re.finditer(s, self.part_two_expected)
                    ]
                    for s in (r"\d+ steps", r"\d+ garden")
                )
            )
        }

    def test_with_infinite_garden(self):
        eg_garden = a23.day_21_load_garden(self.eg_map)
        results = a23.day_21_first_n_results(eg_garden)
        ep = self.extract_p2_expected()
        for st, exp in ep.items():
            if st < 100:
                continue
            predicted = a23.day_21_count_reachable_plots_in_infinite_garden(
                eg_garden, st, results
            )
            if predicted != exp:
                print(f"Test fails for {st} steps")
                assert predicted == exp

    def test_find_relationship(self):
        """Thought experiment: what if you operate on a rock-free grid?
            would the answer be (x + 1)-squared?  Conclusion: yes
            Could it be something like (x + 1)-squared
            minus half the number of rocks within the reachable radius?  No,
                not as simple as this because once blocked, a path cannot
                continue in its original direction"""
        unimpeded_garden = {
            lib.Point(x, y): False
            for x in range(21)
            for y in range(21)
        }
        unimpeded_garden[(10, 10)] = True
        # a23.day_21_rocks = {(9, 10)}
        del unimpeded_garden[(10, 11)]
        del unimpeded_garden[(9, 11)]
        del unimpeded_garden[(8, 11)]
        eg_garden = a23.day_21_load_garden(a23.Puzzle23(21).get_text_input())
        total_steps = 10
        for s in range(total_steps):
        # s = 10
            unimpeded_garden = a23.day_21_make_step_and_print_garden(
                unimpeded_garden
            )
        # print(f"{s=}, {a23.day_21_count_reachable_plots(
        #     unimpeded_garden, s
        # )}, {(s + 1) ** 2=}")
        print(f"{len([*filter(lambda v: v, eg_garden.values())])}, {(total_steps + 1) ** 2=}")
        possible_destinations = len([*filter(lambda v: v, eg_garden.values())])
        print(f"{total_steps=}, {possible_destinations=}"
              f", shortfall from (x + 1)^2 = {(total_steps + 1) ** 2 - possible_destinations}")

        # IDEA: only need to know the furthest out points,
        #       and anything in between is a grid with an 'O'
        #       every two steps, except rocks?
        # NOTE: in example map, none of the N, S, E, W axes are clear
        #       but in the real thing, they all are completely clear
        assert [*filter(lambda k: eg_garden[k], eg_garden.keys())][0] == lib.Point(65, 65)
        assert all(p in eg_garden for p in [lib.Point(65, y) for y in range(130)])
        assert all(p in eg_garden for p in [lib.Point(x, 65) for x in range(130)])

    def test_run_further(self):
        gdn = a23.day_21_load_garden(self.eg_map)
        print([*filter(lambda k: k[0] == 1, gdn.keys())])
        dup_gdn = a23.day_21_duplicate_garden(gdn, (1, 0))
        assert len(dup_gdn) == len(gdn)
        print([*filter(lambda k: k[0] == 12, dup_gdn.keys())])
        assert (1, 7) not in gdn
        assert gdn[(1, 8)] is False
        assert (12, 7) not in dup_gdn
        assert lib.Point(12, 8) in dup_gdn
        # print(f"{gdn[lib.Point(12, 8)]=}")
        assert dup_gdn[(12, 8)] is False
        r_dup = 46
        big_garden = {}
        for m in range(-r_dup, r_dup + 1):
            for n in range(-r_dup, r_dup + 1):
                big_garden.update(
                    a23.day_21_duplicate_garden(gdn, (m, n))
                )
        # assert len(big_garden) == 169 * len(gdn)
        # assert min(big_garden) == (-66, -66)
        # assert max(big_garden) == (76, 76)
        assert len([*filter(lambda v: v, big_garden.values())]) == 1
        exp = self.extract_p2_expected()
        for k, v in exp.items():
            # TAKES OVER 6 MINUTES if going to 500 steps
            #       but at least tests pass
            if k < 1:
                print(f"Testing {k} steps:")
                assert a23.day_21_count_reachable_plots(
                    big_garden, k) == v

        previous, reachable = 0, 0
        for n_steps in range(500):
            big_garden = a23.day_21_make_step(big_garden)
            previous, reachable = (
                reachable,
                len([*filter(lambda v: v, big_garden.values())])
            )
            if n_steps % 11 == 1:
                square_size = (n_steps + 2) ** 2
                print(f"{n_steps + 1:>3} steps -> "
                      f"{reachable:>6}, increment={reachable - previous:>6} "
                      f"(n + 1)-squared={square_size} "
                      f"missing = {square_size - reachable} "
                      f"rocks encountered={a23.day_21_rocks_within(big_garden, n_steps + 1)}"
                      f" furthest = {max(filter(lambda k: big_garden[k], 
                                                big_garden.keys()))}")
                if n_steps < 50:
                    a23.day_21_print_garden(big_garden)

                    # todo: How about: run for the first 100 steps.
                    #                    find the pattern per step
                    #                   add on the increasing-by-80 missing plots each 11 steps

    def test_possible_p2_solution(self):
        base_garden = a23.day_21_load_garden(self.eg_map)
        results = a23.day_21_first_n_results(base_garden)
        answer = a23.day_21_count_reachable_plots_in_infinite_garden(
            base_garden, 100, results
        )
        print(f"{answer=}")

        base_string = "\n".join("." * 11 for _ in range(11))
        base_string = f"{base_string[:65]}S{base_string[66:]}"

        def insert_rock_at(s: str, row: int, col: int) -> str:
            n = (row * 12) + col
            return f"{s[:n]}#{s[n + 1:]}"

        # one_1 = insert_rock_at(base_string, 4, 5)
        # one_2 = insert_rock_at(base_string, 1, 1)
        # one_3 = insert_rock_at(base_string, 8, 3)
        def multi_rock_garden(s: str, rock_locs: [(int,)]) -> str:
            for rl in rock_locs:
                s = insert_rock_at(s, *rl)
            print(s)
            return s

        two_rock_gardens = [
            multi_rock_garden(base_string, [(9, 8), (9, 9)]),
            multi_rock_garden(base_string, [(7, 5), (4, 3)]),

            multi_rock_garden(base_string, [(4, 4), (4, 5), (4, 6), (8, 8), (7, 7)]),
        ]
        for bs in two_rock_gardens:
            garden = a23.day_21_load_garden(bs)
            results = a23.day_21_first_n_results(garden)
            answer = a23.day_21_count_reachable_plots_in_infinite_garden(
                garden, 100, results
            )
            print(f"{answer=}")

        # print("\n", insert_rock_at(base_string, 6, 7))

    def test_rocks_within(self):
        gdn = a23.day_21_load_garden(self.eg_map)
        results = [0, 2, 5, 9, 18]
        for n, r in zip(range(5), results):
            assert a23.day_21_rocks_within(gdn, n) == r#((n + 1) ** 2) + (n ** 2)

    def test_part_two(self):
        puzzle_text = a23.Puzzle23(21).get_text_input()
        gdn = a23.day_21_load_garden(puzzle_text)
        g_size = max(gdn)[0] + 1
        print(f"Garden size: {g_size}, containing {puzzle_text.count('#')} rocks")
        d = a23.day_21_first_n_results(gdn, g_size * 8)
        print(f"Result: {a23.day_21_count_reachable_plots_in_infinite_garden(gdn, 26501365, d)}")


class TestDay20:
    basic_example = """broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a"""
    example = """broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output"""

    def test_setup(self):
        """False = low pulse, True = high pulse
            queue = (destination, pulse type)"""
        assert a23.day_20_part_one(self.basic_example) == 32_000_000
        assert a23.day_20_part_one(self.example) == 11_687_500

    def test_part_one(self):
        lib.verify_solution(a23.day_20_part_one(), 680_278_040)

    def test_p2_exploration(self):
        ff, cj, conn = a23.day_20_set_up()
        print(f"There are {len(ff)} flip-flops and {len(cj)} conjunctions")
        touched_ff = set()
        pressed_times = 0
        monitored_ff, state = "%bs", False
        while pressed_times < 20:
            print(f"{pressed_times=}")
            a23.day_20_push_the_button(ff, cj, conn, pressed_times)
            pressed_times += 1
            ff_on = {f for f in ff if ff[f]}
            first_touched = ff_on - touched_ff
            touched_ff.update(first_touched)
            if ff[monitored_ff] != state:
                print(f"{pressed_times:>6} {monitored_ff}"
                      f" is turned {'on' if ff[monitored_ff] else 'off'}")
                state = ff[monitored_ff]
        print(f"Remaining untouched flip-flops: {set(ff) - touched_ff}")
        # LAST FFs to be ACTIVATED: {'gs', 'dn', 'pp', 'rc'}
        """
        There are 48 flip-flops and 9 conjunctions
     1 {'fs', 'sh', 'ps', 'nm'}
     2 {'db', 'rs', 'rr', 'bz'}
     4 {'cf', 'cq', 'nv', 'ls'}
     8 {'jr', 'vg', 'bv', 'sl'}
    16 {'xz', 'gn', 'cg', 'kz'}
    32 {'rq', 'tf', 'mj', 'dg'}
    64 {'ff', 'hc', 'sr', 'xl'}
   128 {'bk', 'vb', 'hp', 'vr'}
   256 {'qd', 'tv', 'bs', 'ft'}
   512 {'lh', 'lf', 'rj', 'sz'}
  1024 {'nd', 'vx', 'pn', 'fz'}
  2048 {'gs', 'dn', 'pp', 'rc'}
  """

            # print("".join(f"{'# ' if ff[f] else '  '}" for f in ff))
            # print(''.join([f'{sum(cj[c].values())}' for c in cj]))
            # print(f"{'' if pressed_times % 10 else pressed_times:>6} "
            #       f"{''.join(['#' if len(cj[c]) == sum(cj[c].values()) else ' ' for c in cj])}")
            # turned_on = sum(ff.values())
            # print(f"{'' if pressed_times % 10 else pressed_times:>6} "
            #       f"{turned_on}")
            # if turned_on > 36:
            #     print(f"{pressed_times:>6} {turned_on}")
        # t, c = a23.day_20_load(a23.Puzzle23(20).get_text_input())
        # involved = ["broadcaster"]
        # s = 0
        # while "rx" not in involved:
        #     s += 1
        #     print(f"Step {s:>2}: ", end="")
        #     next_nodes = []
        #     for node in involved:
        #         next_nodes += c[node]
        #     # print(" ".join(f"{t[nd]}{nd}" for nd in next_nodes if nd != "rx"))
        #     print(f"{len(next_nodes)=} vs. {len(t)=}, {len(c)=}")
        #     involved = next_nodes

    def test_reverse_journey(self):
        ff, cj, conn = a23.day_20_set_up()
        t = a23.day_20_load_connections(a23.Puzzle23(20).get_text_input())

        def triggered_by(receiver: str) -> [str]:
            return [k for k, v in conn.items()
                    if receiver in v]
        end_points = ["&vf"]

        for steps in range(2):
            prev_ep = [*end_points]
            if len(prev_ep) == 1:
                end_points = triggered_by(end_points[0])
                print(f"{prev_ep[0]} is triggered by {', '.join(end_points)}")
            else:
                # end_points = [new_ep for e in end_points for new_ep in triggered_by(e)]
                # print(f" . . . which are in turn triggeed by {len(end_points)} receivers: {', '.join(full_name(rcv) for rcv in end_points)}")
                print(f"in turn:")
                for ep in end_points:
                    print(f"\t{ep} is triggered by {', '.join(triggered_by(ep))}")

    def test_with_logging(self):
        ff, conj, con = a23.day_20_set_up()
        data = [[*ff.keys(), *conj.keys()]]
        for push in range(10_000):
            a23.day_20_push_the_button(ff, conj, con, push)
            data_row = ([int(ffv) for ffv in ff.values()] +
                        [sum(cjv.values()) for cjv in conj.values()])
            data.append(data_row)
        # with open("output.csv", "w") as file:
        #     file.write(
        #         "\n".join(
        #             ",".join(f"{item}" for item in row)
        #             for row in data
        #         )
        #     )

    def test_part_two(self):
        """Only the conjunction vf transmits to rx.
            vf is itself the product of four other conjunctions
            I think all of their respective sources are also conjunctions
        Max. queue size seems to follow a pattern.  Usually repeats every 4 cycles,
        but every 32 cycles, something interesting happens (alternates 34, 38),
        so maybe something is repeating every 64 cycles?
        This regularity seems to be broken after tens of thousands of cycles
        Make rx into a flip-flop to detect when it is triggered?
        For rx to receive a low pulse, vf must receive four high pulses:
            pm, mk, pk, hf
        But (up to 1,000 cycles) all its inputs are high (1/1) so it only receives four lows"""
        """&vf is receiving a continuous run of lows seemingly from the start.
            &pm, &mk, &pk and &hf must all be receiving highs, which means
            their contributors must all be sending lows.  In theory, &vf->rx is
            triggered when one of &gp, &bn, &rt or &cz sends a high, which
            happens when any one of their inputs is a low"""
        queue = collections.deque([("&bn", "&mk", False)])
        ff, cj, conn = a23.day_20_set_up()
        a23.day_20_process_button_press(ff, cj, conn, queue)
        print(f"After simulation, {cj['&vf']=}")
        # a23.day_20_part_two()


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
        assert len(eg_instr.split("\n")) == 12
        assert eg_dicts[0]["x"] == 787
        assert eg_dicts[1]["a"] == 2067
        assert eg_dicts[3]["s"] == 291
        main_instr, main_dicts = a23.day_19_load_inputs()
        assert len(main_instr.split("\n")) == 523
        # assert all(main_instr.split("\n"))
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
        lib.verify_solution(p1_solution, 280909)

    def test_explore_real_input(self):
        # inst, parts = a23.day_19_load_inputs()
        text = a23.Puzzle23(19).get_text_input()
        row_indices = {m.start() for m in re.finditer("}\n", text)}

        def find_row(string_index: int) -> int:
            earlier_indices = {*filter(lambda ri: ri < string_index, row_indices)}
            if earlier_indices:
                return len(earlier_indices) + 1
            return 1

        assert find_row(0) == 1
        assert find_row(12) == 1
        assert find_row(16) == 2
        assert find_row(re.search(r"tfs{", text).start()) == 203

        for search_term in [r"\D0\D", "-"]:
            for match in re.finditer(search_term, text):
                print(f"{search_term} found on line {find_row(match.start())}")
        """Findings:
        There can be more than one test per workflow on the same letter
        Line 53: xrk{a>122:jf,a<52:A,A} if a <= 122, second 'a' test is irrelevant
        If a part passes first 'in' test, it definitely passed first 'zd' test"""

    def test_concept_for_part_two(self):
        """State: (min_x, max_x, min_m, max_m, . . . max_s, workflow_about_to_be_entered)
            9-tuple
            Initial: (1, 4_000, 1, 4_000, 1, 4_000, 1, 4_000, "in")
            Form queue of states to explore.  For each one, create new states
            for each step in the process that doesn't lead to rejection.
            Put accepted states into a final set to evaluate"""

        def part_two_process(state: tuple, workflows: str) -> [tuple]:
            out_states = []
            raw_instruction = re.search(f"\n{state[-1]}" + "{.+}", workflows).group()
            s, e = (raw_instruction.index(ch) for ch in "{}")
            raw_tests = raw_instruction[s + 1:e]
            for test in raw_tests.split(","):
                if re.search(r"[xmas][<>]", test):
                    state, new = a23.day_19_p2_state_splitter(state, test)
                    outcome = test[test.index(":") + 1:]
                    if outcome.islower():
                        new = tuple((*new[:-1], outcome))
                        out_states.append(new)
                    elif outcome == "A":
                        accepted.add(new)
                    elif outcome == "R":
                        rejected.add(new)
                elif test.islower() and 1 < len(test) < 4:
                    out_states.append(tuple((*state[:-1], test)))
                elif test == "A":
                    accepted.add(state)
                elif test == "R":
                    rejected.add(state)
            return out_states

        workflows, _ = a23.day_19_load_inputs(self.example)
        initial = tuple((*(1, 4_001) * 4, "in"))
        assert a23.day_19_p2_state_splitter(initial, "x<1000:abc") == (
            (1_000, 4_001, 1, 4_001, 1, 4_001, 1, 4_001, "in"),
            (1, 1_000, 1, 4_001, 1, 4_001, 1, 4_001, "in"),
        )
        assert a23.day_19_p2_state_splitter(initial, "m>1000:abc") == (
            (1, 4_001, 1, 1_001, 1, 4_001, 1, 4_001, "in"),
            (1, 4_001, 1_001, 4_001, 1, 4_001, 1, 4_001, "in"),
        )
        real_state = (1, 4_001, 1, 1_000, 1, 4_001, 1_351, 4_001, "qqz")
        assert a23.day_19_p2_state_splitter(real_state, "s>2770:qs") == (
            (1, 4_001, 1, 1_000, 1, 4_001, 1_351, 2_771, "qqz"),
            (1, 4_001, 1, 1_000, 1, 4_001, 2_771, 4_001, "qqz")
        )
        queue = [initial]
        accepted = set()
        rejected = set()
        while queue:
            next_state = queue.pop()
            print(f"Queue size: {len(queue)}, accepted: {len(accepted)}")
            queue += part_two_process(next_state, workflows)
        pprint.pprint(accepted)
        answer = sum(
            math.prod(acc[n + 1] - acc[n] for n in range(0, 8, 2))
            for acc in accepted
        )
        size_of_rejected = sum(
            math.prod(rej[n + 1] - rej[n] for n in range(0, 8, 2))
            for rej in rejected
        )
        print(answer, f"differs by {167409079868000 - answer}")
        print(f"Total states considered: {answer + size_of_rejected}")
        # assert answer + size_of_rejected == 4_000 ** 4

    def test_part_two(self):
        assert a23.day_19_part_two(self.example) == 167409079868000
        lib.verify_solution(a23.day_19_part_two(), 116138474394508, part_two=True)


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
