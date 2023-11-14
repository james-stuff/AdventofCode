import collections
import aoc_2022
import library as lib
import pytest
import pprint


class TestDay19:
    example_text = """Blueprint 1:
  Each ore robot costs 4 ore.
  Each clay robot costs 2 ore.
  Each obsidian robot costs 3 ore and 14 clay.
  Each geode robot costs 2 ore and 7 obsidian.

Blueprint 2:
  Each ore robot costs 2 ore.
  Each clay robot costs 3 ore.
  Each obsidian robot costs 3 ore and 8 clay.
  Each geode robot costs 3 ore and 12 obsidian."""

    true_geodes_per_blueprint = {1: 3, 2: 0, 3: 0, 4: 2, 5: 9, 6: 2,
                                 7: 1, 8: 0, 9: 6, 10: 1, 11: 15, 12: 0,
                                 13: 9, 14: 11, 15: 0, 16: 0, 17: 12,
                                 18: 0, 19: 0, 20: 3, 21: 3, 22: 1,
                                 23: 16, 24: 0, 25: 7, 26: 9, 27: 5,
                                 28: 0, 29: 15, 30: 1}

    def test_load(self):
        costs = aoc_2022.day_19_read_blueprint(self.example_text[:len(self.example_text) // 2])
        assert isinstance(costs, dict)
        assert len(costs) == 4
        assert costs[aoc_2022.ORE] == {aoc_2022.ORE: 4}
        assert costs[aoc_2022.OBSIDIAN] == {aoc_2022.ORE: 3, aoc_2022.CLAY: 14}
        assert all(isinstance(item, dict) for item in costs.values())
        assert costs == aoc_2022.day_19_costs
        ore_components = [v[aoc_2022.ORE] for v in costs.values()]
        print(ore_components)
        assert all([v[aoc_2022.ORE] <= 4 for v in costs.values()])
        actual_input = aoc_2022.Puzzle22(19).get_text_input()
        it = aoc_2022.day_19_blueprint_start_indices(actual_input)
        assert len(it) == 30
        for i, blueprint in enumerate(it):
            end_pos = it[i + 1] if i < len(it) - 1 else blueprint + 500
            bp_costs = aoc_2022.day_19_read_blueprint(actual_input[blueprint:end_pos])
            assert isinstance(bp_costs, dict)
            assert len(bp_costs) == 4
            assert all(len(v) == 2 for v in [*bp_costs.values()][2:])
            assert all(isinstance(item, dict) for item in bp_costs.values())
            assert all([v[aoc_2022.ORE] <= 4 for v in bp_costs.values()])

    def test_data_structures(self):
        production_path = {0: aoc_2022.ORE, 3: aoc_2022.CLAY,
                           5: aoc_2022.CLAY, 7: aoc_2022.CLAY,
                           11: aoc_2022.OBSIDIAN, 12: aoc_2022.CLAY,
                           15: aoc_2022.OBSIDIAN, 18: aoc_2022.GEODE,
                           21: aoc_2022.GEODE}
        assert aoc_2022.day_19_geode_count(production_path) == 9

    def test_resource_availability(self):
        def snip_at_minute(in_dict: dict, last_minute: int) -> dict:
            return {k: v for k, v in in_dict.items() if k <= last_minute}

        example_path = {0: 0, 3: 1, 5: 1, 7: 1, 11: 2, 12: 1, 15: 2, 18: 3, 21: 3}
        assert aoc_2022.day_19_current_resources(snip_at_minute(example_path, 0), 0) == {0: 0}
        assert aoc_2022.day_19_current_resources(snip_at_minute(example_path, 2), 2) == {0: 2}
        assert aoc_2022.day_19_current_resources(snip_at_minute(example_path, 3), 3) == {0: 1, 1: 0}
        assert aoc_2022.day_19_current_resources(snip_at_minute(example_path, 6), 6) == {0: 2, 1: 4}
        assert aoc_2022.day_19_current_resources(snip_at_minute(example_path, 24), 24) == \
               {0: 6, 1: 41, 2: 8, 3: 9}

    def test_purchase_options(self):
        assert not aoc_2022.day_19_purchasing_options({})
        assert not aoc_2022.day_19_purchasing_options({aoc_2022.ORE: 1})
        assert aoc_2022.day_19_purchasing_options({aoc_2022.ORE: 2}) == [aoc_2022.CLAY]
        assert aoc_2022.day_19_purchasing_options({aoc_2022.ORE: 6,
                                                   aoc_2022.CLAY: 13}) == [aoc_2022.ORE,
                                                                           aoc_2022.CLAY]
        assert aoc_2022.day_19_purchasing_options({aoc_2022.ORE: 8,
                                                   aoc_2022.CLAY: 16}) == [aoc_2022.ORE,
                                                                           aoc_2022.CLAY,
                                                                           aoc_2022.OBSIDIAN]
        assert aoc_2022.day_19_purchasing_options({aoc_2022.ORE: 100,
                                                   aoc_2022.CLAY: 100,
                                                   aoc_2022.OBSIDIAN: 100,
                                                   aoc_2022.GEODE: 100}) == [aoc_2022.ORE,
                                                                             aoc_2022.CLAY,
                                                                             aoc_2022.OBSIDIAN,
                                                                             aoc_2022.GEODE]
        assert aoc_2022.day_19_purchasing_options({aoc_2022.ORE: 3,
                                                   aoc_2022.OBSIDIAN: 19}) == [aoc_2022.CLAY,
                                                                               aoc_2022.GEODE]

    def test_p1_debug(self):
        """run for one blueprint at a time"""
        run_for_blueprint = 19
        print("\n", f"BLUEPRINT {run_for_blueprint}:")
        text = aoc_2022.Puzzle22(19).get_text_input()
        start, end = 0, 100_000_000
        # it = aoc_2022.day_19_blueprint_start_indices(text)
        # start = it[run_for_blueprint - 1]
        # if run_for_blueprint < 30:
        #     end = it[run_for_blueprint]
        text = self.example_text
        end = self.example_text.index("Blueprint 2:")
        aoc_2022.day_19_costs = aoc_2022.day_19_read_blueprint(text[start:end])
        paths_up_to_11 = aoc_2022.day_19_next_possible_paths({0: 0}, 10)
        assert {0: 0, 3: 1, 5: 1, 7: 1, 11: 2} in paths_up_to_11

    def test_20_sept_brain_fart(self):
        path = {0: 0}
        assert aoc_2022.day_19_minutes_required_to_produce(aoc_2022.ORE, path) == 5
        assert aoc_2022.day_19_minutes_required_to_produce(aoc_2022.CLAY, path) == 3
        multi_robot_path = {0: 0, 3: 1}
        assert aoc_2022.day_19_minutes_required_to_produce(
            aoc_2022.ORE, multi_robot_path) == 4
        assert aoc_2022.day_19_minutes_required_to_produce(
            aoc_2022.CLAY, multi_robot_path) == 2
        assert aoc_2022.day_19_minutes_required_to_produce(
            aoc_2022.OBSIDIAN, multi_robot_path) == 15
        args = path, 22
        # paths_using_old_method = aoc_2022.original_day_19_next_possible_paths(*args)
        # aoc_2022.day_19_evaluate_complete_paths(paths_using_old_method)
        """already have enough resources"""
        enough_ore_for_clay = {0: 0, 5: 0}
        assert aoc_2022.day_19_minutes_required_to_produce(aoc_2022.CLAY, enough_ore_for_clay) == 2
        enough_for_immediate_clay = {0: 0, 5: 0, 7: 0, 8: 0, 11: 1}
        assert aoc_2022.day_19_minutes_required_to_produce(aoc_2022.CLAY,
                                                           enough_for_immediate_clay) == 1
        print("\t\tNew method:")
        # using_new_method = aoc_2022.day_19_next_possible_paths(*args)
        # aoc_2022.day_19_evaluate_complete_paths(using_new_method)
        # assert len(paths_using_old_method) == len(using_new_method)
        # breaks down for complete paths as new method relaxes the 'worthwhile' condition

    def test_parameters_for_part_one(self):
        origin_path = {0: 0}
        minutes_to_run_for = 14
        obsidian_blueprints = 0

        text = aoc_2022.Puzzle22(19).get_text_input()
        it = aoc_2022.day_19_blueprint_start_indices(text)
        for i, blueprint in enumerate(it):
            print(f"+++ BLUEPRINT {i + 1} +++")
            end_pos = it[i + 1] if i < len(it) - 1 else blueprint + 500
            aoc_2022.day_19_costs = aoc_2022.day_19_read_blueprint(text[blueprint:end_pos])
            initial_paths = aoc_2022.day_19_next_possible_paths(origin_path, minutes_to_run_for)
            print(f"\tProduces {len(initial_paths):>8,} paths initially")#, of which "
                  # f"{len([p for p in initial_paths if max(p.values()) == aoc_2022.GEODE])}"
                  # f" have produced a geode robot")
            potential_scores = collections.defaultdict(int) # {score: no.of paths}
            # scored_paths = [(max(path.keys()) + aoc_2022.GEODE - max(path.values()), path) for path in initial_paths]
            scored_paths = [(aoc_2022.day_19_pseudo_score_partial_path(path), path) for path in initial_paths]
            best_score = min(t[0] for t in scored_paths)
            paths_to_proceed_with = [fp[1] for fp in filter(lambda sp: sp[0] <= best_score + 1, scored_paths)]
            final_paths = []
            for ppw in paths_to_proceed_with:
                final_paths += aoc_2022.day_19_next_possible_paths(ppw)
            print(f"\t-> {len(final_paths)} final paths")
            aoc_2022.day_19_evaluate_complete_paths(final_paths)

            for score, _ in scored_paths:
                potential_scores[score] += 1
            print(f"\t{'   '.join([str(k) + ': ' + str(v) for k, v in sorted(potential_scores.items())])}")
            if i == 27:
                for ip in initial_paths:
                    print(ip)
            assert len(potential_scores) > 0
            # if any(max(path.values()) == aoc_2022.OBSIDIAN for path in initial_paths):
            #     obsidian_blueprints += 1
            #     print(f"\tThis blueprint can produce obsidian already")
        assert not obsidian_blueprints

    def test_part_one_example(self):
        assert aoc_2022.day_19_minutes_required_to_produce(aoc_2022.OBSIDIAN,
                                                           {0: 0, 5: 0, 8: 0, 10: 0, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}
                                                           ) == 2
        # NOW WORKS FOR BOTH EXAMPLES (but takes over four minutes)
        first_half, _, second_half = self.example_text.partition("\n\n")
        expected = {first_half: 9, second_half: 12}
        for blueprint_text, result in expected.items():
            aoc_2022.day_19_costs = aoc_2022.day_19_read_blueprint(blueprint_text)
            # assert aoc_2022.day_19_geodes_found_on_random_walks(24) == result
            assert aoc_2022.day_19_breadth_first_search() == result
            # found_paths = aoc_2022.day_19_branch_and_bound_best_paths()
            # assert aoc_2022.day_19_evaluate_complete_paths(found_paths) == result
            # if blueprint_text is first_half:
            #     ten_minute_paths = aoc_2022.day_19_next_possible_paths({0: 0}, 10)
            #     assert {0: 0, 3: 1, 5: 1, 7: 1, 11: 2} in ten_minute_paths
            #     winning_paths = [*filter(
            #         lambda p: aoc_2022.day_19_geode_count(p) == result,
            #         found_paths)]
            #     for path in winning_paths:
            #         print(path)

    def test_part_one(self):
        print("\n")
        solution = aoc_2022.day_19_part_one()
        lib.verify_solution(solution, 2301)
        assert solution > 2229  # (an improvement from 2197.  And 2212 . . . )
        # runs in approx. 02:07
        # keep an eye on Blueprint 17

    def test_p1_flexible_b_and_b(self):
        results = {}
        text = aoc_2022.Puzzle22(19).get_text_input()
        it = aoc_2022.day_19_blueprint_start_indices(text)
        for i, blueprint in enumerate(it):
            print(f"+++ BLUEPRINT {i + 1} +++")
            end_pos = it[i + 1] if i < len(it) - 1 else blueprint + 500
            aoc_2022.day_19_costs = aoc_2022.day_19_read_blueprint(text[blueprint:end_pos])
            b_and_b_run = aoc_2022.day_19_flexible_branch_and_bound(extend_by_mins=15)
            for duration in 3, 2:
                b_and_b_run = aoc_2022.day_19_flexible_branch_and_bound(b_and_b_run, duration)
            results[i + 1] = aoc_2022.day_19_evaluate_complete_paths(b_and_b_run)
        print(results)
        assert aoc_2022.day_19_get_total_score(results) == 2301
        # currently gives 2261

    def test_random_walk_all_blueprints(self):
        text = aoc_2022.Puzzle22(19).get_text_input()
        it = aoc_2022.day_19_blueprint_start_indices(text)
        for i, blueprint in enumerate(it):
            print(f"+++ BLUEPRINT {i + 1} +++")
            end_pos = it[i + 1] if i < len(it) - 1 else blueprint + 500
            aoc_2022.day_19_costs = aoc_2022.day_19_read_blueprint(text[blueprint:end_pos])
            aoc_2022.day_19_productive_paths = {}
            # for w in aoc_2022.day_19_random_walk({0: 0}):
            #     print(w)
            aoc_2022.day_19_random_walk({0: 0})
            results = len(aoc_2022.day_19_productive_paths)
            # rolls_of_the_dice = 1_000
            # while not results:
            #     print(f"{rolls_of_the_dice=}")
            #     aoc_2022.day_19_random_walk({0: 0}, n_walks=rolls_of_the_dice)
            #     results = len(aoc_2022.day_19_productive_paths)
            #     rolls_of_the_dice *= 10
            if i + 1 not in (7, 10, 22, 30) and (not results):
                print(*aoc_2022.day_19_productive_paths.keys())
                print(f"-> {results:>3} results vs. "
                      f"{self.true_geodes_per_blueprint[i + 1]:>3}")

    def test_full_random_walk_solution(self):
        aoc_2022.day_19_random_walk({0: 0})
        best_paths = [aoc_2022.day_19_productive_paths[t]
                      for t in sorted(aoc_2022.day_19_productive_paths.keys(),
                                      reverse=True)][:2]
        # pprint.pprint(aoc_2022.day_19_productive_paths)
        cutoff = 3
        best_stems = [{k: v for k, v in p.items() if k < cutoff} for p in best_paths]
        while cutoff < 24:
            for st in best_stems:
                aoc_2022.day_19_random_walk(st)
                best_paths = [aoc_2022.day_19_productive_paths[t]
                              for t in sorted(aoc_2022.day_19_productive_paths.keys(),
                                              reverse=True)][:2]
                best_stems = [{k: v for k, v in p.items() if k < cutoff} for p in best_paths]
                cutoff += 1
            print(f"{cutoff=} best: {max(aoc_2022.day_19_productive_paths.keys())}")
            # pprint.pprint(aoc_2022.day_19_productive_paths)
        assert max(aoc_2022.day_19_productive_paths.keys()) == 9

    def test_bfs(self):
        result = aoc_2022.day_19_breadth_first_search()
        assert result == 9

    def test_part_two(self):
        # assert aoc_2022.day_19_method_for_part_two() == 56
        # assert aoc_2022.day_19_geodes_found_on_random_walks(32) == 56
        aoc_2022.day_19_doing_part_two = True
        # assert aoc_2022.day_19_breadth_first_search() == 56
        # bp_2_text = self.example_text[self.example_text.index("Blueprint 2"):]
        # aoc_2022.day_19_costs = aoc_2022.day_19_read_blueprint(bp_2_text)
        # aoc_2022.day_19_set_robot_caps()
        # # assert aoc_2022.day_19_method_for_part_two() == 62
        # # currently fails on next line but only takes 04:33
        # assert aoc_2022.day_19_breadth_first_search() == 62
        proposed = aoc_2022.day_19_part_two()
        lib.verify_solution(proposed, part_two=True)
        assert proposed > 10064 # better answer 09/11/2023, takes 38.5sec
        ## 'best' results so far:
        #   {1: 38, 2: 16, 3: 16} in 55sec
        #       (after visiting 950,951 vertices in Blueprint 3)

        # next thing to try: BFS with sequence of robots as the vertices
        # (e.g. 0, 0, 1, 0, 1, 2 . . .)
        # or revisit calculating upper bounds properly, based on
        # triangular number geode counts (or just creating a new path
        # filled up with geode robots and using the standard geode count function)

    def test_p2_debug(self):
        aoc_2022.day_19_doing_part_two = True
        input_text = aoc_2022.Puzzle22(19).get_text_input()[160:320]
        aoc_2022.day_19_costs = aoc_2022.day_19_read_blueprint(input_text)
        aoc_2022.day_19_set_robot_caps()
        base_path = {0: 0, 5: 0, 8: 0, 10: 1, 11: 0,
                         12: 1, 13: 1, 14: 1, 15: 1,
                         16: 1, 17: 1, 18: 2, 19: 1}
        known_steps = {21: 2, 23: 2, 25: 3, 26: 2, 27: 3, 29: 3, 31: 3}
        for mm in range(20, 32):
            if mm in known_steps:
                base_path[mm] = known_steps[mm]
            else:
                valid_neighbours = aoc_2022.day_19_get_path_neighbours(base_path)
                if any((mm, aoc_2022.CLAY) in vn.items() for vn in valid_neighbours):
                    base_path[mm] = aoc_2022.CLAY
                else:
                    print(f"Cannot produce clay robot at minute {mm}")
        print(f"Final path: {base_path} ->\n{aoc_2022.day_19_geode_count(base_path)} geodes")


class TestDay18:
    example_input = """2,2,2
1,2,2
3,2,2
2,1,2
2,3,2
2,2,1
2,2,3
2,2,4
2,2,6
1,2,5
3,2,5
2,1,5
2,3,5"""

    def test_surface_area_calculations(self):
        test_pairs = {
            "1,1,1\n": 6,
            "1,1,1\n2,1,1": 10,
            self.example_input: 64,
        }
        for given, expected in test_pairs.items():
            cr = "\n"
            print(f"Testing {given.replace(cr, '; ')[:30]} . . .")
            assert aoc_2022.day_18_total_surface_area(
                aoc_2022.day_18_get_cube_set(given)
            ) == expected

    def test_part_one(self):
        solution = aoc_2022.day_18_part_one()
        lib.verify_solution(solution, 4340)
        # currently takes ~3.5sec

    def test_debug_for_part_two(self):
        # assert aoc_2022.day_18_part_two(self.example_input, 64) == 58
        cubes = aoc_2022.day_18_get_cube_set(self.example_input)
        assert aoc_2022.day_18_find_all_empty_regions(7, cubes)[1] == {(2, 2, 5)}
        minimum_enclosing_cubes = """2,2,4
2,2,6
1,2,5
3,2,5
2,1,5
2,3,5"""
        min_cubes = aoc_2022.day_18_get_cube_set(minimum_enclosing_cubes)
        assert len(min_cubes) == 6
        empty_regions = aoc_2022.day_18_find_all_empty_regions(10, min_cubes)
        assert len(empty_regions) == 2
        assert len(empty_regions[0]) == 1000 - 6 - 1
        assert empty_regions[1] == {(2, 2, 5)}

    def test_part_two(self):
        assert aoc_2022.day_18_part_two(self.example_input, 64) == 58
        p2_solution = aoc_2022.day_18_part_two()
        lib.verify_solution(p2_solution, part_two=True)
        assert p2_solution < 2486


class TestDay17:
    example_jets = ">>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>"   # len = 40

    def test_load(self):
        rock_cycle = aoc_2022.day_17_load_rocks()
        five_rocks = [next(rock_cycle) for _ in range(5)]
        assert len(five_rocks) == 5
        print(f"Rock widths: {', '.join(str(rock.width) for rock in five_rocks)}")
        assert [rock.width for rock in five_rocks] == [4, 3, 3, 1, 2]
        jet_cycle = aoc_2022.day_17_load_jets(self.example_jets)
        jets = [next(jet_cycle) for _ in range(60)]
        assert jets[7] == ">"
        assert "".join(jets[41:50]) == ">><<><>><"
        # full_cycle = aoc_2022.day_17_load_jets()
        # real_jets = [next(full_cycle) for _ in range(20000)]

    def test_horizontal_moves(self):
        base_rocks = aoc_2022.day_17_load_rocks()
        our_rock = next(base_rocks)
        our_rock.bottom_y = 4
        our_rock.move_horizontally(">")
        assert our_rock.left_x == 4
        for _ in range(10):
            our_rock.move_horizontally(">")
            assert our_rock.left_x == 4
        our_rock.move_horizontally("<")
        assert our_rock.left_x == 3
        for _ in range(2):
            expected_x = our_rock.left_x - 1
            our_rock.move_horizontally("<")
            assert our_rock.left_x == expected_x
        for _ in range(10):
            our_rock.move_horizontally("<")
            assert our_rock.left_x == 1

    def test_downward_move_hits_bottom(self):
        rock = next(aoc_2022.day_17_load_rocks())
        rock.bottom_y = 2
        for _ in range(10):
            rock.move_down()
            assert rock.bottom_y == 1
        rock.settle()

    def test_collision_detection(self):
        cyclical_rocks = aoc_2022.day_17_load_rocks()
        first_three_rocks = [next(cyclical_rocks) for _ in range(3)]
        rock_1, rock_2 = (first_three_rocks[r] for r in (1, 2))
        first_rock = first_three_rocks[0]
        first_rock.bottom_y = 1
        first_rock.settle()
        aoc_2022.day_17_stationary_rocks = [rock_1]
        rock_1.bottom_y = 2
        rock_1.settle()
        rock_2.bottom_y = 6
        rock_2.move_down()
        assert rock_2.bottom_y == 5
        rock_2.settle()
        rock_2.move_down()
        assert rock_2.bottom_y == 5

    def test_rockfall(self):
        assert aoc_2022.day_17_run_simulation(self.example_jets, 6) == 10
        assert aoc_2022.day_17_testing_count_hashes() == 26
        assert aoc_2022.day_17_run_simulation(self.example_jets, 10) == 17
        assert aoc_2022.day_17_run_simulation(self.example_jets) == 3068
        # print("\n".join(aoc_2022.day_17_cavern[::-1]))

    def test_debug_part_one(self):
        """This helped show that after a small number of rocks,
            the number of hashes in the cavern was not as expected,
            indicating that an empty part of a falling rock
            was sometimes overwriting an existing rock particle
            LESSON: think of everything you possibly can to test early on
            - could easily have spotted this with a simple test"""
        aoc_2022.day_17_run_simulation(self.example_jets, 22)
        print("\n".join(aoc_2022.day_17_cavern[::-1]))

    def test_part_one(self):
        proposed_solution = aoc_2022.day_17_part_one()
        # print("\n".join(aoc_2022.day_17_cavern[::-1]))
        lib.verify_solution(proposed_solution, 3239)

    def _exploratory_stuff_for_part_two(self):
        # does the pattern of rocks repeat itself?
        big_number = 1_000_000_000_000
        period = 2_783
        rocks_before_start = big_number % period
        print(f"Do {rocks_before_start} rows ")
        aoc_2022.day_17_run_simulation(self.example_jets, how_many_times=12_000)
        print("Simulation has run")
        cavern = aoc_2022.day_17_cavern
        random_rows = cavern[5000:5020]

        def match_bunch_of_rows(start_row: int) -> bool:
            for i, row in enumerate(random_rows):
                if cavern[start_row + i] != row:
                    return False
            return True

        for row_id in range(15_000):
            if row_id % 1_000 == 0:
                print(f"Looking at row {row_id}")
            if match_bunch_of_rows(row_id):
                print(f"Pattern repeats from row {row_id}")
        """Found repetitions of the 20-row segment at:
            (16132, 18915, 21698, 24481, 27264) - that is every 2,783 rows"""

    def test_part_two_example(self):
        solution = aoc_2022.day_17_height_of_trillion_rock_tower(self.example_jets)
        assert solution == 1_514_285_714_288

    def test_part_two(self):
        proposed = aoc_2022.day_17_part_two()
        assert proposed < 1596672403891
        lib.verify_solution(proposed, part_two=True)


class TestDay16:
    example = """Valve AA has flow rate=0; tunnels lead to valves DD, II, BB
Valve BB has flow rate=13; tunnels lead to valves CC, AA
Valve CC has flow rate=2; tunnels lead to valves DD, BB
Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE
Valve EE has flow rate=3; tunnels lead to valves FF, DD
Valve FF has flow rate=0; tunnels lead to valves EE, GG
Valve GG has flow rate=0; tunnels lead to valves FF, HH
Valve HH has flow rate=22; tunnel leads to valve GG
Valve II has flow rate=0; tunnels lead to valves AA, JJ
Valve JJ has flow rate=21; tunnel leads to valve II"""

    @pytest.fixture
    def eg_valve_data(self):
        return aoc_2022.day_16_load_valve_data(self.example)

    def test_setup(self):
        valves = aoc_2022.day_16_load_valve_data(self.example)
        assert len(valves) == 10
        assert valves["JJ"] == (21, ["II"])
        assert valves["EE"] == (3, ["FF", "DD"])

    def test_journey_scoring(self, eg_valve_data):
        with open("inputs_2022\\day16example_desc.txt", "r") as file:
            text = file.read()
        minutes = text.split("\n\n")
        moves = [mn.split("\n")[2][-3:-1] for mn in minutes if mn.count("\n") == 2]
        assert aoc_2022.day_16_score_journey([], eg_valve_data) == 0
        assert aoc_2022.day_16_score_journey(["AA", "BB", "CC"], eg_valve_data) == 0
        assert aoc_2022.day_16_score_journey(["BB", "BB", "CC", "CC"], eg_valve_data) == 364 + 52
        assert aoc_2022.day_16_score_journey(["DD", "DD"], eg_valve_data) == 20 * 28
        assert aoc_2022.day_16_score_journey(moves, eg_valve_data) == 1651

    def test_exploring_actual_input(self):
        valve_data = aoc_2022.day_16_load_valve_data(aoc_2022.Puzzle22(16).get_text_input())
        assert len(valve_data) == 59
        print(f"\nThere are {len([v for v in valve_data.values() if v[0] > 0])} "
              f"valves with a non-zero flow rate")

    def test_distances_table(self, eg_valve_data):
        assert aoc_2022.day_16_get_shortest_distance_between("AA", "BB", eg_valve_data) == 1
        assert aoc_2022.day_16_get_shortest_distance_between("AA", "CC", eg_valve_data) == 2
        distances = aoc_2022.day_16_build_distances_table(eg_valve_data)
        print(distances)
        assert distances["DD"]["JJ"] == 3
        assert distances["JJ"]["HH"] == 7
        assert distances["DD"]["JJ"] == distances["JJ"]["DD"]
        assert distances["AA"]["DD"] == 1
        assert len(distances) == 7
        assert all([len(distances[row]) == 6 for row in distances])
        for k in distances.keys():
            with pytest.raises(KeyError):
                print(distances[k][k])
        print("Building distances table for the big input . . .")
        big_table = aoc_2022.day_16_build_distances_table(aoc_2022.day_16_load_valve_data(aoc_2022.Puzzle22(16).get_text_input()))
        big_rows = len(big_table)
        print(f"The big table has {big_rows} rows")
        pprint.pprint(big_table)
        assert big_rows == 16
        assert "AA" in big_table
        # pprint.pprint(big_table)

    def test_brute_force_with_worthwhile_points(self, eg_valve_data):
        aoc_2022.day_16_route_table = aoc_2022.day_16_build_distances_table(eg_valve_data)
        valid_routes = aoc_2022.day_16_get_all_valid_routes
        assert valid_routes([""] * 30) == [[""] * 30]
        assert valid_routes([""] * 27 + ["AA"]) == [[""] * 27 + ["AA"]]
        assert valid_routes(["BB", "BB"] + [""] * 24 + ["CC"]) == \
               [["BB", "BB"] + [""] * 24 + ["CC"] + ["DD", "DD"]]
        stem = [""] * 26 + ["AA"]
        assert valid_routes(stem) == [stem + ["BB", "BB"], stem + ["DD", "DD"]]
        all_routes = valid_routes([])
        print(f"There are {len(all_routes)} valid routes in the example")
        print(f"Solution could be "
              f"{max([aoc_2022.day_16_score_journey(r, eg_valve_data) for r in all_routes])}")
        print(f"That is "
              f"{max(all_routes, key=lambda r: aoc_2022.day_16_score_journey(r, eg_valve_data))}")

    def test_permutation_based_journey_lengths(self, eg_valve_data):
        aoc_2022.day_16_route_table = aoc_2022.day_16_build_distances_table(eg_valve_data)
        assert aoc_2022.day_16_journey_time(("AA", "BB")) == 2
        assert aoc_2022.day_16_journey_time(("AA", "CC")) == 3
        assert aoc_2022.day_16_journey_time(("BB", "DD", "JJ")) == 7

    def test_using_perms_and_combs(self, eg_valve_data):
        aoc_2022.day_16_best_journey_using_permutations(#eg_valve_data)
            aoc_2022.day_16_load_valve_data(aoc_2022.Puzzle22(16).get_text_input()))

    def test_tuple_journey_scoring(self, eg_valve_data):
        aoc_2022.day_16_route_table = aoc_2022.day_16_build_distances_table(eg_valve_data)
        assert aoc_2022.day_16_score_tuple_journey((), eg_valve_data) == 0
        assert aoc_2022.day_16_score_tuple_journey(("AA", "BB", "CC"), eg_valve_data) == 364 + 52
        assert aoc_2022.day_16_score_tuple_journey(("AA", "DD"), eg_valve_data) == 20 * 28
        with open("inputs_2022\\day16example_desc.txt", "r") as file:
            text = file.read()
        activity_by_minute = text.split("\n\n")
        moves = ["AA"]
        moves += [act.split("\n")[2][-3:-1] for act in activity_by_minute
                  if act.count("\n") == 2
                  and act[-14:-10] == "open"]
        journey = tuple(moves)
        assert aoc_2022.day_16_score_tuple_journey(journey, eg_valve_data) == 1651

    def test_one_step_at_a_time(self, eg_valve_data):
        aoc_2022.day_16_route_table = aoc_2022.day_16_build_distances_table(eg_valve_data)
        nr = aoc_2022.day_16_best_journey_by_taking_it_one_step_at_a_time(eg_valve_data)
        print(nr)
        assert len(set(nr)) == len(nr)
        best_route = max(nr, key=lambda r: aoc_2022.day_16_score_tuple_journey(r, eg_valve_data))
        assert aoc_2022.day_16_score_tuple_journey(best_route, eg_valve_data) == 1651

    def test_part_one(self, eg_valve_data):
        aoc_2022.day_16_route_table = aoc_2022.day_16_build_distances_table(eg_valve_data)
        assert aoc_2022.day_16_by_traversal_of_all_routes_between_worthwhile_points(self.example) == 1651
        solution = aoc_2022.day_16_part_one()
        assert solution < 1897  # first incorrect attempt
        lib.verify_solution(solution, 1789)

    def test_scoring_double_headed_journeys(self, eg_valve_data):
        method = aoc_2022.day_16_score_double_headed_journey
        assert method([[], []], eg_valve_data) == 0
        j1 = [["DD", "DD"], []]
        assert method(j1, eg_valve_data) == 480
        j2 = [[""] * 8 + ["EE", "EE"], ["DD", "DD"]]
        assert method(j2, eg_valve_data) == 480 + 48
        with open("inputs_2022\\day16eg2.txt") as eg_file:
            text = eg_file.readlines()
        my_journey, elephant_journey = [], []
        for line in text:
            match line[:4]:
                case "You ":
                    my_journey.append(line[-4:-2])
                case "The ":
                    elephant_journey.append(line[-4:-2])
        best_eg_journey = [my_journey, elephant_journey]
        assert method(best_eg_journey, eg_valve_data) == 1707

    def test_double_headed_route_finding(self, eg_valve_data):
        aoc_2022.day_16_route_table = aoc_2022.day_16_build_distances_table(eg_valve_data)
        test_method = aoc_2022.day_16_get_all_valid_team_routes

        def blanks(n_empties: int) -> [str]:
            return [""] * n_empties
        me_stem, e_stem = blanks(23) + ["BB"], blanks(23) + ["CC"]
        assert test_method([me_stem, e_stem]) == [[me_stem, e_stem]]    # short of time
        me_stem, e_stem = blanks(22) + ["CC"], blanks(22) + ["DD"]
        assert test_method([me_stem, e_stem]) == [[me_stem + ["BB", "BB"],  # only one
                                                   e_stem + ["EE", "EE"]]]  # option each
        more_options = test_method([["BB", "CC"], ["DD", "EE"]])
        assert len(more_options) > 1
        assert len(more_options) == 2
        pprint.pprint(more_options)
        extended_options = test_method([["BB"], ["DD"]])
        assert len(extended_options) > 2
        # pprint.pprint(extended_options)
        # for xe in extended_options:
        #     xe = [[*filter(lambda s: len(s) > 0, el)] for el in xe]
        #     print(xe)
        #     overlap = set(xe[0]).intersection(set(xe[1]))
        #     assert len(overlap) == 0
        # test for unequal options because in example, each player always has same no. (?)
        unequal_options = test_method([["", "", "BB"],
                                       ["CC", "DD", "EE"] + blanks(21) + ["HH"]])
        assert len(unequal_options) == 1
        assert unequal_options[0][0] == ["", "", "BB"] + blanks(2) + ["JJ", "JJ"]
        assert unequal_options[0][1] == ["CC", "DD", "EE"] + blanks(21) + ["HH"]
        no_options = test_method([["CC", "DD", "EE"], ["BB", "HH", "JJ"]])
        assert no_options == [[["CC", "DD", "EE"], ["BB", "HH", "JJ"]]]

    def test_part_two(self, eg_valve_data):
        aoc_2022.day_16_route_table = aoc_2022.day_16_build_distances_table(eg_valve_data)
        assert aoc_2022.day_16_score_double_headed_tuple_journey(
            (("AA", "JJ", "BB", "CC", ), ("AA", "DD", "HH", "EE", )), eg_valve_data
        ) == 1707
        eg_solution = aoc_2022.day_16_by_teaming_up_with_elephant(self.example)
        assert eg_solution == 1707
        lib.verify_solution(aoc_2022.day_16_part_two(), part_two=True)

        # TODO: IDEAS
        #   What to do next?  Give me the best combinations of three(?) steps
        #           that move things on the most
        #   Inputs: pool of available points to visit (or robots to build - complicated)
        #           starting conditions, i.e. production rates
        #               would anything be gained by representing paths between valves
        #               in the same way as a production timeline?
        #   Chain together sets of three steps


class TestDay15:
    example = """Sensor at x=2, y=18: closest beacon is at x=-2, y=15
Sensor at x=9, y=16: closest beacon is at x=10, y=16
Sensor at x=13, y=2: closest beacon is at x=15, y=3
Sensor at x=12, y=14: closest beacon is at x=10, y=16
Sensor at x=10, y=20: closest beacon is at x=10, y=16
Sensor at x=14, y=17: closest beacon is at x=10, y=16
Sensor at x=8, y=7: closest beacon is at x=2, y=10
Sensor at x=2, y=0: closest beacon is at x=2, y=10
Sensor at x=0, y=11: closest beacon is at x=2, y=10
Sensor at x=20, y=14: closest beacon is at x=25, y=17
Sensor at x=17, y=20: closest beacon is at x=21, y=22
Sensor at x=16, y=7: closest beacon is at x=15, y=3
Sensor at x=14, y=3: closest beacon is at x=15, y=3
Sensor at x=20, y=1: closest beacon is at x=15, y=3
"""

    def test_setup(self):
        data = aoc_2022.day_15_load_sensor_beacon_data(self.example)
        assert len(data) == 14
        beacons = {*data.values()}
        assert len(beacons) == 6

    def test_method_for_part_one(self):
        assert aoc_2022.day_15_count_positions_without_sensor(self.example, 10) == 26

    def test_part_one(self):
        lib.verify_solution(aoc_2022.day_15_part_one(), 4737567)

    def test_build_up_to_part_two(self):
        data = aoc_2022.day_15_load_sensor_beacon_data(self.example)
        assert not aoc_2022.day_15_point_is_not_reached_by_any_sensor(data, lib.Point(0, 0))
        assert aoc_2022.day_15_point_is_not_reached_by_any_sensor(data, lib.Point(14, 11))
        assert aoc_2022.day_15_get_gradients_and_intercepts(lib.Point(8, 7),
                                                            lib.Point(2, 10)) == \
               [(1, -11), (1, 9), (-1, 5), (-1, 25)]
        assert aoc_2022.day_15_get_gradients_and_intercepts(lib.Point(9, 16),
                                                            lib.Point(10, 16)) == \
               [(1, 5), (1, 9), (-1, 23), (-1, 27)]
        assert aoc_2022.day_15_get_gradients_and_intercepts(lib.Point(0, 0),
                                                            lib.Point(5, 0)) == \
               [(1, -6), (1, 6), (-1, -6), (-1, 6)]
        assert aoc_2022.day_15_get_gradients_and_intercepts(lib.Point(5, 0),
                                                            lib.Point(0, 0)) == \
               [(1, -11), (1, 1), (-1, -1), (-1, 11)]
        intersections = aoc_2022.day_15_find_periphery_intersections(lib.Point(8, 7),
                                                                    lib.Point(9, 16), data)
        print(intersections)
        result = aoc_2022.day_15_find_single_blind_spot(self.example)
        assert result == lib.Point(14, 11)
        assert aoc_2022.day_15_tuning_frequency(result) == 56000011

    def test_part_two(self):
        answer = aoc_2022.day_15_part_two()
        assert answer > 2110007570474
        lib.verify_solution(answer, 13267474686239, part_two=True)
        # 2110007570474 is too low


class TestDay14:
    example = """498,4 -> 498,6 -> 496,6
503,4 -> 502,4 -> 502,9 -> 494,9"""

    def test_setup(self):
        rock_points = aoc_2022.day_14_create_lines("498,4 -> 498,6 -> 496,6")
        print("\n", rock_points)
        assert len(rock_points) == 5
        expected_points = [lib.Point(x=498, y=4), lib.Point(x=498, y=5),
                           lib.Point(x=498, y=6), lib.Point(497, 6), lib.Point(496, 6)]
        assert rock_points == expected_points
        all_points = aoc_2022.day_14_load_all_points(self.example)
        assert len(all_points) == 20
        assert max([pt.y for pt in all_points]) == 9

    def test_sand_dropping(self):
        rock = aoc_2022.day_14_load_all_points(self.example)
        initial_size = len(rock)
        new_configuration = aoc_2022.day_14_drop_sand_particle(rock)
        assert len(new_configuration) == initial_size + 1
        config_2 = aoc_2022.day_14_drop_sand_particle(new_configuration)
        assert len(config_2) == initial_size + 2
        longer_term_config = config_2
        for _ in range(50):
            longer_term_config = aoc_2022.day_14_drop_sand_particle(longer_term_config)
        assert len(longer_term_config) == initial_size + 24

    def test_part_one(self):
        lib.verify_solution(aoc_2022.day_14_part_one(), 961)

    def test_method_for_part_two(self):
        blocked = aoc_2022.day_14_load_all_points(self.example)
        rocks = [*blocked]
        aoc_2022.day_14_infinite_floor_level = max([pt.y for pt in blocked]) + 2
        units_retained = 0
        while lib.Point(500, 0) not in blocked:
            blocked = aoc_2022.day_14_drop_particle_onto_infinite_floor(blocked)
            units_retained += 1

        def display_char(point: lib.Point) -> str:
            if point in rocks:
                return "#"
            if point in blocked:
                return "o"
            return "."
        print("\n")
        for y in range(12):
            print("".join(display_char(lib.Point(x, y)) for x in range(489, 515)))
        assert units_retained == 93

    def test_part_two(self):
        lib.verify_solution(aoc_2022.day_14_part_two(), 26375, part_two=True)


class TestDay13:
    example = """[1,1,3,1,1]
[1,1,5,1,1]

[[1],[2,3,4]]
[[1],4]

[9]
[[8,7,6]]

[[4,4],4,4]
[[4,4],4,4,4]

[7,7,7,7]
[7,7,7]

[]
[3]

[[[]]]
[[]]

[1,[2,[3,[4,[5,6,7]]]],8,9]
[1,[2,[3,[4,[5,6,0]]]],8,9]"""

    def test_setup(self):
        # pairs = aoc_2022.day_13_load_pairs(self.example)
        pairs = aoc_2022.day_13_eval_load_pairs(self.example)
        print(pairs)
        assert len(pairs) == 8
        assert all([len(pr) == 2 for pr in pairs])
        assert all([isinstance(item, list) for item in pairs[1]])
        print(pairs[1])
        assert all([len(item) == 2 for item in pairs[1]])
        left_1, right_1 = pairs[1]
        assert isinstance(left_1[0], list)
        assert left_1[0][0] == 1
        assert right_1[1] == 4
        last_pair = pairs[-1]
        print(f"Last pair: {last_pair}")

    def test_comparisons(self):
        assert aoc_2022.day_13_order_is_correct(1, 2) == (True, True)
        assert aoc_2022.day_13_order_is_correct(1, [2]) == (True, True)
        assert aoc_2022.day_13_order_is_correct([1,1,3,1,1], [1,1,5,1,1]) == (True, True)
        assert aoc_2022.day_13_order_is_correct([[1],[2,3,4]], [[1],4]) == (True, True)
        assert aoc_2022.day_13_order_is_correct([9], [[8,7,6]]) == (True, False)
        assert aoc_2022.day_13_order_is_correct([[4,4],4,4], [[4,4],4,4,4]) == (True, True)
        assert aoc_2022.day_13_order_is_correct([7,7,7,7], [7,7,7]) == (True, False)
        assert aoc_2022.day_13_order_is_correct([], [3]) == (True, True)
        assert aoc_2022.day_13_order_is_correct([[[]]], [[]]) == (True, False)
        assert aoc_2022.day_13_order_is_correct([1,[2,[3,[4,[5,6,7]]]],8,9],
                                                [1,[2,[3,[4,[5,6,0]]]],8,9]) == (True, False)
        assert aoc_2022.day_13_get_sum_of_indices_of_correctly_ordered_pairs(self.example) == 13

    def test_part_one(self):
        lib.verify_solution(aoc_2022.day_13_part_one(), 5720)

    def test_part_two(self):
        assert aoc_2022.day_13_insert_markers(self.example) == 140
        lib.verify_solution(aoc_2022.day_13_part_two(), 23504, part_two=True)


class TestDay12:
    example = """Sabqponm
abcryxxl
accszExk
acctuvwj
abdefghi"""

    def test_setup(self):
        aoc_2022.day_12_start(self.example)
        assert aoc_2022.day_12_find_terminus() == lib.Point(0, 0)
        print(aoc_2022.day_12_get_valid_options(lib.Point(0, 4)))
        print(aoc_2022.day_12_get_valid_options(lib.Point(6, 0)))
        assert len(aoc_2022.day_12_get_valid_options(lib.Point(5, 3))) == 3

    def test_dijkstra(self):
        aoc_2022.day_12_start(self.example)
        starting_point = aoc_2022.day_12_find_terminus()
        assert aoc_2022.day_12_dijkstra_shortest_distance(starting_point) == 31
        part_two_candidates = [aoc_2022.day_12_dijkstra_shortest_distance(pt)
                               for pt in aoc_2022.day_12_find_all("a")]
        p2_example_solution = min(part_two_candidates)
        assert p2_example_solution == 29

    def test_part_one(self):
        lib.verify_solution(aoc_2022.day_12_part_one(), 352)

    def test_part_two(self):
        lib.verify_solution(aoc_2022.day_12_part_two(), part_two=True)


class TestDay11:
    example_notes = """Monkey 0:
  Starting items: 79, 98
  Operation: new = old * 19
  Test: divisible by 23
    If true: throw to monkey 2
    If false: throw to monkey 3

Monkey 1:
  Starting items: 54, 65, 75, 74
  Operation: new = old + 6
  Test: divisible by 19
    If true: throw to monkey 2
    If false: throw to monkey 0

Monkey 2:
  Starting items: 79, 60, 97
  Operation: new = old * old
  Test: divisible by 13
    If true: throw to monkey 1
    If false: throw to monkey 3

Monkey 3:
  Starting items: 74
  Operation: new = old + 3
  Test: divisible by 17
    If true: throw to monkey 0
    If false: throw to monkey 1"""

    def test_monkey_methods(self):
        m = aoc_2022.Monkey("""1:
  Starting items: 54, 65, 75, 74
  Operation: new = old + 6
  Test: divisible by 19
    If true: throw to monkey 2
    If false: throw to monkey 0""")
        assert m.held_items == [54, 65, 75, 74]
        # m.take_turn()

    def test_build_a_lambda_function(self):
        text = """  Test: divisible by 17
    If true: throw to monkey 0
    If false: throw to monkey 1"""
        divisor, true_destination, false_destination = (int(line.split(" ")[-1])
                                                        for line in text.split("\n"))
        assert (divisor, true_destination, false_destination) == (17, 0, 1)
        func = lambda x: false_destination if x % divisor else true_destination

    def test_setup(self):
        aoc_2022.day_11_initialise(self.example_notes)
        assert len(aoc_2022.day_11_monkeys) == 4
        assert aoc_2022.day_11_monkeys[2].held_items == [79, 60, 97]
        aoc_2022.day_11_monkeys[0].take_turn()
        assert aoc_2022.day_11_monkeys[0].held_items == []
        assert aoc_2022.day_11_monkeys[3].held_items == [74, 500, 620]
        aoc_2022.day_11_play_round()
        assert aoc_2022.day_11_monkeys[0].held_items == [20, 23, 27, 26]
        assert aoc_2022.day_11_monkeys[1].held_items == [2080, 25, 167, 207, 401, 1046]
        assert aoc_2022.day_11_monkeys[2].held_items == []
        assert aoc_2022.day_11_monkeys[3].held_items == []

    def test_part_one(self):
        aoc_2022.day_11_initialise(self.example_notes)
        for _ in range(20):
            aoc_2022.day_11_play_round()
        assert aoc_2022.day_11_monkeys[0].held_items == [10, 12, 14, 26, 34]
        assert aoc_2022.day_11_monkeys[1].held_items == [245, 93, 53, 199, 115]
        assert aoc_2022.day_11_monkeys[2].held_items == []
        assert aoc_2022.day_11_monkeys[3].held_items == []
        assert [m.throw_count for m in aoc_2022.day_11_monkeys] == [101, 95, 7, 105]
        assert aoc_2022.day_11_part_one(self.example_notes) == 10605
        lib.verify_solution(aoc_2022.day_11_part_one(aoc_2022.Puzzle22(11).get_text_input()),
                        55930)

    def test_monkey_patching(self):
        aoc_2022.day_11_initialise(self.example_notes)
        aoc_2022.day_11_monkey_patch()
        assert aoc_2022.day_11_monkeys[0].inspect_item(1) == 19

    def test_part_two(self):
        assert aoc_2022.day_11_part_two(self.example_notes) == 2713310158


class TestDay10:
    trivial_example = """noop
addx 3
addx -5"""
    example = """addx 15
addx -11
addx 6
addx -3
addx 5
addx -1
addx -8
addx 13
addx 4
noop
addx -1
addx 5
addx -1
addx 5
addx -1
addx 5
addx -1
addx 5
addx -1
addx -35
addx 1
addx 24
addx -19
addx 1
addx 16
addx -11
noop
noop
addx 21
addx -15
noop
noop
addx -3
addx 9
addx 1
addx -3
addx 8
addx 1
addx 5
noop
noop
noop
noop
noop
addx -36
noop
addx 1
addx 7
noop
noop
noop
addx 2
addx 6
noop
noop
noop
noop
noop
addx 1
noop
noop
addx 7
addx 1
noop
addx -13
addx 13
addx 7
noop
addx 1
addx -33
noop
noop
noop
addx 2
noop
noop
noop
addx 8
noop
addx -1
addx 2
addx 1
noop
addx 17
addx -9
addx 1
addx 1
addx -3
addx 11
noop
noop
addx 1
noop
addx 1
noop
noop
addx -13
addx -19
addx 1
addx 3
addx 26
addx -30
addx 12
addx -1
addx 3
addx 1
noop
noop
noop
addx -9
addx 18
addx 1
addx 2
noop
noop
addx 9
noop
noop
noop
addx -1
addx 2
addx -37
addx 1
addx 3
noop
addx 15
addx -21
addx 22
addx -6
addx 1
noop
addx 2
addx 1
noop
addx -10
noop
noop
addx 20
addx 1
addx 2
addx 2
addx -6
addx -11
noop
noop
noop"""
    correct_image = """##..##..##..##..##..##..##..##..##..##..
###...###...###...###...###...###...###.
####....####....####....####....####....
#####.....#####.....#####.....#####.....
######......######......######......####
#######.......#######.......#######.....
"""

    def test_step_by_step(self):
        dic = aoc_2022.day_10_value_after_cycle_completions(self.trivial_example)
        assert dic == {1: 1, 3: 4, 5: -1}
        expected_results = {1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: -1}
        for er in expected_results:
            assert aoc_2022.day_10_find_value_during_cycle(er, dic) == expected_results[er]
        assert aoc_2022.day_10_get_aggregate_signal_strength(self.trivial_example) == \
               sum(-1 * v for v in range(20, 240, 40))
        assert aoc_2022.day_10_get_aggregate_signal_strength(self.example) == 13140

    def test_part_one(self):
        lib.verify_solution(aoc_2022.day_10_part_one())

    def test_part_two(self):
        print("\n")
        history = aoc_2022.day_10_value_after_cycle_completions(self.example)
        example_image = aoc_2022.day_10_render_image(history)
        print(example_image)
        assert example_image == self.correct_image
        aoc_2022.day_10_part_two()


class TestDay9:
    example = """R 4
U 4
L 3
D 1
R 4
D 1
L 5
R 2"""
    part_2_example = """R 5
U 8
L 8
D 3
R 17
D 10
L 25
U 20"""

    def test_setup(self):
        pt = lib.Point
        test_point = pt(100, 100)
        new_point = lib.point_moves["U"](test_point)
        assert new_point == lib.Point(100, 101)
        new_head, _ = aoc_2022.day_9_make_move((pt(50, 50), pt(0, 0)), "L 5")
        assert new_head == lib.Point(45, 50)
        assert aoc_2022.day_9_make_journey(self.example.split("\n")) == 13

    def test_part_one(self):
        lib.verify_solution(aoc_2022.day_9_part_one(), 5930)

    def test_long_rope_moves(self):
        pt = lib.Point
        origin = [pt(0, 0) for _ in range(10)]
        assert origin[-1] == pt(0, 0)
        new_points = aoc_2022.day_9_make_move(origin, "R 4")
        assert new_points == tuple([pt(x, 0) for x in range(4, 0, -1)] +
                                   [pt(0, 0) for _ in range(6)])
        assert aoc_2022.day_9_make_journey(self.example.split("\n"), 10) == 1
        assert aoc_2022.day_9_make_journey(self.part_2_example.split("\n"), 10) == 36

    def test_part_two(self):
        lib.verify_solution(aoc_2022.day_9_part_two(), 2443, part_two=True)


class TestDay8:
    example = """30373
25512
65332
33549
35390"""

    def test_set_up(self):
        grid = aoc_2022.day_8_make_grid(self.example)
        print(grid)
        assert len(grid) == 5
        assert len(grid[1]) == 5
        assert all([all(isinstance(tree, int) for tree in row) for row in grid])
        big_grid = aoc_2022.day_8_make_grid(aoc_2022.Puzzle22(8).get_text_input())
        print(f"The main grid is a {len(big_grid)} x {len(big_grid[1])} grid")

    def test_p1_example(self):
        grid = aoc_2022.day_8_make_grid(self.example)
        assert aoc_2022.day_8_count_visible_trees(grid) == 21

    def test_part_one(self):
        p1_grid = aoc_2022.day_8_make_grid(aoc_2022.Puzzle22(8).get_text_input())
        lib.verify_solution(aoc_2022.day_8_count_visible_trees(p1_grid), 1870)

    def test_scenic_scores(self):
        test_grid = aoc_2022.day_8_make_grid(self.example)
        assert aoc_2022.day_8_calculate_scenic_score(0, 4, test_grid) == 0
        assert aoc_2022.day_8_calculate_scenic_score(4, 4, test_grid) == 0
        assert aoc_2022.day_8_calculate_scenic_score(1, 2, test_grid) == 4
        assert aoc_2022.day_8_calculate_scenic_score(3, 2, test_grid) == 8
        assert aoc_2022.day_8_part_two(test_grid) == 8

    def test_part_two(self):
        big_grid = aoc_2022.day_8_make_grid(aoc_2022.Puzzle22(8).get_text_input())
        lib.verify_solution(aoc_2022.day_8_part_two(big_grid), 517440, part_two=True)


class TestDay7:
    example = """$ cd /
$ ls
dir a
14848514 b.txt
8504156 c.dat
dir d
$ cd a
$ ls
dir e
29116 f
2557 g
62596 h.lst
$ cd e
$ ls
584 i
$ cd ..
$ cd ..
$ cd d
$ ls
4060174 j
8033020 d.log
5626152 d.ext
7214296 k"""

    def test_building_structure(self):
        files = aoc_2022.day_7_build_directory_structure(self.example)
        assert len(files) == 4
        assert files[2][3][0] == 584

    def test_p1_example(self):
        files = aoc_2022.day_7_build_directory_structure(self.example)
        assert aoc_2022.day_7_get_total_size_of_small_directories(files) == 95437

    def test_part_one(self):
        lib.verify_solution(aoc_2022.day_7_part_one(), 1118405)

    def test_p2_example(self):
        files = aoc_2022.day_7_build_directory_structure(self.example)
        assert aoc_2022.day_7_space_needed_to_be_freed(files) == 30_000_000 - 21618835
        assert aoc_2022.day_7_get_smallest_directory_to_delete(files) == 24933642

    def test_part_two(self):
        lib.verify_solution(aoc_2022.day_7_part_two(), part_two=True)


class TestDay6:
    first_example = "mjqjpqmgbljsphdztnvjfqwrcgsmlb"
    examples = """bvwbjplbgvbhsrlpgdmjqwftvncz: first marker after character 5
nppdvjthqldpwncqszvftbrmjlhg: first marker after character 6
nznrnfrfntjfmvfwmzdfjlvtqnbhcprsg: first marker after character 10
zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw: first marker after character 11"""

    def test_examples(self):
        assert aoc_2022.day_6_get_marker_end(self.first_example) == 7
        example_lines = self.examples.split("\n")
        e_dict = {ln.split(":")[0]: int(ln.split(" ")[-1]) for ln in example_lines}
        for buffer, expected in e_dict.items():
            assert aoc_2022.day_6_get_marker_end(buffer) == expected
        assert aoc_2022.day_6_get_message_start(self.first_example) == 19
        p2_expected = [23, 23, 29, 26]
        for p2_buffer, exp in zip(e_dict.keys(), p2_expected):
            assert aoc_2022.day_6_get_message_start(p2_buffer) == exp

    def test_part_one(self):
        lib.verify_solution(aoc_2022.day_6_part_one(), correct=1702)

    def test_part_two(self):
        lib.verify_solution(aoc_2022.day_6_part_two(), correct=3559, part_two=True)


class TestDay5:
    example = """    [D]    
[N] [C]    
[Z] [M] [P]
 1   2   3 

move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2"""

    def test_get_starting_config(self):
        starting = aoc_2022.day_5_get_starting_configuration(self.example)
        assert starting[1] == "ZN"
        assert starting[2] == "MCD"
        assert starting[3] == "P"
        assert len(starting) == 3

    def test_interpreting_moves(self):
        assert aoc_2022.day_5_interpret_move("move 3 from 1 to 3\n") == (3, 1, 3)
        assert aoc_2022.day_5_interpret_move("move 26 from 1 to 7") == (26, 1, 7)

    def test_make_move(self):
        config = {1: "AD", 2: "B"}
        new_config = aoc_2022.day_5_make_move((1, 1, 2), config)
        assert new_config[1] == "A"
        assert new_config[2] == "BD"
        bigger_config = {1: "AJFAKSLDFJ", 3: "ALKFJIDHG"}
        final_config = aoc_2022.day_5_make_move((4, 3, 1), bigger_config)
        assert final_config[1] == "AJFAKSLDFJGHDI"
        assert final_config[3] == "ALKFJ"

    def test_example(self):
        assert aoc_2022.day_5_part_one(self.example) == "CMZ"
        assert aoc_2022.day_5_part_two(self.example) == "MCD"

    def test_part_one(self):
        lib.verify_solution(aoc_2022.day_5_part_one(aoc_2022.Puzzle22(5).get_text_input()),
                        correct="VWLCWGSDQ")

    def test_part_two(self):
        lib.verify_solution(aoc_2022.day_5_part_two(aoc_2022.Puzzle22(5).get_text_input()),
                        correct="TCGLQSLPW", part_two=True)



class TestDay4:
    example = """2-4,6-8
2-3,4-5
5-7,7-9
2-8,3-7
6-6,4-6
2-6,4-8"""

    def test_begin(self):
        print("\nhello")
        inputs = aoc_2022.Puzzle22(4).convert_input(self.example, aoc_2022.day_4_split_function)
        answers = [False for _ in range(3)] + [True for _ in range(2)] + [False]
        assert [aoc_2022.day_4_one_wholly_contains_other(inp) for inp in inputs] == answers
        assert aoc_2022.day_4_part_one(inputs) == 2
        p2_answers = [False for _ in range(2)] + [True for _ in range(4)]
        assert [aoc_2022.day_4_any_overlap(inp) for inp in inputs] == p2_answers
        assert aoc_2022.day_4_part_two(inputs) == 4

    def test_part_one(self):
        pairings = aoc_2022.Puzzle22(4).input_as_list(aoc_2022.day_4_split_function)
        lib.verify_solution(aoc_2022.day_4_part_one(pairings), 441)

    def test_part_two(self):
        pairings = aoc_2022.Puzzle22(4).input_as_list(aoc_2022.day_4_split_function)
        lib.verify_solution(aoc_2022.day_4_part_two(pairings), 0, True)


class TestDay3:
    example = """vJrwpWtwJgWrhcsFMMfFFhFp
jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL
PmmdzqPrVvPwwTWBwg
wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn
ttgJtRGJQctTZtZT
CrZsJsPPZsGzwwsLwLmpwMDw"""

    def test_example(self):
        rucksack_list = aoc_2022.Puzzle22(3).convert_input(self.example, None)
        print("")
        assert aoc_2022.day_3_get_priority_for_rucksack(rucksack_list[0]) == 16
        assert aoc_2022.day_3_get_priority_for_rucksack(rucksack_list[1]) == 38
        assert aoc_2022.day_3_get_priority_for_rucksack("AbAc") == 27
        assert aoc_2022.day_3_get_priority_for_rucksack("AZZc") == 52
        assert aoc_2022.day_3_get_priority_for_rucksack("abca") == 1
        assert aoc_2022.day_3_get_priority_for_rucksack("zzzP") == 26
        assert aoc_2022.day_3_part_one(rucksack_list) == 157

    def test_part_one(self):
        rucksack_list = aoc_2022.Puzzle22(3).input_as_list(None)
        solution = aoc_2022.day_3_part_one(rucksack_list)
        lib.verify_solution(solution, 7863)

    def test_p_2_example(self):
        assert aoc_2022.day_3_get_priority_for_group(["abck", "abc", "azzp"]) == 1
        assert aoc_2022.day_3_get_priority_for_group(["aZbcd", "apZi", "ZZZ"]) == 52
        rucksack_list = aoc_2022.Puzzle22(3).convert_input(self.example, None)
        assert aoc_2022.day_3_part_two(rucksack_list) == 70


    def test_part_two(self):
        rucksack_list = aoc_2022.Puzzle22(3).input_as_list(None)
        solution = aoc_2022.day_3_part_two(rucksack_list)
        lib.verify_solution(solution, 0, part_two=True)


class TestDay2:
    example = """A Y
B X
C Z"""

    def test_example(self):
        puzz = aoc_2022.Puzzle22(2)
        rounds = puzz.convert_input(self.example, None)
        print(rounds)
        for r in rounds:
            aoc_2022.day_2_score_round(r)
        assert aoc_2022.day_2_part_one(rounds) == 15
        assert aoc_2022.day_2_part_two(rounds) == 12

    def test_part_one(self):
        puzzle = aoc_2022.Puzzle22(2)
        input_list = puzzle.input_as_list(None)
        solution = aoc_2022.day_2_part_one(input_list)
        print(f"Answer: {solution}")
        assert solution == 15422

    def test_part_two(self):
        puzzle = aoc_2022.Puzzle22(2)
        solution = aoc_2022.day_2_part_two(puzzle.input_as_list(None))
        print(f"Part Two solution is {solution}")
        assert solution > 0


class TestDayOne:
    def test_part_one(self):
        solution = aoc_2022.day_1_part_one()
        print(f"Answer is {solution}")
        assert solution == 72602

    def test_part_two(self):
        answer = aoc_2022.day_1_part_two()
        print(f"The answer to part two is {answer}")
        assert answer == 207410


if __name__ == "__main__":
    TestDayOne().test_part_one()
