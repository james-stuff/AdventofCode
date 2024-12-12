import collections
import copy
import math
from main import Puzzle
import re
import library as lib
from aoc_2023 import point_moves_2023 as pm23
import itertools
import operator


class Puzzle24(Puzzle):
    def get_text_input(self) -> str:
        with open(f'inputs\\2024\\input{self.day}.txt', 'r') as input_file:
            return input_file.read()


def day_12_load_map_data() -> [{}]:
    text = Puzzle24(12).get_text_input()
    return {
        ch: {m.start() for m in
             re.finditer(ch, text)}
        for ch in {*text}
    }


def day_12_find_regions(map_data: {}) -> {}:
    regions = {}
    for letter in map_data:
        if letter != "\n":
            l_regions = []
            letter_inst = {*map_data[letter]}
            while letter_inst:
                pos = [*letter_inst][0]
                block = day_12_all_contiguous(pos, map_data)
                l_regions.append(block)
                letter_inst -= block
            regions[letter] = l_regions
    return regions


def day_12_all_contiguous(pos: int, map_data: {}, existing: {} = None) -> {int}:
    if existing is None:
        existing = set()
    if day_12_like_neighbours(pos, map_data):
        contiguous = {pos}
        existing.add(pos)
        n_sets = [day_12_all_contiguous(p1, map_data, existing)
                  for p1 in day_12_like_neighbours(pos, map_data)
                  if p1 not in existing]
        for s in n_sets:
            contiguous.update(s)
        return contiguous
    return {pos}


def day_12_like_neighbours(pos: int, map_data: {},
                           inverse: bool = False, text: str = "") -> {int}:
    letter = [k for k, v in map_data.items()
              if pos in v][0]
    if inverse:
        return {
            pp for pp in day_12_all_neighbours(pos, map_data)
            if pp < 0 or pp >= len(text)
               or pp in map_data["\n"]
               or text[pp] != letter
        }
    occurrences = map_data[letter]
    return day_12_all_neighbours(pos, map_data).intersection(occurrences)


def day_12_all_neighbours(pos: int, map_data: {}) -> {int}:
    line_length = min(map_data["\n"]) + 1
    return {pos + i for i in (1, -1, line_length, -line_length)}


def day_12_part_one() -> int:
    md = day_12_load_map_data()
    regions_by_letter = day_12_find_regions(md)
    total_cost = 0
    for letter in regions_by_letter:
        for reg in regions_by_letter[letter]:
            area = len(reg)
            perimeter = sum(
                4 - len(day_12_like_neighbours(pp, md))
                for pp in reg
            )
            total_cost += area * perimeter
    return total_cost


def day_12_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle24(12).get_text_input()
    md = day_12_load_map_data()
    every_square = [
        loc
        for k, v in md.items()
        for loc in v
        if k.isalpha()]
    fence_pairs = [
        (sq, ngb)
        for sq in every_square
        for ngb in day_12_like_neighbours(sq, md, inverse=True, text=text)
    ]
    print(f"{len(fence_pairs)=}")
    return sum(
        day_12_count_sides(r, fence_pairs) * len(r)
        for reg in day_12_find_regions(md).values()
        for r in reg
        # if reg == "T"   # TODO: change back to "\n"
    )


def day_12_count_sides(region_locs: {}, boundary_pairs: []) -> int:
    total_sides, line_length = 0, 141
    b_pairs = [*filter(lambda bp: bp[0] in region_locs, boundary_pairs)]
    for facing_diff, f_name in zip(
            [1, -1, line_length, -line_length],
            ["left", "right", "up", "down"]
    ):
        facing_this_way = [
            *filter(lambda bb: bb[0] == bb[1] + facing_diff, b_pairs)
        ]
        ftw_sides = 0
        # print(facing_this_way)
        ftw_pos = [pp[0] for pp in facing_this_way]
        while ftw_pos:
            n = min(ftw_pos)
            run = []
            while n in ftw_pos:
                run.append(n)
                n += abs(line_length if abs(facing_diff) == 1 else 1)
            ftw_pos = [*filter(lambda x: x not in run, ftw_pos)]
            ftw_sides += 1
        # print(f"There are {ftw_sides} {f_name}-facing sides")
        total_sides += ftw_sides
    return total_sides


def day_11_part_one(text: str = "") -> int:
    stones = day_11_load_stones(text)
    for _ in range(25):
        stones = day_11_blink(stones)
    return len(stones)


def day_11_part_two(text: str = "") -> int:
    stones = {
        s: 1
        for s in day_11_load_stones(text)
    }
    transformations = {}
    for b_count in range(75):
        next_stones = collections.defaultdict(int)
        for st, no_of_st in stones.items():
            if st in transformations:
                tr = transformations[st]
                for nn in {*tr}:
                    next_stones[nn] += (no_of_st * tr.count(nn))
            else:
                trans = day_11_blink([st])
                transformations[st] = trans
                for nn in {*trans}:
                    next_stones[nn] += (no_of_st * trans.count(nn))
        stones = next_stones
        if not (b_count + 1) % 5:
            print(f"Blinked {b_count + 1} times, now have {len(stones)} stones")
            print(f"\t{len(transformations)=}")
    return sum(stones.values())


def day_11_load_stones(text: str = "") -> [int]:
    if not text:
        text = Puzzle24(11).get_text_input()
    return [int(n) for n in text.split(" ")]


def day_11_blink(stones: [int]) -> [int]:
    transformed = []
    for stone in stones:
        if stone == 0:
            transformed.append(1)
        else:
            as_string = f"{stone}"
            if not len(as_string) % 2:
                midway = len(as_string) // 2
                transformed.extend(
                    [
                        int(f"{as_string[:midway]}"),
                        int(f"{as_string[midway:]}")
                    ])
            else:
                transformed.append(stone * 2024)
    return transformed


def day_10_part_one(text: str = "") -> int:
    m = day_10_load_map(text)
    trailheads, summits = (m[i] for i in (0, 9))
    return sum(day_10_can_reach(th, su, m)
               for th in trailheads
               for su in summits)


def day_10_can_reach(trailhead: int, summit: int, area: {}) -> bool:
    path_queue = [[trailhead]]
    while path_queue:
        path = path_queue.pop()
        height = [k for k, v in area.items() if path[-1] in v][0]
        for ngb in filter(
                lambda pt: pt in area[height + 1],
                day_10_neighbours(path[-1], area)
        ):
            if ngb == summit:
                return True
            elif height < 8:
                path_queue.append(path + [ngb])
    return False


def day_10_part_two(text: str = "") -> int:
    m = day_10_load_map(text)
    trailheads, summits = (m[i] for i in (0, 9))
    return sum(day_10_count_ways_to_reach(th, su, m)
               for th in trailheads
               for su in summits)


def day_10_count_ways_to_reach(trailhead: int, summit: int, area: {}) -> int:
    possible_routes = 0
    path_queue = [[trailhead]]
    while path_queue:
        path = path_queue.pop()
        height = [k for k, v in area.items() if path[-1] in v][0]
        for ngb in filter(
                lambda pt: pt in area[height + 1],
                day_10_neighbours(path[-1], area)
        ):
            if ngb == summit:
                possible_routes += 1
            elif height < 8:
                path_queue.append(path + [ngb])
    return possible_routes


def day_10_load_map(text: str = "") -> {}:
    if not text:
        text = Puzzle24(10).get_text_input()
    topo_map = {
        int(t) if t.isnumeric() else t:
        {m.start() for m in re.finditer(t, text)}
        for t in "".join(f"{n}" for n in range(10)) + "\n"
    }
    # for k in topo_map.keys():
    #     print(f"{k:>2}: {topo_map[k]}")
    return topo_map


def day_10_neighbours(location: int, topography: {}) -> {}:
    line_length = min(topography["\n"]) + 1
    return (location + (d * j) for d, j in
            itertools.product((-1, 1), (1, line_length)))


def day_9_part_one(text: str = "") -> int:
    return day_9_checksum(
        day_9_compact(day_9_load_raw_map(text))
    )


def day_9_load_raw_map(text: str = "") -> {}:
    if not text:
        text = Puzzle24(9).get_text_input().strip("\n")
    layout = {}
    get_file = True
    loc, file_id = 0, -1
    for ch in text:
        size = int(ch)
        if get_file:
            file_id += 1
            layout[file_id] = loc, size
        loc += size
        get_file = not get_file
    assert all(s <= 9 for ll, s in layout.values())
    print(f"{len(text)=}")
    assert len(layout) == (len(text) // 2) + 1
    return layout


def day_9_compact(file_pos: {}) -> {}:
    free_space = day_9_find_free_space(file_pos)

    def fs_to_left(pos: int) -> {}:
        return {p: l for p, l in free_space.items() if p < pos}

    compacted = {}
    for file_id in sorted(file_pos.keys(), reverse=True):
        position, length = file_pos[file_id]
        if sum(fs_to_left(position).values()):
            new_compacted_item = []
            while length:
                fsl = fs_to_left(position).items()
                next_free_space = min(fsl) if fsl else (0, 0)
                start, available = next_free_space
                if available:
                    use_fs = min(length, available)
                    length -= use_fs
                    del free_space[start]
                    new_compacted_item.append((start, use_fs))
                    if available - use_fs:
                        free_space[start + use_fs] = available - use_fs
                else:
                    new_compacted_item.append((position, length))
                    break
            compacted[file_id] = new_compacted_item
        else:
            compacted[file_id] = [file_pos[file_id]]
    # print(f"After: {free_space=}")
    # print(f"{compacted=}")
    for k in file_pos:
        assert sum(v[1] for v in compacted[k]) == file_pos[k][1]
        # print(f"{k} ok;", end="")
    return compacted


def day_9_find_free_space(file_pos: {}) -> {}:
    free_space = {}
    for fid in range(max(file_pos.keys())):
        s, l = file_pos[fid]
        fs_start = s + l
        fs_length = file_pos[fid + 1][0] - fs_start
        if fs_length:
            free_space[fs_start] = fs_length
    # print(f"{free_space=}")
    # does it matter that final element of free space = 0?
    assert all(s <= 9 for s in free_space.values())
    print(f"Zeroes in free_space values: {sum(fs_len == 0 for fs_len in free_space.values())}")
    return free_space


def day_9_checksum(compacted_map: {}) -> int:
    checksum = 0
    for k, v in compacted_map.items():
        for li in v:
            start, length = li
            for index in range(length):
                checksum += (start + index) * k
    return checksum


def day_9_part_two(text: str = "") -> int:
    return day_9_checksum(
        day_9_p2_compact(day_9_load_raw_map(text))
    )


def day_9_p2_compact(file_pos: {}) -> {}:
    new_layout = {}
    free_space = day_9_find_free_space(file_pos)

    def large_enough_fs_to_left(pos: int, required_size: int) -> {}:
        return {p: s for p, s in free_space.items()
                if p < pos and s >= required_size}

    for file_id in sorted(file_pos.keys(), reverse=True):
        position, length = file_pos[file_id]
        available_spots = large_enough_fs_to_left(position, length)
        if available_spots:
            leftmost_pos, free_length = min(available_spots.items())
            new_layout[file_id] = [(leftmost_pos, length)]
            del free_space[leftmost_pos]
            free_length -= length
            if free_length:
                free_space[leftmost_pos + length] = free_length
        else:
            new_layout[file_id] = [file_pos[file_id]]
    return new_layout


def day_8_map_from_input(text: str = "") -> {}:
    if not text:
        text = Puzzle24(8).get_text_input().strip("\n")
    return {
        lib.Point(y, x): ch
        for y, row in enumerate(text.split("\n"))
        for x, ch in enumerate(row)
    }


def day_8_part_one(text: str = "") -> int:
    antinodes = set()
    m = day_8_map_from_input(text)
    node_types = {*filter(lambda v: v != ".", m.values())}
    for nt in node_types:
        node_locs = [k for k, v in m.items()
                     if v == nt]
        for i, nl in enumerate(node_locs):
            for j, other_node in enumerate(node_locs[i + 1:]):
                v_diff, h_diff = (other_node[n] - nl[n] for n in range(2))
                outer_1 = lib.Point(
                    *(pt + d for d, pt in zip((v_diff, h_diff), max(nl, other_node))
                ))
                outer_2 = lib.Point(
                    *(pt - d for d, pt in zip((v_diff, h_diff), min(nl, other_node))
                ))
                # print(f"{nl} -> {outer_1=} {outer_2=}")
                if all(dd % 3 == 0 for dd in (v_diff, h_diff)):
                    start = min(nl, other_node)
                    inner1 = lib.Point(
                        start[0] + (v_diff // 3),
                        start[1] + (h_diff // 3)
                    )
                    inner2 = lib.Point(
                        start[0] + (2 * (v_diff // 3)),
                        start[1] + (2 * (h_diff // 3))
                    )
                    # print(f"{nl} -> {inner1=} {inner2=}")
                    antinodes.update((inner1, inner2))
                antinodes.update(
                    (
                        on for on in (outer_1, outer_2)
                        if on in m
                    ))
    return len(antinodes)


def day_8_part_two(text: str = "") -> int:
    m = day_8_map_from_input(text)
    y_max, _ = max(m)
    antinodes = set()
    node_types = {*filter(lambda v: v not in ".#", m.values())}
    for nt in node_types:
        node_locs = [k for k, v in m.items()
                     if v == nt]
        for i, nl in enumerate(node_locs):
            for j, other_node in enumerate(node_locs[i + 1:]):
                antinodes.update(
                    day_8_two_points_to_line(nl, other_node, y_max)
                )
    return len(antinodes)


def day_8_two_points_to_line(
        pt1: lib.Point, pt2: lib.Point, limit: int) -> [lib.Point]:
    line = [lib.Point(*pt1)]
    v_to_h_ratio = tuple((pt2[c] - pt1[c] for c in range(2)))
    if all(dd != 0 for dd in v_to_h_ratio):
        v_to_h_ratio = tuple(
            (f // math.gcd(*v_to_h_ratio) for f in v_to_h_ratio)
        )
    for ratio in (v_to_h_ratio, (-rn for rn in v_to_h_ratio)):
        v_step, h_step = ratio
        new_pt = lib.Point(*pt1)
        while True:
            new_pt = lib.Point(new_pt[0] + v_step, new_pt[1] + h_step)
            if all(0 <= new_pt[c] <= limit for c in range(2)):
                line.append(new_pt)
            else:
                break
    return line


def day_7_load_line(line: str) -> (int, [int]):
    target, _, components = line.partition(": ")
    return int(target), [int(c) for c in components.split(" ")]


def day_7_load(text: str = "") -> {}:
    if not text:
        text = Puzzle24(7).get_text_input().strip("\n")
    return {
        k: v
        for k, v in [day_7_load_line(ln) for ln in text.split("\n")]
    }


def day_7_computes(target: int, components: [int],
                   with_concatenation: bool = False) -> bool:
    possible_ops = "*+"
    if with_concatenation:
        possible_ops += "|"
    for operators in itertools.product(
            possible_ops, repeat=len(components) - 1
    ):
        cumulative = components[0]
        for i, c in enumerate(components[1:]):
            op = operators[i]
            if op == "*":
                cumulative *= c
            elif op == "+":
                cumulative += c
            else:
                cumulative = int(f"{cumulative}{c}")
        if cumulative == target:
            return True
    return False


def day_7_part_one(text: str = "") -> int:
    return sum(k for k, v in day_7_load(text).items()
               if day_7_computes(k, v))


def day_7_part_two(text: str = "") -> int:
    input_data = day_7_load(text)
    cumulative = 0
    for k, v in input_data.items():
        if day_7_computes(k, v):
            cumulative += k
        else:
            if day_7_computes(k, v, with_concatenation=True):
                cumulative += k
    return cumulative


def day_6_load_map(text: str = "") -> {}:
    if not text:
        text = Puzzle24(6).get_text_input().strip("\n")
    return {
        lib.Point(y, x): char
        for y, row in enumerate(text.split("\n"))
        for x, char in enumerate(row)
    }


def day_6_part_one(text: str = "") -> int:
    room = day_6_load_map(text)
    return len(
        {
            vl[0] for vl in
            day_6_visited_locations_with_facing(
                room, [k for k, v in room.items() if v == "^"][0])
            if vl[0] in room
        }
    )


def day_6_visited_locations_with_facing(
        room: {},
        guard_pos: lib.Point,
        facing: str = "U") -> [(lib.Point, str)]:
    visited = {(guard_pos, facing)}
    room[guard_pos] = "."
    while guard_pos in room:
        loc_in_front = pm23[facing](guard_pos)
        if loc_in_front in room:
            if room[loc_in_front] == ".":
                guard_pos = loc_in_front
                if (guard_pos, facing) in visited:
                    break
                visited.add((guard_pos, facing))
            else:
                facing = day_6_turn_to_right(facing)
        else:
            guard_pos = loc_in_front
            visited.add((guard_pos, facing))
    return visited


def day_6_distance_to_next_blocker(point: lib.Point,
                                   facing: str, area: {}) -> int:
    co_ord = 0 if facing in "UD" else 1
    fixed = not co_ord
    comparison = operator.gt if facing in "DR" else operator.lt
    blockers = [pt[co_ord] for pt, char in area.items()
                if pt[fixed] == point[fixed]
                and comparison(pt[co_ord], point[co_ord])
                and char == "#"]
    if blockers:
        distance_func = (lambda p: min(blockers) - p) \
            if facing in "DR" else (lambda p: p - max(blockers))
        return distance_func(point[co_ord])
    return 1_000_000


def day_6_back_up(point: lib.Point, facing: str) -> lib.Point:
    """Go back one step in the opposite direction"""
    directions = "URDL"
    return pm23[directions[(directions.index(facing) + 2) % 4]](point)


def day_6_turn_to_right(facing: str) -> str:
    """Change the facing to rotate 90 degrees to the right"""
    directions = "URDL"
    return directions[(directions.index(facing) + 1) % 4]


def day_6_move_n_steps_in_direction(
        point: lib.Point, facing: str, steps: int) -> lib.Point:
    f_dict = {
        "U": (-1, 0),
        "R": (0, 1),
        "D": (1, 0),
        "L": (0, -1),
    }
    delta = (steps * dd for dd in f_dict[facing])
    return lib.Point(*(c + d for c, d in zip(point, delta)))


def day_6_part_two_simple(text: str = "") -> int:
    m = day_6_load_map(text)
    new_obstacles = 0
    starting_point = [k for k, v in m.items() if v == "^"][0]
    all_visited_data = day_6_visited_locations_with_facing(m, starting_point)
    candidate_points = {pt for pt in all_visited_data
                        if pt[0] != starting_point}
    m[starting_point] = "."
    loop_counter = 0
    for candidate, f in candidate_points:
        loop_counter += 1
        m[candidate] = "#"
        pt = day_6_back_up(candidate, f)
        f = day_6_turn_to_right(f)
        will_visit = day_6_visited_locations_with_facing(m, pt, f)
        if all(wv[0] in m for wv in will_visit):
            new_obstacles += 1
        m[candidate] = "."
        if loop_counter % 100 == 0:
            print(f"{loop_counter:,}: {new_obstacles=}")
    return new_obstacles



def day_6_part_two(text: str = "") -> int:
    """Idea: it's only worth placing an obstacle in the
            area the guard can already reach (so get the
            set of visited points)"""
    # (8, 3) addition proves it's not just a rectangle
    m = day_6_load_map(text)
    new_obstacles = 0
    # a point may have been visited multiple times in
    # part one.  So only deal with its first appearance
    all_visited_data = day_6_visited_locations_with_facing(text)
    starting_point = all_visited_data[0][0]
    candidate_points = {pt for pt in all_visited_data
                        if pt[0] != starting_point}
    m[starting_point] = "."
    pf_will_leave = set()
    known_routes_to_exit = []
    len_cp, point_counter = len(candidate_points), 0
    for candidate, f in candidate_points:
        point_counter += 1
        # modded_map = {**m}
        m[candidate] = "#"
        # _, f = [vp for vp in all_visited_data if vp[0] == candidate][0]
        pt = day_6_back_up(candidate, f)
        f = day_6_turn_to_right(f)
        will_visit = []
        no_escape = False
        while pt in m:
            pt = day_6_move_n_steps_in_direction(
                pt, f,
                day_6_distance_to_next_blocker(pt, f, m) - 1
            )
            if (pt, f) in will_visit:
                new_obstacles += 1
                break
            elif pt not in m:
                if all(
                        lib.manhattan_distance(wv[0], candidate) > 1
                        for wv in will_visit
                ):
                    if will_visit and will_visit[0] not in pf_will_leave:
                        known_routes_to_exit.append(will_visit)
                    pf_will_leave.update(will_visit)
            will_visit.append((pt, f))
            if (pt, f) in pf_will_leave and len(will_visit) > 10:
                last_three = will_visit[-3:]
                for kre in known_routes_to_exit:
                    if ((pt, f) in kre) and any(
                        kre[i:i + 3] == last_three
                        for i, loc in enumerate(kre[:-3])
                    ):
                        spot = kre.index(last_three[0])
                        # print(f"Found repeated route!! {last_three=} found={kre[spot:spot + 3]}")
                        if day_6_escape_route_blocked(kre[spot:], candidate):
                            # print(" . . . but the route is blocked :(")
                            no_escape = True
                        else:
                            pt = (-1, -1), ""
                            break
            f = day_6_turn_to_right(f)
        m[candidate] = "."
        if point_counter % 100 == 0:
            print(f"{point_counter:,} of {len_cp} points. "
                  f"{len(will_visit):,} visited before "
                  f"{'leaving grid' if pt not in m else 'looping'} "
                  f"{len(known_routes_to_exit)=}")
    assert 0 < new_obstacles < day_6_part_one(text)
    return new_obstacles


def day_6_escape_route_blocked(route: [(lib.Point, str)],
                               blocker: lib.Point) -> bool:
    for i, loc in enumerate(route[:-1]):
        pt1, pt2 = (x[0] for x in (loc, route[i + 1]))
        if (pt1[0] <= blocker[0] <= pt2[0] or
                pt2[0] <= blocker[0] <= pt1[0]) and pt1[1] == blocker[1]:
            return True
        if (pt1[1] <= blocker[1] <= pt2[1] or
                pt2[1] <= blocker[1] <= pt1[1]) and pt1[0] == blocker[0]:
            return True
    return False


def day_5_load(text: str = "") -> ([(str,)],):
    if not text:
        text = Puzzle24(5).get_text_input().strip("\n")
    rules = [
        tuple(m.group().split("|"))
        for m in re.finditer(r"\d+\|\d+", text)
    ]
    updates = [
        um.group().split(",")
        for um in re.finditer(r"(\d+,)+\d+", text)
    ]
    return rules, updates


def day_5_is_valid(update: [str,], rules: [(str,)]) -> bool:
    for i, page_no in enumerate(update):
        following_pages = update[i + 1:]
        if any(
                (page_no == right) and (left in following_pages)
                for left, right in rules
        ):
            return False
    return True


def day_5_sum_of_middle_pages(booklets: [str,]) -> int:
    return sum(int(bklt[len(bklt) // 2]) for bklt in booklets)


def day_5_part_one(text: str = "") -> int:
    rules, updates = day_5_load(text)
    valid = [*filter(lambda u: day_5_is_valid(u, rules), updates)]
    return day_5_sum_of_middle_pages(valid)


def day_5_re_order_to_valid(pages: [str], rules: [(str,)]) -> [str]:
    print(f"Re-ordering {pages}")
    if len(pages) == 1:
        return pages
    left = 0
    for nn in pages:
        the_rest = [v for v in pages if v != nn]
        if not any((vr, nn) in rules for vr in the_rest):
            print(f"{nn} can safely be the leftmost page")
            left = nn
            break
    re_ordered = [left] + day_5_re_order_to_valid(
        [v for v in pages if v != left], rules
    )
    assert day_5_is_valid(re_ordered, rules)
    return re_ordered


def day_5_part_two(text: str = "") -> int:
    rules, updates = day_5_load(text)
    invalid = [*filter(lambda u: not day_5_is_valid(u, rules), updates)]
    return day_5_sum_of_middle_pages(
        day_5_re_order_to_valid(iv, rules) for iv in invalid
    )


def day_4_load_grid(text: str) -> {}:
    return {
        lib.Point(y, x): letter
        for y, row in enumerate(text.split("\n"))
        for x, letter in enumerate(row)
    }


def day_4_matches(x_loc: lib.Point, grid: {}) -> int:
    """count the number of wordsearch matches in the grid that start
        from the given location"""
    directions = ["U", "UL", "L", "DL", "D", "DR", "R", "UR"]
    target = "XMAS"
    matches = 0
    for d in directions:
        found_word = "X"
        next_co_ord = x_loc
        for step in range(3):
            for move in d:
                next_co_ord = pm23[move](next_co_ord)
            if next_co_ord not in grid:
                break
            found_word += grid[next_co_ord]
            if found_word != target[:step + 2]:
                break
        matches += (found_word == target)
    return matches


def day_4_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle24(4).get_text_input().strip("\n")
    grid = day_4_load_grid(text)
    x_s = [*filter(lambda k: grid[k] == "X", grid.keys())]
    return sum(day_4_matches(x, grid) for x in x_s)


def day_4_x_mas_match(a_loc: lib.Point, grid: {}) -> bool:
    directions = [("UL", "DR"), ("UR", "DL")]
    for diagonal in directions:
        diag_letters = ""
        for d in diagonal:
            point = a_loc
            for move in d:
                point = pm23[move](point)
            if point not in grid:
                return False
            diag_letters += grid[point]
        if set(diag_letters) != set("MS"):
            return False
    return True


def day_4_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle24(4).get_text_input().strip("\n")
    grid = day_4_load_grid(text)
    a_s = [*filter(lambda k: grid[k] == "A", grid.keys())]
    return sum(day_4_x_mas_match(a, grid) for a in a_s)


def day_3_sum_of_muls(content: str) -> int:
    mul_pairs = [
        (int(mm.group()) for mm in
         re.finditer(r"\d+", m.group()))
        for m in re.finditer(r"mul\(\d+,\d+\)", content)
    ]
    return sum(i * j for i, j in mul_pairs)


def day_3_part_one(text: str = "") -> int:
    if not text:
        text = Puzzle24(3).get_text_input()
    return day_3_sum_of_muls(text)


def day_3_part_two(text: str = "") -> int:
    if not text:
        text = Puzzle24(3).get_text_input()
    do_s = [0] + [m.end() for m in re.finditer(r"do\(\)", text)]
    dont_s = [m.start() for m in re.finditer(r"don't\(\)", text)] + [-1]
    on_slices = []
    for d in do_s:
        e = min(filter(lambda x: x > d, dont_s)) if d < max(dont_s) else -1
        if not on_slices or d > on_slices[-1].stop:
            on_slices.append(slice(d, e))
    return sum(
        day_3_sum_of_muls(text[os]) for os in on_slices
    )


def day_2_load_reports(text: str) -> [[int]]:
    return [
        [int(n) for n in report.split(" ")]
        for report in text.strip("\n").split("\n")
    ]


def day_2_is_safe(report: [int]) -> bool:
    if (sorted(report) == report or
            sorted(report, reverse=True) == report):
        for i, n in enumerate(report[1:]):
            if not 1 <= abs(n - report[i]) <= 3:
                return False
        return True
    return False


def day_2_can_be_made_safe(report: [int]) -> bool:
    if day_2_is_safe(report):
        return True
    for mask in range(len(report)):
        if day_2_is_safe(report[:mask] + report[mask + 1:]):
            return True
    return False


def day_2_count_safe_reports(text: str, part_two: bool = False) -> int:
    if not text:
        text = Puzzle24(2).get_text_input()
    checker_func = day_2_can_be_made_safe if part_two else day_2_is_safe
    return sum(
        checker_func(rept)
        for rept in day_2_load_reports(text)
    )


def day_2_part_one(text: str = "") -> int:
    return day_2_count_safe_reports(text)


def day_2_part_two(text: str = "") -> int:
    return day_2_count_safe_reports(text, True)


def day_1_extract_lists(text: str = "") -> ([int],):
    if not text:
        text = Puzzle24(1).get_text_input()
    terms = r"\d+ ", r"\d+\n"
    return (
        [int(m.group()[:-1]) for m in re.finditer(t, text)]
        for t in terms
    )


def day_1_part_one(text: str = "") -> int:
    s1, s2 = (sorted(nn) for nn in day_1_extract_lists(text))
    return sum(abs(v1 - v2) for v1, v2 in zip(s1, s2))


def day_1_part_two(text: str = "") -> int:
    nums1, nums2 = day_1_extract_lists(text)
    left_nums = set(nums1)
    return sum(
        n * nums1.count(n) * nums2.count(n)
        for n in left_nums
    )
