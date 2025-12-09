import library as lib
import re


def day_9_part_two(t="") -> int:
    reds = day_9_load_red_tiles(t)
    pairings = sorted(
        [
            (tile, other_tile)
            for ti, tile in enumerate(reds)
            for other_tile in reds[ti + 1:]
        ],
        key=lambda pair: day_9_area(*pair),
        reverse=True
    )

    def all_edges_are_within(edges: [int]) -> bool:
        return (any(r.vt <= edges[0] and r.hz <= edges[2] for r in reds) and
                any(r.vt <= edges[0] and r.hz >= edges[3] for r in reds) and
                any(r.vt >= edges[1] and r.hz <= edges[2] for r in reds) and
                any(r.vt >= edges[1] and r.hz >= edges[3] for r in reds))

    for pr in pairings:
        box_edges = [f(p[i] for p in pr) for i in range(2) for f in (min, max)]
        if not any(box_edges[0] < pt.vt < box_edges[1] and
                   box_edges[2] < pt.hz < box_edges[3]
                   for pt in reds) and all_edges_are_within(box_edges):
            print(f"{pr=}")
            return day_9_area(*pr)
    return 0


def day_9_part_one(t="") -> int:
    red_tiles = day_9_load_red_tiles()
    print(f"{len(red_tiles)=}")
    return max(
        day_9_area(tile, other_tile)
        for ti, tile in enumerate(red_tiles)
        for other_tile in red_tiles[ti + 1:]
    )


def day_9_area(tile: (lib.Point,), other_tile: (lib.Point,)) -> int:
    return ((abs(other_tile[0] - tile[0]) + 1) *
            (abs(other_tile[1] - tile[1]) + 1))


def day_9_load_red_tiles(t="") -> [lib.Point]:
    text = lib.load(t)
    return [
        lib.Point(*map(int, (v, h)))
        for row in text.split("\n")
        for h, v in [row.split(",")]
    ]


def day_8_part_one(t="") -> int:
    max_conn = 10 if t else 1000
    largest_1, largest_2, largest_3 = sorted(
        (len(cir) for cir in day_8_traverse(
            t, max_connections=max_conn
        )), reverse=True
    )[:3]
    return largest_1 * largest_2 * largest_3


def day_8_part_two(t="") -> int:
    b1, b2 = day_8_traverse(t)
    print(f"{b1=}, {b2=}")
    return b1[0] * b2[0]


def day_8_traverse(t: str = "", max_connections: int = 0) -> [{}]:
    t = lib.load(t)
    boxes = [tuple(map(int, tuple(nums.split(",")))) for nums in t.split("\n")]
    print(f"{len(boxes)=}")
    box_pairs = sorted([
        (b, bb)
        for i, b in enumerate(boxes)
        for bb in boxes[i + 1:]
        if b != bb
    ], key=lambda pair: day_8_euclidean_distance(*pair), reverse=True)
    circuits = []
    connections = 0
    ultimate, penultimate = None, None
    unfinished_condition = (lambda nc: nc < max_connections) \
        if max_connections else (lambda nc: len(circuits) == 0 or len(circuits[0]) < len(boxes))
    while unfinished_condition(connections):
        bp = box_pairs.pop()
        touched_circuits = [
            *filter(lambda c: any(bx in c for bx in bp), circuits)
        ]
        if len(touched_circuits) == 2:
            circuits = [
                *filter(lambda cr: cr not in touched_circuits, circuits)
            ] + [touched_circuits[0].union(touched_circuits[1])]
        elif len(touched_circuits) == 1:
            # if all(bbx in existing_circuits[0] for bbx in bp):
            #     continue
            circuits = [
                *filter(lambda cr: cr not in touched_circuits, circuits)
            ] + [touched_circuits[0].union({*bp})]
        else:
            circuits.append({*bp})
        connections += 1
        ultimate, penultimate = bp#[1], ultimate
    print(f"{len(circuits[0])=}, {connections=}")
    return circuits if max_connections else (ultimate, penultimate)


def day_8_euclidean_distance(pt1: (int,), pt2: (int,)):
    return sum((x1 - x2) ** 2 for x1, x2 in zip(pt1, pt2)) ** 0.5


def day_7_part_one(t="") -> int:
    grid = lib.load_grid(t)
    # print(f"Input grid has {max(pt.vt for pt in grid)} rows", end="")
    # print(f" and {len([*filter(lambda v: v == '^', grid.values())])} splitters")
    beam_cols = {[*filter(lambda k: grid[k] == 'S', grid.keys())][0].hz}
    splits = 0
    for row_id in range(max(pt.vt for pt in grid) + 1):
        row = {
            pt: v for pt, v in
            filter(lambda i: i[0].vt == row_id, grid.items())
        }
        split_cols = [
            loc.hz for loc in row
            if loc.hz in beam_cols and grid[loc] == "^"
        ]
        if split_cols:
            splits += len(split_cols)
            uninterrupted_beams = {
                *filter(lambda bc: bc not in split_cols, beam_cols)
            }
            beam_cols = {
                split_col + deflection
                for split_col in split_cols
                for deflection in (-1, 1)
            }
            beam_cols.update(uninterrupted_beams)
    return splits


def day_7_part_two(t: str = ""):
    grid = lib.load_grid(t)
    ways_to_get_there = {rr: grid[lib.Point(0, rr)] == "S"
                         for rr in range(max(grid)[1] + 1)}
    for row_id in range(max(pt.vt for pt in grid) + 1):
        row = {
            pt: v for pt, v in
            filter(lambda i: i[0].vt == row_id, grid.items())
        }
        split_cols = [
            loc.hz for loc in row
            if ways_to_get_there[loc.hz] and grid[loc] == "^"
        ]
        for sc in split_cols:
            w = ways_to_get_there[sc]
            ways_to_get_there[sc - 1] += w
            ways_to_get_there[sc + 1] += w
            ways_to_get_there[sc] = 0
    return sum(ways_to_get_there.values())


def day_6_part_one(t="") -> int:
    text = lib.load(t)
    rows = text.split("\n")
    number_rows = [[int(mm.group()) for mm in re.finditer(r"\d+", nr)] for nr in rows[:-1]]
    op_row = [m.group() for m in re.finditer(r"[+*]", rows[-1])]
    total = 0
    for col_no in range(len(op_row)):
        op = op_row[col_no]
        if op == "+":
            total += sum(number_rows[n][col_no] for n in range(len(number_rows)))
        else:
            n1 = number_rows[0][col_no]
            for i in range(1, len(number_rows)):
                n1 *= number_rows[i][col_no]
            total += n1
    return total


def day_6_part_two(t="") -> int:
    text = lib.load(t)
    total = 0
    rows = [r[::-1] for r in text.split("\n")]
    numbers = []
    for row_index in range(len(rows[0])):
        column = "".join(rr[row_index] for rr in rows)
        num = re.search(r"\d+", column)
        if num:
            numbers.append(int(num.group()))
        if re.search(r"[+*]", column[-1]):
            if column[-1] == "+":
                total += sum(numbers)
            else:
                n1 = numbers[0]
                for i in range(1, len(numbers)):
                    n1 *= numbers[i]
                total += n1
            numbers = []
    return total


def day_5_part_one(t="") -> int:
    ranges, ii = day_5_ranges_and_ingredients()
    return sum(
        any(int(ingredient) in fr for fr in ranges)
        for ingredient in ii.split("\n")
    )


def day_5_part_two(t="") -> int:
    ranges, _ = day_5_ranges_and_ingredients(t)
    merged_ranges = []

    def touches(range_1: range, range_2: range) -> bool:
        if range_1.start <= range_2.start and range_2.stop <= range_1.stop:
            return True
        if range_2.start <= range_1.start and range_1.stop <= range_2.stop:
            return True
        return (range_1.start <= range_2.start <= range_1.stop or
                range_1.start <= range_2.stop <= range_1.stop)

    for i, rr in enumerate(ranges):
        """
        1. See how rr interacts with existing merged_ranges.
            This could be one of  . . . ways:
                a. not at all - just add it to merged_ranges
                b. one end is in one or more merged_ranges - merge
                c. entirely contained within a merged_range
                d. entirely contains one or more merged_ranges
        2. new range is the min of the starts and the max of the ends
            of any of the touched ranges
        3. new merged_ranges with any touched ranges filtered out,
            plus the new range
        """
        touched_existing = [*filter(lambda mr: touches(rr, mr), merged_ranges)]
        merged_ranges = [
                            range(
                                min(r.start for r in touched_existing + [rr]),
                                max(r.stop for r in touched_existing + [rr])
                            )
                        ] + [
            *filter(
                lambda existing: existing not in touched_existing,
                merged_ranges
            )
        ]
    return sum(r.stop - r.start for r in merged_ranges)


def day_5_ranges_and_ingredients(t="") -> ([range], str):
    rr, _, ii = lib.load(t).partition("\n\n")
    ranges = [
        range(a, b + 1)
        for a_b in rr.split("\n")
        for a, b in [map(int, a_b.split("-"))]
    ]
    return ranges, ii


def day_4_part_one(t="") -> int:
    return len(day_4_removables(lib.load_grid(t, ".")))


def day_4_part_two(t="") -> int:
    grid = lib.load_grid(t, ".")
    removed = 0
    while can_remove := day_4_removables(grid):
        removed += len(can_remove)
        grid = {k: v for k, v in grid.items() if k not in can_remove}
    return removed


def day_4_removables(grid: {}) -> [lib.Point]:
    def is_removable(roll_location: lib.Point) -> bool:
        return len(
            [*filter(
                lambda nbr: nbr in grid,
                lib.neighbours(roll_location, diagonals=True)
            )]
        ) < 4

    return [*filter(lambda r: is_removable(r), grid)]


def day_3_part_one(t="") -> int:
    text = lib.load(t)
    total_joltage = 0
    for bank in text.split("\n"):
        for n in range(99, 9, -1):
            n1, n2 = f"{n}"
            if re.search(f"{n1}.*{n2}", bank):
                total_joltage += n
                break
    return total_joltage


def day_3_part_two(t="") -> int:
    text = lib.load(t)
    solution = 0
    for bank in text.split("\n"):
        best = day_3_best_by_grabbing_max(bank)#day_3_best_joltage(bank)
        print(f"{bank} -> {best}")
        solution += int(best)
    return solution


def day_3_best_by_grabbing_max(b: str) -> str:
    grabbed = ""
    while len(grabbed) < 12:
        best = max(b[:len(grabbed) - 11]) if len(grabbed) < 11 else max(b)
        pos = re.search(best, b).start()
        b = b[pos + 1:]
        grabbed += best
    return grabbed


def day_3_best_joltage(b: str) -> str:
    if len(b) == 12:
        return b
    best_known = day_3_best_joltage(b[1:])
    candidate = b[0] + re.sub(f"{min(best_known)}", "", best_known, count=1)
    assert len(candidate) == 12
    # print(f"{len(candidate)}\t{candidate=}")
    return max(candidate, best_known)


def day_2_part_two() -> int:
    permissible_repeaters = {i: [1] for i in range(2, 11)}
    permissible_repeaters[4] = [1, 2]
    permissible_repeaters[6] = [1, 2, 3]
    permissible_repeaters[8] = [1, 2, 4]
    permissible_repeaters[9] = [1, 3]
    permissible_repeaters[10] = [1, 2, 5]

    invalid = set()
    for rr in day_2_ranges():
        # print(rr, rr[1] - rr[0])
        for n in range(rr[0], rr[1] + 1):
            n_digits = len(f"{n}")
            if n_digits > 1:
                permitted = permissible_repeaters[n_digits]
                for n_repeat_len in permitted:
                    if f"{n}"[:n_repeat_len] * (n_digits // n_repeat_len) == f"{n}":
                        invalid.add(n)
        # print(f"\tTotal invalid: {len(invalid)}")
    return sum(invalid)


def day_2_part_one() -> int:
    sum_of_invalid = 0
    for rr in day_2_ranges():
        # print(rr, rr[1] - rr[0])
        for n in range(rr[0], rr[1] + 1):
            str_n = f"{n}"
            len_n = len(str_n)
            if len_n % 2 == 0:
                if f"{str_n[:len_n // 2]}" == f"{str_n[len_n // 2:]}":
                    sum_of_invalid += n
    return sum_of_invalid


def day_2_ranges() -> ():
    text = lib.load()
    return (
        tuple(int(x) for x in r.split("-"))
        for r in text.split(",")
    )


def day_1_part_two(text: str = "") -> int:
    text = lib.load(text)
    pos = 50
    zeroes = 0
    for move in text.split("\n"):
        print(f"{move=}, ", end="")
        n_steps = int(move[1:])
        direction = -1 if move[0] == "L" else 1
        zero_passes = n_steps // 100
        print(f"straight zp: {zero_passes}", end=", ")
        new_pos = (pos + (n_steps * direction)) % 100
        if ((direction == -1 and (new_pos > pos) and pos != 0) or
                (direction == 1 and (new_pos < pos)) or new_pos == 0):
            zero_passes += 1
            print(f"directional zero pass", end="")
        print(f"total zp: {zero_passes}")
        zeroes += zero_passes
        pos = new_pos
    return zeroes


def day_1_part_one(text: str = "") -> int:
    text = lib.load(text)
    pos = 50
    zeroes = 0
    for move in text.split("\n"):
        n_steps = int(move[1:])
        direction = -1 if move[0] == "L" else 1
        pos = (pos + (n_steps * direction)) % 100
        zeroes += pos == 0
    return zeroes
