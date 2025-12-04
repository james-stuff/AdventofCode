import library as lib
import re


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
