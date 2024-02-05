import argparse
import math
from math import comb
import pickle

num_players = 5
players = [1 << i for i in range(num_players)]
set_first_to = 7
match_result_key = (0, 0, 0, 0, 0, 0, 0, 0)
search_method_key = (0, 0, 0, 0, 0, 0, 0, 1)
win_prob_key = (0, 0, 0, 0, 0, 0, 0, 2)
team_member_key = (0, 0, 0, 0, 0, 0, 0, 3)


def print_labeled_array(array, row_labels, col_labels):
    col_label_str = "     " + "  ".join(f'{col_labels[i]:<10}' for i in range(num_players))
    print(col_label_str)

    for idx, row in enumerate(array):
        row_str = f'{row_labels[idx]:<10}' + "  ".join(f'{elem:<10.2f}' for elem in row)
        print(row_str)


def win_probability(rating1, rd1, rating2, rd2):
    q = math.log(10) / 400
    g_rd2 = 1 / math.sqrt(1 + (3 * q ** 2 * rd2 ** 2) / (math.pi ** 2))

    expected_score = 1 / (1 + 10 ** (g_rd2 * (rating1 - rating2) / -400))

    return expected_score

def find_round_probability(x):
    def expression(a):
        return sum((1 - a) ** n * a ** 7 * comb(6 + n, n) for n in range(7))

    # Binary search for a value in [0.0, 1.0] to get the expression result close to x
    lo, hi = 0.0, 1.0
    while lo < hi:
        mid = (lo + hi) / 2
        if expression(mid) < x:
            lo = mid
        else:
            hi = mid

        if hi - lo < 1e-10:  # Convergence tolerance
            break

    return (lo + hi) / 2


# Position description: (us_bitmask,them_bitmask,us_score,them_score,us_rounds,them_rounds,pick_state, who_picked)
# End positions: 0,0,us_score,them_score
# Position outcomes: Win or loss
# Position value: Win probability
# Pick: 0 us (first pick) 1 us (second pick) 2 them (first pick) 3 them (second pick) 4 both
# Search method: 0 alpha beta both sides, 1 alpha beta us them random


def get_outcome(table, pos, search_method, win_prob_arr):
    # print(pos)
    if pos in table:
        return table[pos][0]
    # Case where position is won
    if pos[0] == 0 and pos[1] == 0:
        win_prob = 1.0 if (pos[2] > pos[3] or (pos[2] == pos[3] and pos[4] > pos[5])) else 0.0
        table[pos] = [win_prob, 0]
        return win_prob
    # Otherwise, game tree search
    if pos[6] == 0:
        # Pick the best outcome
        row = 0
        max_win = -0.1
        for us_player in range(num_players):
            if pos[0] & players[us_player] == 0:
                continue
            val = get_outcome(table,
                              (pos[0] & ~players[us_player], pos[1], pos[2], pos[3], pos[4], pos[5], 3, us_player),
                              search_method, win_prob_arr)
            if val > max_win:
                max_win = val
                row = us_player
        table[pos] = [max_win, row]
        return max_win
    elif pos[6] == 1:
        # Pick the best outcome
        row = 0
        max_win = -0.1
        for us_player in range(num_players):
            if pos[0] & players[us_player] == 0:
                continue
            val = simulate_game(table, win_prob_arr, us_player, pos[7], search_method, pos)
            if val > max_win:
                max_win = val
                row = us_player
        table[pos] = [max_win, row]
        return max_win
    elif pos[6] == 2:
        # Pick the best outcome (worst for us)
        row = 0
        min_win = 1.1
        for them_player in range(num_players):
            if pos[1] & players[them_player] == 0:
                continue
            val = get_outcome(table,
                              (pos[0], pos[1] & ~players[them_player], pos[2], pos[3], pos[4], pos[5], 1, them_player),
                              search_method, win_prob_arr)
            if val < min_win:
                min_win = val
                row = them_player
        table[pos] = [min_win, row]
        return min_win
    elif pos[6] == 3:
        # Pick the best outcome (worst for us)
        row = 0
        min_win = 1.1
        for them_player in range(num_players):
            if pos[1] & players[them_player] == 0:
                continue
            val = simulate_game(table, win_prob_arr, pos[7], them_player, search_method, pos)
            if val < min_win:
                min_win = val
                row = them_player
        table[pos] = [min_win, row]
        return min_win
    elif pos[6] == 4:
        # Calculate all possible outcomes. Then figure out some strategy for picking the best one
        # TODO: add support for both sides picking when not all players available
        outcomes = []
        for us_player in range(num_players):
            outcome = []
            for them_player in range(num_players):
                outcome.append(simulate_game(table, win_prob_arr, us_player, them_player, search_method, pos))
            outcomes.append(outcome)

        # Picks are at the same time, which makes this a bit more complicated
        # If opponent is picking randomly, then we should pick assuming that
        # If opponent is picking optimally, we want to pick the player with the highest min win probability
        # More strategies are possible like picking assuming the opponent picks like above
        # There is probably a more optimal way to do this that gives me PageRank vibes
        if search_method == 0:
            row = 0
            max_win = 0
            for i in range(num_players):
                min_win = min(outcomes[i])
                if min_win > max_win:
                    max_win = min_win
                    row = i
            table[pos] = [max_win, row]
            return max_win
        elif search_method == 1:
            row = 0
            max_win = 0
            for i in range(num_players):
                win = sum(outcomes[i]) / 5
                if win > max_win:
                    max_win = win
                    row = i
            table[pos] = [max_win, row]
            return max_win
    print(f"Should not happen {pos}")


def simulate_game(table, win_prob, player_us, player_them, search_method, pos):
    player_us_win_prob = win_prob[player_us][player_them]
    player_them_win_prob = 1 - player_us_win_prob

    us_players_removed = pos[0] & ~players[player_us]
    them_players_removed = pos[1] & ~players[player_them]
    # We win
    # Probability of outcome: # of ways outcome can happen * probability of outcome
    # Last game must be a win
    prob_sum = 0
    win_prob_sum = 0
    outcomes = {}
    us_win_score = pos[2] + 8
    us_rounds = pos[4] + 1
    for them_wins in range(0, set_first_to):
        prob = (player_them_win_prob) ** them_wins * player_us_win_prob ** set_first_to * comb(
            set_first_to + them_wins - 1, them_wins)
        prob_sum += prob
        win_prob_val = get_outcome(table, (
            us_players_removed, them_players_removed, us_win_score, pos[3] + them_wins, us_rounds, pos[5], 2,
            num_players), search_method, win_prob)
        win_prob_sum += prob * win_prob_val
        outcomes[(set_first_to, them_wins)] = (prob, win_prob_val)

    # We lose
    them_win_score = pos[3] + 8
    them_rounds = pos[5] + 1
    for us_wins in range(0, set_first_to):
        prob = (player_them_win_prob) ** set_first_to * player_us_win_prob ** us_wins * comb(set_first_to + us_wins - 1,
                                                                                             us_wins)
        prob_sum += prob
        win_prob_val = get_outcome(table, (
            us_players_removed, them_players_removed, pos[2] + us_wins, them_win_score, pos[4], them_rounds, 0,
            num_players), search_method, win_prob)
        win_prob_sum += prob * win_prob_val
        outcomes[(us_wins, set_first_to)] = (prob, win_prob_val)

    if abs(prob_sum - 1) > 0.01:
        print(f"Probability sum is not 1: {prob_sum}")

    match_match_key = (us_players_removed, them_players_removed, pos[2], pos[3], pos[4], pos[5])

    if match_match_key not in table[match_result_key]:
        table[match_result_key][match_match_key] = {}
    if player_us not in table[match_result_key][match_match_key]:
        table[match_result_key][match_match_key][player_us] = {}

    table[match_result_key][match_match_key][player_us][player_them] = (outcomes, win_prob_sum)

    return win_prob_sum


def get_persistent(prompt, validator):
    while True:
        val = input(prompt)
        if validator(val):
            return val
        else:
            print("Invalid input, try again")


def get_pick(player_map, prompt, mask):
    # TODO: dissallow picking players that are not in the game
    for i in range(num_players):
        if players[i] & mask == 0:
            continue
        print(f"{i} {player_map[i]}")
    return int(
        get_persistent(prompt, lambda x: x.isnumeric() and 0 <= int(x) < num_players and players[int(x)] & mask != 0))


def get_round_result(current_pos, player_us, player_them, us_map, them_map):
    # Get the result of the round (score)
    while True:
        # Get our score
        us_score = int(get_persistent(f"Enter how many points {us_map[player_us]} scored: ", lambda x: x.isnumeric()))
        # Get their score
        them_score = int(
            get_persistent(f"Enter how many points {them_map[player_them]} scored: ", lambda x: x.isnumeric()))
        if us_score == them_score:
            print("Tie game, try again")
        elif us_score != set_first_to and them_score != set_first_to:
            print("No one reached the set score, try again")
        else:
            break
    total_us_score = current_pos[2] + us_score + (1 if us_score == set_first_to else 0)
    total_them_score = current_pos[3] + them_score + (1 if them_score == set_first_to else 0)
    total_us_rounds = current_pos[4] + 1 if us_score == set_first_to else current_pos[4]
    total_them_rounds = current_pos[5] + 1 if them_score == set_first_to else current_pos[5]
    pick_state = 2 if us_score == set_first_to else 0
    return (
    current_pos[0], current_pos[1], total_us_score, total_them_score, total_us_rounds, total_them_rounds, pick_state,
    num_players)


def main():
    parser = argparse.ArgumentParser(description='Process filenames')
    parser.add_argument('--win-prob', type=str, help='Filename for win probability', required=True)
    parser.add_argument('--saved-table', type=str, help='Filename for saved table', required=True)
    parser.add_argument('--search-method', type=int, help='Method being used to search', required=True)
    parser.add_argument('--generate-new', action='store_true', help='Flag to generate new data')
    parser.add_argument('--search-root', action='store_true', help='Flag for whether to search from root or not')
    parser.add_argument("--is-glicko", action="store_true", help="If this is set, input is actually glicko ratings")
    parser.add_argument("--set-prob", action="store_true",
                        help="Flag to set if testing probability of winning a round or a set")

    args = parser.parse_args()

    with open(args.win_prob, "r") as wp:
        lines = wp.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
        our_team = lines[-2].split(",")
        other_team = lines[-1].split(",")
        if args.is_glicko:
            # First row is our team glicko ratings
            # Second row is our team rd
            # Third row is their team glicko ratings
            # Fourth row is their team rd
            # Calculate win probability matrix
            our_team_glicko = [float(i) for i in lines[0].split(",")]
            our_team_rd = [float(i) for i in lines[1].split(",")]
            their_team_glicko = [float(i) for i in lines[2].split(",")]
            their_team_rd = [float(i) for i in lines[3].split(",")]
            win_prob_arr = [
                [win_probability(our_team_glicko[i], our_team_rd[i], their_team_glicko[j], their_team_rd[j]) for j in
                 range(len(their_team_glicko))] for i in range(len(our_team_glicko))]
        else:
            win_prob_arr = [[float(j) for j in i.split(",")] for i in lines[:-2]]

    if args.generate_new:
        table = {}
        table[match_result_key] = {}
        table[search_method_key] = args.search_method
        table[win_prob_key] = args.win_prob
        table[team_member_key] = (our_team, other_team)
    else:
        with open(args.saved_table, "rb") as of:
            table = pickle.load(of)
            if table is None:
                print("Error reading table")
                return
            if table[search_method_key] != args.search_method:
                print("Search method does not match")
                return
            if table[win_prob_key] != args.win_prob:
                print("Win probability does not match")
                return
            if table[team_member_key] != (our_team, other_team):
                print("Team members do not match; you probably generated this table with different team members")
                return

    if args.set_prob:
        print("Set win probability")
        print_labeled_array(win_prob_arr, our_team, other_team)
        for i in range(len(win_prob_arr)):
            for j in range(len(win_prob_arr[i])):
                win_prob_arr[i][j] = find_round_probability(win_prob_arr[i][j])


    us_player_map = {0: our_team[0], 1: our_team[1], 2: our_team[2], 3: our_team[3], 4: our_team[4]}

    them_player_map = {0: other_team[0], 1: other_team[1], 2: other_team[2], 3: other_team[3], 4: other_team[4]}

    # Print win probability table
    print("Round win probability")
    print_labeled_array(win_prob_arr, us_player_map, them_player_map)

    starting_pos = (2 ** num_players - 1, 2 ** num_players - 1, 0, 0, 0, 0, 4, num_players)

    # Search from root
    if args.search_root:
        result = get_outcome(table, starting_pos, args.search_method, win_prob_arr)
        # Print win prob
        print("Win probability: ", result[0])
        print("Best pick: ", us_player_map[result[1]])

        # Save table
        should_save = get_persistent("Save table? (y/n): ", lambda x: x in ["y", "n"])
        if should_save == "y":
            with open(args.saved_table, "wb") as of:
                pickle.dump(table, of)

    # Command line UI to query table and stuff
    current_pos = starting_pos
    while True:
        # Print current position
        print("Current position: ", current_pos)
        # Print players names in the game
        print("Us: ", [us_player_map[i] for i in range(num_players) if current_pos[0] & players[i] != 0])
        print("Them: ", [them_player_map[i] for i in range(num_players) if current_pos[1] & players[i] != 0])
        # Print scores and rounds
        print("Us score: ", current_pos[2])
        print("Them score: ", current_pos[3])
        print("Us rounds: ", current_pos[4])
        print("Them rounds: ", current_pos[5])
        # Print pick state, converted to human readable text
        pick_state = current_pos[6]
        if pick_state == 0:
            print("We are picking first")
        elif pick_state == 1:
            print("We are picking second")
            they_picked = current_pos[7]
            print("They picked: ", them_player_map[they_picked])
        elif pick_state == 2:
            print("They are picking first")
        elif pick_state == 3:
            print("They are picking second")
            us_picked = current_pos[7]
            print("We picked: ", us_player_map[us_picked])
        elif pick_state == 4:
            print("Both are picking")
        # Print what we can do; we can move to a completely new position, move to a nearby position, or query the table in some way
        print("Options: ")
        print("1. Move to a new position")
        print("2. Move to a nearby position (enter moves from current position)")
        print("3. Query a match result")
        print("4. Analyze the position")
        print("5. Reset to starting position")
        print("6. Export win probability array")
        print("7. Quit")
        choice = get_persistent("Enter choice: ", lambda x: x in ["1", "2", "3", "4", "5", "6", "7"])
        if choice == "1":
            # Move to a new position
            print("Enter new position")
            # First get which players are in the game
            print("Enter which players are in the game (aka: have not yet played a round or been picked)")
            us_players = 0
            them_players = 0
            for i in range(num_players):
                val = get_persistent(f"Is {us_player_map[i]} in? (y/n): ", lambda x: x in ["y", "n"])
                if val == "y":
                    us_players |= players[i]
            for i in range(num_players):
                val = get_persistent(f"Is {them_player_map[i]} in? (y/n): ", lambda x: x in ["y", "n"])
                if val == "y":
                    them_players |= players[i]
            # Then get the scores and rounds
            us_score = int(get_persistent("Enter our score: ", lambda x: x.isnumeric()))
            them_score = int(get_persistent("Enter their score: ", lambda x: x.isnumeric()))
            us_rounds = int(get_persistent("Enter our rounds: ", lambda x: x.isnumeric()))
            them_rounds = int(get_persistent("Enter their rounds: ", lambda x: x.isnumeric()))
            # Then get the pick state
            pick_state = int(get_persistent(
                "Who is picking? (0 us (first pick) 1 us (second pick) 2 them (first pick) 3 them (second pick) 4 both): ",
                lambda x: x.isnumeric() and 0 <= int(x) <= 4))
            # If relevant, who got picked
            who_got = num_players
            if pick_state == 1 or pick_state == 3:
                print("Who got picked?")
                who_got = get_pick(them_player_map if pick_state == 1 else us_player_map,
                                   "Enter the number of the person who got picked first: ",
                                   current_pos[1] if pick_state == 1 else current_pos[0])
            # Then move to the new position
            current_pos = (us_players, them_players, us_score, them_score, us_rounds, them_rounds, pick_state, who_got)
        elif choice == "2":
            if current_pos[6] == 0:
                # Ask who got picked
                who_got = get_pick(us_player_map, "Enter the number of the person we picked: ", current_pos[0])
                # Then move to the new position
                current_pos = (
                    current_pos[0] & ~players[who_got], current_pos[1], current_pos[2], current_pos[3], current_pos[4],
                    current_pos[5], 3, who_got)
            elif current_pos[6] == 2:
                # Ask who got picked
                who_got = get_pick(them_player_map, "Enter the number of the person they picked: ", current_pos[1])
                # Then move to the new position
                current_pos = (
                    current_pos[0], current_pos[1] & ~players[who_got], current_pos[2], current_pos[3], current_pos[4],
                    current_pos[5], 1, who_got)
            elif current_pos[6] == 4:
                # Ask who got picked on our team
                who_picked_us = get_pick(us_player_map, "Enter the number of the person we picked: ", current_pos[0])
                # Ask who got picked on their team
                who_picked_them = get_pick(them_player_map, "Enter the number of the person they picked: ",
                                           current_pos[1])
                # Then move to the new position
                current_pos = (current_pos[0] & ~players[who_picked_us], current_pos[1] & ~players[who_picked_them],
                               current_pos[2], current_pos[3], current_pos[4], current_pos[5], 4, num_players)
                current_pos = get_round_result(current_pos, who_picked_us, who_picked_them, us_player_map,
                                               them_player_map)
            elif current_pos[6] == 1:
                who_got = get_pick(us_player_map, "Enter the number of the person we picked: ", current_pos[0])
                current_pos = (
                    current_pos[0] & ~players[who_got], current_pos[1], current_pos[2], current_pos[3], current_pos[4],
                    current_pos[5], 2, current_pos[7])
                current_pos = get_round_result(current_pos, who_got, current_pos[7], us_player_map, them_player_map)

            elif current_pos[6] == 3:
                who_got = get_pick(them_player_map, "Enter the number of the person they picked: ", current_pos[1])
                current_pos = (
                    current_pos[0], current_pos[1] & ~players[who_got], current_pos[2], current_pos[3], current_pos[4],
                    current_pos[5], 0, current_pos[7])
                current_pos = get_round_result(current_pos, current_pos[7], who_got, us_player_map, them_player_map)
        elif choice == "3":
            us_player = get_pick(us_player_map, "Enter the number of the person on our team: ", current_pos[0])
            them_player = get_pick(them_player_map, "Enter the number of the person on their team: ", current_pos[1])
            match_match_key = (
                current_pos[0] & ~players[us_player], current_pos[1] & ~players[them_player], current_pos[2],
                current_pos[3], current_pos[4], current_pos[5])
            if match_match_key not in table[match_result_key]:
                print("Match not found")
            elif us_player not in table[match_result_key][match_match_key]:
                print("Match not found")
            elif them_player not in table[match_result_key][match_match_key][us_player]:
                print("Match not found")
            else:
                print("Match found")
                match = table[match_result_key][match_match_key][us_player][them_player]
                print("Match result: ", match)
                for i in range(0, set_first_to):
                    for j in range(0, set_first_to):
                        if match[(j, 7)] > match[(7, i)]:
                            print(f"Throwing after they get to {i} points might be advantageous due to counterpicks")
                            print(f"Make sure to get to at least {j} points before throwing")
                            break
        elif choice == "4":
            if current_pos not in table:
                print("Analyzing position")
                get_outcome(table, current_pos, args.search_method, win_prob_arr)
                should_save = get_persistent("Save table? (y/n): ", lambda x: x in ["y", "n"])
                if should_save == "y":
                    with open(args.saved_table, "wb") as of:
                        pickle.dump(table, of)
            result = table[current_pos]
            win_prob = result[0]
            player = result[1]
            pick_state = current_pos[6]
            if pick_state == 0:
                print("We should pick: ", us_player_map[player])
            elif pick_state == 1:
                print("We should pick: ", us_player_map[player])
            elif pick_state == 2:
                print("They should pick: ", them_player_map[player])
            elif pick_state == 3:
                print("They should pick: ", them_player_map[player])
            elif pick_state == 4:
                print("We should pick: ", us_player_map[player])
            print("Win probability: ", win_prob)
        elif choice == "5":
            current_pos = starting_pos
        elif choice == "6":
            with open("win_prob_export.csv", "w") as wp:
                for row in win_prob_arr:
                    wp.write(",".join(str(i) for i in row) + "\n")
                # Write team names
                wp.write(",".join(our_team) + "\n")
                wp.write(",".join(other_team) + "\n")
        elif choice == "7":
            break
        print("\n\n")


if __name__ == "__main__":
    main()
