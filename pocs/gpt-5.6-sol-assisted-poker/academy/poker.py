import itertools
import random
from collections import Counter


RANKS = "23456789TJQKA"
SUITS = "cdhs"
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
RANK_NAMES = {"T": "10", "J": "J", "Q": "Q", "K": "K", "A": "A"}
HAND_NAMES = [
    "High card",
    "One pair",
    "Two pair",
    "Three of a kind",
    "Straight",
    "Flush",
    "Full house",
    "Four of a kind",
    "Straight flush",
    "Royal flush",
]


def deck():
    return [rank + suit for rank in RANKS for suit in SUITS]


def card_view(card):
    rank, suit = card
    return {
        "code": card,
        "rank": RANK_NAMES.get(rank, rank),
        "suit": SUIT_SYMBOLS[suit],
        "red": suit in "dh",
        "name": f"{RANK_NAMES.get(rank, rank)} of {suit_name(suit)}",
    }


def cards_view(cards):
    return [card_view(card) for card in cards]


def suit_name(suit):
    return {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}[suit]


def score_five(cards):
    values = sorted([RANKS.index(card[0]) + 2 for card in cards], reverse=True)
    counts = Counter(values)
    groups = sorted(((count, value) for value, count in counts.items()), reverse=True)
    flush = len({card[1] for card in cards}) == 1
    unique = sorted(set(values), reverse=True)
    if unique == [14, 5, 4, 3, 2]:
        straight_high = 5
    else:
        straight_high = unique[0] if len(unique) == 5 and unique[0] - unique[4] == 4 else 0
    if flush and straight_high == 14:
        return 9, (14,)
    if flush and straight_high:
        return 8, (straight_high,)
    if groups[0][0] == 4:
        return 7, (groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return 6, (groups[0][1], groups[1][1])
    if flush:
        return 5, tuple(values)
    if straight_high:
        return 4, (straight_high,)
    if groups[0][0] == 3:
        kickers = sorted((value for value in values if value != groups[0][1]), reverse=True)
        return 3, (groups[0][1], *kickers)
    pairs = sorted((value for count, value in groups if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = next(value for value in values if value not in pairs)
        return 2, (pairs[0], pairs[1], kicker)
    if len(pairs) == 1:
        kickers = sorted((value for value in values if value != pairs[0]), reverse=True)
        return 1, (pairs[0], *kickers)
    return 0, tuple(values)


def score(cards):
    return max(score_five(list(group)) for group in itertools.combinations(cards, 5))


def hand_name(cards):
    if len(cards) < 5:
        if cards[0][0] == cards[1][0]:
            return "Pocket pair"
        return "Unmade hand"
    return HAND_NAMES[score(cards)[0]]


def equity(hero, board, samples=900):
    return equity_report(hero, board, samples)["equity"]


def equity_report(hero, board, samples=900):
    known = set(hero + board)
    remaining = [card for card in deck() if card not in known]
    seed = "".join(sorted(known)) + str(len(board))
    rng = random.Random(seed)
    wins = 0
    ties = 0
    needed = 5 - len(board)
    for _ in range(samples):
        draw = rng.sample(remaining, 2 + needed)
        opponent = draw[:2]
        complete_board = board + draw[2:]
        hero_score = score(hero + complete_board)
        opponent_score = score(opponent + complete_board)
        if hero_score > opponent_score:
            wins += 1
        elif hero_score == opponent_score:
            ties += 1
    win_rate = round(wins * 100 / samples)
    tie_rate = round(ties * 100 / samples)
    loss_rate = 100 - win_rate - tie_rate
    return {
        "equity": round((wins + ties * 0.5) * 100 / samples),
        "wins": win_rate,
        "ties": tie_rate,
        "losses": loss_rate,
        "samples": samples,
    }


def winning_routes(hero, board):
    cards = hero + board
    cards_to_come = 5 - len(board)
    routes = []
    if not board:
        label = preflop_label(hero)
        routes.append(f"Your {label} can win by pairing a private rank or by completing a stronger five-card pattern.")
    elif len(cards) >= 5:
        routes.append(f"You currently hold {hand_name(cards).lower()}. It can win by staying ahead through the remaining cards.")
    suit_counts = Counter(card[1] for card in cards)
    best_suit, best_suit_count = max(suit_counts.items(), key=lambda item: item[1])
    if cards_to_come and best_suit_count == 4:
        routes.append(f"One more {suit_name(best_suit)[:-1]} completes a flush route.")
    rank_values = {RANKS.index(card[0]) + 2 for card in cards}
    if 14 in rank_values:
        rank_values.add(1)
    closest_straight = min(5 - len(rank_values.intersection(set(range(start, start + 5)))) for start in range(1, 11))
    if cards_to_come and 0 < closest_straight <= cards_to_come:
        routes.append(f"A straight route is {closest_straight} helpful rank{'s' if closest_straight != 1 else ''} away.")
    rank_counts = Counter(card[0] for card in cards)
    if cards_to_come and max(rank_counts.values()) >= 2:
        routes.append("Matching a paired rank can improve the hand to trips, a full house, or four of a kind.")
    if not cards_to_come:
        routes.append("No cards remain. The current five-card value must already beat the hidden opponent hand.")
    return routes[:3]


def fresh_hand():
    cards = deck()
    random.SystemRandom().shuffle(cards)
    return {"hero": cards[:2], "board": [], "draw": cards[2:7], "street": 0}


def advance_hand(state):
    street = state["street"]
    if street == 0:
        state["board"] = state["draw"][:3]
        state["street"] = 1
    elif street < 3:
        state["board"].append(state["draw"][street + 2])
        state["street"] += 1
    return state


def street_name(street):
    return ["Pre-flop", "Flop", "Turn", "River"][street]


def preflop_label(hero):
    first, second = hero
    values = sorted([RANKS.index(first[0]) + 2, RANKS.index(second[0]) + 2], reverse=True)
    pair = values[0] == values[1]
    suited = first[1] == second[1]
    gap = values[0] - values[1]
    if pair and values[0] >= 10:
        return "premium pair"
    if pair:
        return "pocket pair"
    if values[0] >= 13 and values[1] >= 10:
        return "strong broadway cards"
    if suited and gap <= 2:
        return "suited connected cards"
    if values[0] >= 11 and values[1] >= 9:
        return "playable high cards"
    return "speculative holding"


def advice(hero, board, chance):
    if not board:
        label = preflop_label(hero)
        if chance >= 61:
            action = "Raise"
            reason = f"This is a {label}. Build the pot while your starting range is ahead."
        elif chance >= 49:
            action = "Call"
            reason = f"This is a {label}. Continue at a modest price and reassess on the flop."
        else:
            action = "Fold to pressure"
            reason = f"This is a {label}. Keep the pot small unless checking is free."
        return action, reason
    name = hand_name(hero + board)
    if chance >= 72:
        return "Bet for value", f"Your {name.lower()} is ahead of many likely holdings. Charge draws and weaker made hands."
    if chance >= 52:
        return "Check or call", f"Your {name.lower()} has reasonable equity, but a large pot can expose it to stronger ranges."
    if chance >= 35:
        return "Continue only cheaply", f"Your {name.lower()} still has paths to improve. Compare the price with the cards that help you."
    return "Fold to a meaningful bet", f"Your {name.lower()} has limited equity. Preserve chips unless the next card is free."


def stage_copy(street):
    return [
        "Two private cards define your starting range. Position, pair strength, suitedness, and connectedness matter most now.",
        "Three shared cards create most draws and made hands. Compare your best five-card path with the board texture.",
        "The fourth shared card changes draw odds sharply. With one card to come, weak draws need a better price.",
        "All shared cards are visible. Your decision is now about value, bluff-catching, or letting the hand go.",
    ][street]
