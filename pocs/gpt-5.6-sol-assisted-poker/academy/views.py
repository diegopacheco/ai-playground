import random

from django.http import HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .agents import PROVIDERS, agent_action
from .poker import advice, advance_hand, cards_view, deck, equity, equity_report, fresh_hand, hand_name, score, stage_copy, street_name, winning_routes


HAND_RANKINGS = [
    {"name": "Royal flush", "cards": ["A♠", "K♠", "Q♠", "J♠", "10♠"], "odds": "1 in 649,740", "note": "The ten through ace, all in one suit. It cannot be beaten."},
    {"name": "Straight flush", "cards": ["9♥", "8♥", "7♥", "6♥", "5♥"], "odds": "1 in 72,193", "note": "Five consecutive cards in one suit."},
    {"name": "Four of a kind", "cards": ["Q♠", "Q♥", "Q♦", "Q♣", "4♠"], "odds": "1 in 4,165", "note": "Four cards sharing the same rank."},
    {"name": "Full house", "cards": ["J♠", "J♥", "J♦", "7♣", "7♠"], "odds": "1 in 694", "note": "Three of one rank plus a pair of another."},
    {"name": "Flush", "cards": ["A♣", "J♣", "8♣", "5♣", "2♣"], "odds": "1 in 509", "note": "Five cards in one suit, not consecutive."},
    {"name": "Straight", "cards": ["10♠", "9♥", "8♦", "7♣", "6♠"], "odds": "1 in 255", "note": "Five consecutive ranks in mixed suits."},
    {"name": "Three of a kind", "cards": ["8♠", "8♥", "8♦", "K♣", "3♠"], "odds": "1 in 47", "note": "Three cards sharing one rank."},
    {"name": "Two pair", "cards": ["A♠", "A♥", "5♦", "5♣", "9♠"], "odds": "1 in 21", "note": "Two different pairs with one kicker."},
    {"name": "One pair", "cards": ["K♠", "K♥", "10♦", "6♣", "2♠"], "odds": "1 in 2.4", "note": "Two cards sharing one rank."},
    {"name": "High card", "cards": ["A♠", "J♥", "8♦", "5♣", "2♠"], "odds": "1 in 2", "note": "No made combination; the highest card leads."},
]
for ranking in HAND_RANKINGS:
    ranking["card_views"] = [
        {"rank": card[:-1], "suit": card[-1], "red": card[-1] in "♥♦"}
        for card in ranking["cards"]
    ]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = [
    {"name": "Spades", "symbol": "♠", "red": False},
    {"name": "Hearts", "symbol": "♥", "red": True},
    {"name": "Diamonds", "symbol": "♦", "red": True},
    {"name": "Clubs", "symbol": "♣", "red": False},
]


def fresh_guided_hand():
    state = fresh_hand()
    state["journey"] = []
    state["complete"] = False
    return state


def guidance_snapshot(state):
    chance = equity(state["hero"], state["board"])
    action, reason = advice(state["hero"], state["board"], chance)
    return {
        "street": street_name(state["street"]),
        "chance": chance,
        "action": action,
        "reason": reason,
        "hand": hand_name(state["hero"] + state["board"]),
    }


def simulation_context(request):
    state = request.session.get("simulation")
    if not state:
        state = fresh_guided_hand()
        request.session["simulation"] = state
    state.setdefault("journey", [])
    state.setdefault("complete", False)
    chance = equity(state["hero"], state["board"])
    action, reason = advice(state["hero"], state["board"], chance)
    previous_chance = state["journey"][-1]["chance"] if state["journey"] else None
    chance_delta = chance - previous_chance if previous_chance is not None else None
    return {
        "sim": state,
        "sim_hero": cards_view(state["hero"]),
        "sim_board": cards_view(state["board"]),
        "sim_chance": chance,
        "sim_action": action,
        "sim_reason": reason,
        "sim_street": street_name(state["street"]),
        "sim_stage_copy": stage_copy(state["street"]),
        "sim_hand_name": hand_name(state["hero"] + state["board"]),
        "sim_journey": state["journey"],
        "sim_step": 4 if state["complete"] else state["street"] + 1,
        "sim_delta": chance_delta,
    }


def ai_context(request):
    provider = request.session.get("provider", "claude")
    game = request.session.get("ai_game")
    context = {"provider": provider, "providers": PROVIDERS, "ai_game": game}
    if game:
        context.update({
            "ai_hero": cards_view(game["hero"]),
            "ai_board": cards_view(game["board"]),
            "ai_street": street_name(game["street"]),
            "ai_provider_label": PROVIDERS[provider]["label"],
        })
        if game.get("finished"):
            context["ai_villain"] = cards_view(game["villain"])
        elif game.get("show_equity"):
            report = equity_report(game["hero"], game["board"], 1200)
            context["ai_equity"] = {
                **report,
                "hand": hand_name(game["hero"] + game["board"]),
                "cards_to_come": 5 - len(game["board"]),
                "unseen": 52 - len(game["hero"]) - len(game["board"]),
                "routes": winning_routes(game["hero"], game["board"]),
            }
    return context


def home(request):
    tab = request.GET.get("tab", "cards")
    if tab not in {"cards", "simulation", "ai"}:
        tab = "cards"
    context = {
        "tab": tab,
        "rankings": HAND_RANKINGS,
        "ranks": RANKS,
        "suits": SUITS,
        **simulation_context(request),
        **ai_context(request),
    }
    return render(request, "academy/home.html", context)


@require_POST
def simulation_new(request):
    request.session["simulation"] = fresh_guided_hand()
    return redirect(f"{reverse('home')}?tab=simulation")


@require_POST
def simulation_advance(request):
    state = request.session.get("simulation") or fresh_guided_hand()
    state.setdefault("journey", [])
    state.setdefault("complete", False)
    if state["complete"]:
        return redirect(f"{reverse('home')}?tab=simulation")
    state["journey"].append(guidance_snapshot(state))
    if state["street"] < 3:
        advance_hand(state)
    else:
        state["complete"] = True
    request.session["simulation"] = state
    request.session.modified = True
    return redirect(f"{reverse('home')}?tab=simulation")


@require_POST
def provider_select(request):
    provider = request.POST.get("provider")
    if provider not in PROVIDERS:
        return HttpResponseBadRequest("Unknown provider")
    request.session["provider"] = provider
    return redirect(f"{reverse('home')}?tab=ai")


def fresh_ai_game():
    cards = fresh_hand()
    reserved = set(cards["hero"] + cards["draw"])
    available = [card for card in deck() if card not in reserved]
    villain = random.SystemRandom().sample(available, 2)
    return {
        **cards,
        "villain": villain,
        "pot": 30,
        "hero_stack": 990,
        "villain_stack": 980,
        "history": ["Blinds posted. You are on the button."],
        "to_call": 10,
        "show_equity": False,
        "finished": False,
        "result": "",
    }


def complete_street(game):
    if game["street"] < 3:
        advance_hand(game)
        return
    hero_score = score(game["hero"] + game["board"])
    villain_score = score(game["villain"] + game["board"])
    game["finished"] = True
    if hero_score > villain_score:
        game["result"] = f"You win with {hand_name(game['hero'] + game['board']).lower()}."
    elif villain_score > hero_score:
        game["result"] = f"The opponent wins with {hand_name(game['villain'] + game['board']).lower()}."
    else:
        game["result"] = "The pot is split. Both hands are equal."


@require_POST
def ai_new(request):
    request.session["ai_game"] = fresh_ai_game()
    return redirect(f"{reverse('home')}?tab=ai")


@require_POST
def ai_reveal(request):
    game = request.session.get("ai_game")
    if not game or game.get("finished"):
        return HttpResponseBadRequest("No active hand")
    game["show_equity"] = True
    request.session["ai_game"] = game
    request.session.modified = True
    return redirect(f"{reverse('home')}?tab=ai")


@require_POST
def ai_move(request):
    game = request.session.get("ai_game")
    user_action = request.POST.get("action")
    if not game or game.get("finished") or user_action not in {"fold", "check", "call", "raise"}:
        return HttpResponseBadRequest("Invalid move")
    to_call = game.get("to_call", 0)
    if to_call and user_action == "check":
        return HttpResponseBadRequest("Invalid move")
    if not to_call and user_action == "call":
        return HttpResponseBadRequest("Invalid move")
    if user_action == "fold":
        game["finished"] = True
        game["result"] = "You folded. The opponent wins the pot."
        game["history"].append("You folded.")
    elif to_call and user_action == "call":
        contribution = min(to_call, game["hero_stack"])
        game["hero_stack"] -= contribution
        game["pot"] += contribution
        game["to_call"] = 0
        game["history"].append(f"You called {contribution}.")
        complete_street(game)
    else:
        contribution = to_call + 80 if user_action == "raise" else 0
        contribution = min(contribution, game["hero_stack"])
        game["hero_stack"] -= contribution
        game["pot"] += contribution
        past_tense = {"check": "checked", "call": "called", "raise": "raised"}[user_action]
        game["history"].append(f"You {past_tense}.")
        opponent_chance = equity(game["villain"], game["board"], 500)
        legal = ["fold", "call", "raise"] if user_action == "raise" else ["check", "raise"]
        if to_call:
            legal = ["fold", "call"]
        board_text = ", ".join(game["board"]) or "no community cards"
        hole_text = ", ".join(game["villain"])
        situation = f"your hole cards {hole_text}; street {street_name(game['street'])}; board {board_text}; pot {game['pot']}"
        ai_choice, status = agent_action(request.session.get("provider", "claude"), situation, opponent_chance, legal)
        game["history"].append(status)
        if ai_choice == "fold":
            game["finished"] = True
            game["result"] = "The opponent folded. You win the pot."
        else:
            ai_contribution = 80 if ai_choice == "raise" else 20 if ai_choice == "call" else 0
            if ai_choice == "call" and user_action == "raise":
                ai_contribution = 80
            ai_contribution = min(ai_contribution, game["villain_stack"])
            game["villain_stack"] -= ai_contribution
            game["pot"] += ai_contribution
            if ai_choice == "raise":
                game["to_call"] = 80
            else:
                game["to_call"] = 0
                complete_street(game)
    request.session["ai_game"] = game
    game["show_equity"] = False
    request.session.modified = True
    return redirect(f"{reverse('home')}?tab=ai")
