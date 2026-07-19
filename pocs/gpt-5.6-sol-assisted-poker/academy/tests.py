from unittest.mock import patch

from django.test import Client, SimpleTestCase, TestCase

from .poker import equity, hand_name, score


class PokerEngineTests(SimpleTestCase):
    def test_royal_flush_is_highest_hand(self):
        royal = ["As", "Ks", "Qs", "Js", "Ts", "2d", "3c"]
        quads = ["Ah", "Ad", "Ac", "As", "Kd", "2c", "3h"]
        self.assertEqual("Royal flush", hand_name(royal))
        self.assertGreater(score(royal), score(quads))

    def test_wheel_straight_is_recognized(self):
        cards = ["As", "2d", "3c", "4h", "5s", "Kd", "Qd"]
        self.assertEqual("Straight", hand_name(cards))

    def test_equity_is_repeatable_and_bounded(self):
        first = equity(["As", "Ah"], [], 120)
        second = equity(["As", "Ah"], [], 120)
        self.assertEqual(first, second)
        self.assertGreater(first, 70)
        self.assertLessEqual(first, 100)


class PageTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_all_tabs_render(self):
        for tab in ["cards", "simulation", "ai"]:
            response = self.client.get("/", {"tab": tab})
            self.assertEqual(200, response.status_code)
            self.assertContains(response, "Riverlight")

    def test_provider_choice_is_remembered(self):
        response = self.client.post("/ai/provider", {"provider": "codex"})
        self.assertEqual(302, response.status_code)
        self.assertEqual("codex", self.client.session["provider"])

    def test_invalid_provider_is_rejected(self):
        response = self.client.post("/ai/provider", {"provider": "unknown"})
        self.assertEqual(400, response.status_code)

    def test_simulation_advances_to_flop(self):
        self.client.get("/", {"tab": "simulation"})
        response = self.client.post("/simulation/advance")
        self.assertEqual(302, response.status_code)
        state = self.client.session["simulation"]
        self.assertEqual(1, state["street"])
        self.assertEqual(3, len(state["board"]))
        self.assertEqual(1, len(state["journey"]))

    def test_guidance_covers_all_four_streets(self):
        self.client.get("/", {"tab": "simulation"})
        for _ in range(4):
            response = self.client.post("/simulation/advance")
            self.assertEqual(302, response.status_code)
        state = self.client.session["simulation"]
        self.assertTrue(state["complete"])
        self.assertEqual(4, len(state["journey"]))
        page = self.client.get("/", {"tab": "simulation"})
        self.assertContains(page, "Four streets complete")

    def test_ai_equity_can_be_revealed(self):
        self.client.post("/ai/new")
        response = self.client.post("/ai/reveal")
        self.assertEqual(302, response.status_code)
        self.assertTrue(self.client.session["ai_game"]["show_equity"])
        page = self.client.get("/", {"tab": "ai"})
        self.assertContains(page, "Your chance to win")

    @patch("academy.views.agent_action", return_value=("call", "Claude chose call."))
    def test_ai_hand_accepts_move(self, action):
        self.client.post("/ai/new")
        response = self.client.post("/ai/move", {"action": "raise"})
        self.assertEqual(302, response.status_code)
        game = self.client.session["ai_game"]
        self.assertEqual(1, game["street"])
        self.assertIn("Claude chose call.", game["history"])
