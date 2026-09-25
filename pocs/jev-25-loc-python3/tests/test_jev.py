import os
import sys
import unittest

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jev import build_prompt, classify, load_model, to_probabilities

CHOICES = ["Legitimate", "Spam", "Phishing"]


class ProbabilityMathTest(unittest.TestCase):
    def test_probabilities_form_a_distribution_so_they_can_be_read_as_confidence(self):
        _, probabilities = to_probabilities(numpy.asarray([26.254, 27.262, 29.614]))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=9)
        self.assertTrue(numpy.all(probabilities > 0))

    def test_article_logits_give_article_probabilities(self):
        logprobs, probabilities = to_probabilities(numpy.asarray([26.254, 27.262, 29.614]))
        self.assertEqual(numpy.round(logprobs, 3).tolist(), [-3.482, -2.474, -0.122])
        self.assertEqual(numpy.round(probabilities, 3).tolist(), [0.031, 0.084, 0.885])

    def test_only_logit_differences_matter_because_other_tokens_are_ignored(self):
        _, base = to_probabilities(numpy.asarray([1.0, 2.0, 3.0]))
        _, shifted = to_probabilities(numpy.asarray([101.0, 102.0, 103.0]))
        numpy.testing.assert_allclose(base, shifted)

    def test_higher_logit_always_wins(self):
        _, probabilities = to_probabilities(numpy.asarray([0.5, 3.0, 1.0]))
        self.assertEqual(int(numpy.argmax(probabilities)), 1)


class PromptTest(unittest.TestCase):
    def test_each_choice_gets_a_single_letter_label_the_model_can_answer_with(self):
        prompt = build_prompt("hello", CHOICES)
        self.assertIn("A. Legitimate\nB. Spam\nC. Phishing", prompt)

    def test_prompt_ends_after_an_empty_think_block_so_next_token_is_the_answer(self):
        prompt = build_prompt("hello", CHOICES)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertIn("Email: hello", prompt)


class ModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def decide(self, email):
        scores = classify(self.model, email, CHOICES)
        return scores, CHOICES[int(numpy.argmax(scores["Probabilities"]))]

    def test_reproduces_the_article_numbers(self):
        scores, decision = self.decide("Payroll asks for your password on a non-company sign-in page.")
        self.assertEqual(decision, "Phishing")
        self.assertEqual(numpy.round(scores["Probabilities"].astype(float), 3).tolist(), [0.031, 0.084, 0.885])

    def test_prize_email_is_spam(self):
        _, decision = self.decide("Congratulations! You won a free cruise, click here to claim your prize now!!!")
        self.assertEqual(decision, "Spam")

    def test_team_meeting_email_is_legitimate(self):
        _, decision = self.decide("Hi team, the sprint retro moved to Thursday 3pm in room 4B. Agenda attached.")
        self.assertEqual(decision, "Legitimate")

    def test_same_input_gives_same_scores_because_state_is_reset(self):
        email = "Payroll asks for your password on a non-company sign-in page."
        first, _ = self.decide(email)
        self.decide("Hi team, lunch is at noon.")
        second, _ = self.decide(email)
        numpy.testing.assert_allclose(first["Logits"], second["Logits"], rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
