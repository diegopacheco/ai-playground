import unittest
from unittest.mock import patch

from agent import Agent, process_runner
from graph import Graph, Node, build_graph


class AgentTest(unittest.TestCase):
    @patch("agent.subprocess.run")
    def test_success_excludes_provider_diagnostics(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = "answer\n"
        run.return_value.stderr = "provider diagnostics\n"
        self.assertEqual("answer", process_runner(["provider"]))

    def test_provider_commands_are_argument_arrays(self):
        commands = []
        runner = lambda command: commands.append(command) or "ok"
        providers = ["claude", "codex", "agy", "ollama"]
        for provider in providers:
            self.assertEqual("ok", Agent(provider, runner).call("model-a", "hello", ["--flag"]))
        self.assertEqual([
            ["claude", "-p", "--model", "model-a", "--flag", "hello"],
            ["codex", "exec", "--model", "model-a", "--flag", "hello"],
            ["agy", "-p", "--model", "model-a", "--flag", "hello"],
            ["ollama", "run", "--flag", "model-a", "hello"],
        ], commands)

    def test_invalid_provider_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            Agent("invalid", lambda command: "ok").call("model", "hello")


class GraphTest(unittest.TestCase):
    def test_fan_out_results_feed_the_final_node(self):
        prompts = []

        def llm(prompt):
            prompts.append(prompt)
            return f"result-{len(prompts)}"

        results = build_graph(llm).run("Should we ship?")
        self.assertEqual(["facts", "risks", "answer"], list(results))
        self.assertIn("facts:\nresult-1", prompts[2])
        self.assertIn("risks:\nresult-2", prompts[2])
        self.assertEqual("result-3", results["answer"])

    def test_cycle_prevents_partial_execution(self):
        calls = []
        graph = Graph([Node("a", "A"), Node("b", "B")], [("a", "b"), ("b", "a")], calls.append)
        with self.assertRaisesRegex(ValueError, "cycle"):
            graph.run("question")
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
