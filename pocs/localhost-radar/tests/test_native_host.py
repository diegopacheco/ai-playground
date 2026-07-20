import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "native" / "native_host.py"
SPEC = importlib.util.spec_from_file_location("native_host", MODULE_PATH)
HOST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOST)


class NativeHostTests(unittest.TestCase):
    def test_parses_listening_ports(self):
        output = "p123\ncnode\nn*:3000\nn127.0.0.1:3000\np456\ncjava\nn[::1]:8080\n"
        self.assertEqual(
            HOST.parse_lsof(output),
            [
                {"pid": 123, "name": "node", "port": 3000},
                {"pid": 456, "name": "java", "port": 8080}
            ]
        )

    def test_detects_languages(self):
        self.assertEqual(HOST.language_for("node", "node server.js", "/work"), "JavaScript / TypeScript")
        self.assertEqual(HOST.language_for("api", "/work/target/debug/api", "/work"), "Rust")
        self.assertEqual(HOST.language_for("server", "/work/server", "/work"), "Native / Other")

    def test_formats_podman_ports(self):
        value = [{"host_port": 8080, "container_port": 80, "protocol": "tcp"}]
        self.assertEqual(HOST.normalize_ports(value), "8080->80/tcp")

    def test_rejects_invalid_container_actions(self):
        with self.assertRaises(RuntimeError):
            HOST.container_action("bad container; exit", "stop")
        with self.assertRaises(RuntimeError):
            HOST.container_action("valid", "remove")

    def test_searches_homebrew_and_system_paths(self):
        self.assertIn("/opt/homebrew/bin", HOST.TOOL_PATHS)
        self.assertIn("/usr/sbin", HOST.TOOL_PATHS)
        self.assertTrue(HOST.executable("lsof").endswith("lsof"))

    def test_skips_start_when_machine_is_running(self):
        original_run = HOST.run
        HOST.run = lambda command, timeout=10: (0, "[]", "")
        try:
            self.assertEqual(HOST.start_podman_machine(), {"state": "running"})
        finally:
            HOST.run = original_run

    def test_recognizes_podman_connection_failures(self):
        self.assertTrue(HOST.podman_connection_error("dial tcp: connection refused"))
        self.assertTrue(HOST.podman_connection_error("ssh: handshake failed: connection reset by peer"))
        self.assertFalse(HOST.podman_connection_error("invalid container filter"))

    def test_podman_guard_is_reusable(self):
        with HOST.podman_guard():
            value = "locked"
        with HOST.podman_guard():
            value = "relocked"
        self.assertEqual(value, "relocked")


if __name__ == "__main__":
    unittest.main()
