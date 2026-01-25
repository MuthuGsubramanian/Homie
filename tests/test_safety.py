import unittest

from homie.node.safety import is_command_safe


class SafetyTests(unittest.TestCase):
    def test_blocks_rm_rf_root(self):
        safe, why = is_command_safe("rm -rf /", {"blocked_substrings": [], "max_command_len": 300})
        self.assertFalse(safe)
        self.assertIn("blocked pattern", why)

    def test_enforces_max_len(self):
        long_cmd = "echo " + ("x" * 400)
        safe, _ = is_command_safe(long_cmd, {"blocked_substrings": [], "max_command_len": 100})
        self.assertFalse(safe)

    def test_allows_safe_command(self):
        safe, _ = is_command_safe("echo hello", {"blocked_substrings": [], "max_command_len": 300})
        self.assertTrue(safe)


if __name__ == "__main__":
    unittest.main()
