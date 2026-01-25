import unittest

from homie.utils import ensure_ip_literal
from homie.controller.node_api import NodeApiClient, NodeApiError


class IpPolicyTests(unittest.TestCase):
    def test_accepts_ipv4_literal(self):
        self.assertEqual(ensure_ip_literal("100.64.0.1"), "100.64.0.1")

    def test_rejects_hostname(self):
        with self.assertRaises(ValueError):
            ensure_ip_literal("my-host")

    def test_node_api_client_requires_ip(self):
        with self.assertRaises(NodeApiError):
            NodeApiClient("https://controller.local:8443", "secret")


if __name__ == "__main__":
    unittest.main()
