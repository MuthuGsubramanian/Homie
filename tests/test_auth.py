import hashlib
import hmac
import unittest

from homie.controller.node_api import _build_signature


class AuthSignatureTests(unittest.TestCase):
    def test_signature_matches_reference(self):
        secret = "supersecret"
        method = "POST"
        path = "/status"
        body = "{}"
        ts = 1700000000

        expected = hmac.new(secret.encode(), f"{method}|{path}|{ts}|{body}".encode(), hashlib.sha256).hexdigest()
        self.assertEqual(_build_signature(secret, method, path, body, ts), expected)


if __name__ == "__main__":
    unittest.main()
