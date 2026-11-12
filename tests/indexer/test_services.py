# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Exercise the supervisor's HTTP contract, including Dynamo's empty health body."""
import http.server
import threading
import unittest
import urllib.error

from evolving_infra.router.indexer.services import request


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(503 if self.path == '/unhealthy' else 200)
        self.end_headers()
        if self.path == '/registration':
            self.wfile.write(b'{"registered":true}')
        elif self.path == '/invalid':
            self.wfile.write(b'not-json')

    def log_message(self, *args):
        pass


class HealthContract(unittest.TestCase):
    def test_empty_health_and_json_registration_with_failure_propagation(self):
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            self.assertIsNone(request(base + '/health'))
            self.assertEqual(request(base + '/registration'), {'registered': True})
            with self.assertRaises(urllib.error.HTTPError):
                request(base + '/unhealthy')
            with self.assertRaises(ValueError):
                request(base + '/invalid')
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
