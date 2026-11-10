"""Prepared-query wire equivalence and rejection of ambiguous inputs."""
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from evolving_infra.router.indexer.eagle_indexer_query import create_app, query_payload


class QueryContract(unittest.TestCase):
    def test_prepared_path_never_hashes_again(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("prepared path must not hash again")
        self.assertEqual(query_payload({
            "model_name": "glm", "cache_salt": "already-in-chain",
            "block_hashes": [0, 2**63, 2**64 - 1],
        }, 64, forbidden), {
            "model_name": "glm", "block_hashes": [0, -(2**63), -1],
        })

    def test_legacy_eagle_namespace_and_prepared_wire_are_identical(self):
        def hasher(tokens, size, **kwargs):
            self.assertEqual(tokens, [1, 2, 3])
            self.assertEqual(size, 64)
            self.assertEqual(kwargs, dict(is_eagle=True, cache_namespace="session"))
            return [2**64 - 1]
        legacy = query_payload({"model_name": "glm", "token_ids": [1, 2, 3],
                                "cache_salt": "session"}, 64, hasher)
        prepared = query_payload({"model_name": "glm", "block_hashes": [-1]}, 64)
        self.assertEqual(legacy, prepared)

    def test_plain_page_engines_hash_without_eagle_windows(self):
        def hasher(tokens, size, **kwargs):
            self.assertEqual(kwargs, dict(is_eagle=False, cache_namespace=None))
            return [7]
        self.assertEqual(query_payload({"model_name": "dsv4", "token_ids": [1] * 256}, 256, hasher,
                                       is_eagle=False),
                         {"model_name": "dsv4", "block_hashes": [7]})

    def test_bad_hashes_and_ambiguous_sources_rejected(self):
        for extra in ({}, {"token_ids": [], "block_hashes": []},
                      {"block_hashes": [True]}, {"block_hashes": [2**64]},
                      {"block_hashes": [-(2**63)-1]}, {"block_hashes": ["ff"]}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                query_payload({"model_name": "glm", **extra}, 64)


class PreparedQueryOverHttp(unittest.IsolatedAsyncioTestCase):
    async def test_router_prepared_query_reaches_query_by_hash(self):
        """The router's prepared body (u64 hashes, model_name) must reach Dynamo's
        /query_by_hash as signed hashes without cache_salt; /query is never used."""
        seen = []

        async def query_by_hash(req):
            seen.append((req.path, await req.json()))
            return web.json_response({"instances": {"20001": {"gpu": 256, "cpu": 0}}})

        async def query(req):  # Dynamo's /query: token_ids required
            seen.append((req.path, await req.json()))
            return web.Response(status=422)

        upstream = web.Application()
        upstream.router.add_post('/query_by_hash', query_by_hash)
        upstream.router.add_post('/query', query)
        async with TestServer(upstream) as up:
            app = create_app(str(up.make_url('')), 256, is_eagle=False)
            async with TestClient(TestServer(app)) as client:
                response = await client.post('/query', json={
                    "block_hashes": [2**64 - 1, 5], "model_name": "deepseek-v4-flash"})
                self.assertEqual(response.status, 200)
                self.assertEqual((await response.json())["instances"]["20001"]["gpu"], 256)
        self.assertEqual(seen, [('/query_by_hash',
                                 {"model_name": "deepseek-v4-flash", "block_hashes": [-1, 5]})])
