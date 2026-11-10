"""Dynamo owns the directory. Prepared hashes bypass central token hashing.

Every query leaves through Dynamo's `/query_by_hash`: token IDs are hashed here
with the engines' page contract (EAGLE windows or plain pages), prepared hashes
(the distributed tokenizer's, already salted) pass through. Dynamo's own `/query`
accepts token IDs only, so it cannot serve the prepared path.
"""
import argparse
import json
import time

from aiohttp import ClientSession, ClientTimeout, web


def signed_hash(value):
    """Normalize block hashes to the directory query wire representation.

    Args:
        value: Signed or unsigned 64-bit integer block hash; booleans are rejected.

    Returns:
        Equivalent signed 64-bit integer preserving the original bits; raises
        ValueError outside the accepted domain.
    """
    if type(value) is not int or not -(2**63) <= value < 2**64:
        raise ValueError("block hashes must be signed or unsigned 64-bit integers")
    return value if value < 2**63 else value - 2**64


def query_payload(body, block_size, hasher=None, is_eagle=True):
    """Build a Dynamo query from token IDs or already prepared hashes.

    Args:
        body: Mapping with model_name and exactly one of token_ids or block_hashes;
            optional cache_salt for token hashing.
        block_size: Tokens per directory block, passed to the hasher.
        hasher: Optional compatible hash function; defaults to Dynamo when tokens are
            supplied.
        is_eagle: The engines' page hashing: EAGLE overlapping windows or plain pages.

    Returns:
        Mapping of model_name and signed block_hashes. Validates the input schema;
        prepared hashes are not recomputed.
    """
    model = body.get("model_name")
    if not isinstance(model, str) or not model:
        raise ValueError("model_name is required")
    tokens, hashes = body.get("token_ids"), body.get("block_hashes")
    if (tokens is None) == (hashes is None):
        raise ValueError("provide exactly one of token_ids or block_hashes")
    if hashes is not None:
        if not isinstance(hashes, list):
            raise ValueError("block_hashes must be a list")
        # Namespace and page form are already included in the prepared hash chain.
    else:
        if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("token_ids must be nonnegative integers")
        if hasher is None:
            from dynamo.llm import compute_block_hash_for_seq
            hasher = compute_block_hash_for_seq
        hashes = hasher(tokens, block_size, is_eagle=is_eagle,
                        cache_namespace=body.get("cache_salt") or None)
    return {"model_name": model, "block_hashes": [signed_hash(h) for h in hashes]}


def create_app(upstream, block_size=64, hasher=None, is_eagle=True):
    """Construct the directory query adapter.

    Args:
        upstream: Dynamo indexer HTTP base URL.
        block_size: Positive directory block size in tokens.
        hasher: Optional hash implementation for token-ID queries.
        is_eagle: Hash token-ID queries as EAGLE windows (else plain pages).

    Returns:
        Configured aiohttp application with query and health routes; not yet listening.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    async def resources(app):
        """Own the application HTTP session across startup and shutdown.

        Args:
            app: Application receiving a shared outbound HTTP session.

        Yields:
            Yields None once while the session is available, then closes it during cleanup.
        """
        app["client"] = ClientSession(timeout=ClientTimeout(total=1))
        yield
        await app["client"].close()

    async def query(req):
        """Validate and forward a query to Dynamo query_by_hash.

        Args:
            req: Incoming aiohttp request containing query JSON.

        Returns:
            HTTP response with the upstream query status/body, or HTTP 400 for invalid
            input. Logs timing and counts without full token contents.
        """
        start = time.perf_counter()
        try:
            body = await req.json()
            if not isinstance(body, dict):
                raise ValueError("expected an object")
            payload = query_payload(body, block_size, hasher, is_eagle)
        except (ValueError, TypeError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        hash_ms = (time.perf_counter() - start) * 1000
        async with req.app["client"].post(upstream.rstrip('/') + '/query_by_hash',
                                          json=payload) as response:
            data, status = await response.read(), response.status
        print(json.dumps(dict(at=time.time(), tokens=len(body.get("token_ids", [])),
                              prepared="block_hashes" in body,
                              blocks=len(payload["block_hashes"]), hash_ms=hash_ms,
                              total_ms=(time.perf_counter() - start) * 1000,
                              status=status)), flush=True)
        return web.Response(body=data, status=status, content_type="application/json")

    async def health(req):
        """Proxy the directory health check without generating a cache query.

        Args:
            req: Incoming request providing the application HTTP session.

        Returns:
            Upstream health status and response body; transport failures propagate.
        """
        async with req.app["client"].get(upstream.rstrip('/') + '/health') as response:
            return web.Response(body=await response.read(), status=response.status,
                                content_type="application/json")

    app = web.Application(client_max_size=8 * 1024**2)
    app.cleanup_ctx.append(resources)
    app.router.add_post('/query', query)
    app.router.add_get('/health', health)
    return app


def main():
    """Run the query adapter from command-line configuration.

    Args:
        None. Uses the enclosing service state or process arguments.

    Returns:
        None after the web server stops. Reads process arguments and binds the adapter
        to loopback.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--eagle', action=argparse.BooleanOptionalAction, default=True,
                        help='hash token-ID queries as EAGLE windows (--no-eagle: plain pages)')
    args = parser.parse_args()
    web.run_app(create_app(args.upstream, args.block_size, is_eagle=args.eagle), host='127.0.0.1',
                port=args.port, access_log=None)


if __name__ == '__main__':
    main()
