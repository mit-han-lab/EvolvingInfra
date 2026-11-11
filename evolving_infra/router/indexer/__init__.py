"""KV directory: supervises the pinned Dynamo indexer plus the event bridge and hash-query adapter.

Importing this package starts no process. Run the supervisor with
``python -m evolving_infra.router.indexer --config deployment.json --output new-dir``.
"""
