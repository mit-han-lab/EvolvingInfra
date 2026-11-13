# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""KV directory: supervises the pinned Dynamo indexer plus the event bridge and hash-query adapter.

Importing this package starts no process. Run the supervisor with
``python -m evolving_infra.router.indexer --config deployment.json --output new-dir``.
"""
