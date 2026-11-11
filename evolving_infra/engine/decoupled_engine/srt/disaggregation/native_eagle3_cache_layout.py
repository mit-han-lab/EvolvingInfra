"""Admit native flat-MLA EAGLE3 radix only with a complete draft cache."""


def validate_native_eagle3_cache(tree, target, draft, *, host_enabled):
    def require(ok, detail):
        if not ok:
            raise ValueError("Native EAGLE3 radix cache requires " + detail)

    def inherits(obj, name):
        return obj is not None and any(
            cls.__name__ == name for cls in type(obj).__mro__
        )

    require(
        inherits(target, "MLATokenToKVPool")
        and not inherits(target, "DSATokenToKVPool"),
        "a flat MLA target",
    )
    require(
        inherits(draft, "MLATokenToKVPool") and not inherits(draft, "DSATokenToKVPool"),
        "a flat MLA draft",
    )
    require(draft.layer_num == 1, "one draft layer")
    require(
        target.page_size == draft.page_size == tree.page_size, "identical page sizes"
    )
    require(target.size == draft.size, "shared target/draft token index capacity")
    require(
        inherits(tree, "UnifiedRadixCache") and tree.tree_core.is_eagle,
        "a unified EAGLE bigram tree",
    )
    if not host_enabled:
        return
    # L3 storage: only the peer_kv backend, whose servicer serves the draft
    # sidecar with the target pages (checked below). Other backends do not
    # attest the draft pages, so a hit could leave the draft prefix stale.
    peer_storage = bool(tree.enable_storage) and inherits(
        getattr(tree.cache_controller, "storage_backend", None), "PeerKVBackend"
    )
    require(
        not tree.enable_storage or peer_storage,
        "a local CPU cache without L3 storage, or peer_kv storage",
    )
    group = tree.cache_controller.mem_pool_host
    entries = list(group.entry_map.values())
    targets = [
        e for e in entries if e.device_pool is target and e.is_primary_index_anchor
    ]
    drafts = [e for e in entries if e.device_pool is draft]
    require(
        len(targets) == len(drafts) == 1,
        "one live target anchor and one live draft sidecar",
    )
    anchor, sidecar = targets[0], drafts[0]
    if peer_storage:
        require(
            getattr(sidecar, "indices_from_pool", None) == anchor.name,
            "a draft sidecar served on the target anchor's pages (peer_kv)",
        )
    host, draft_host = anchor.host_pool, sidecar.host_pool
    require(
        inherits(host, "MLATokenToKVPoolHost")
        and inherits(draft_host, "MLATokenToKVPoolHost"),
        "MLA target and draft host pools",
    )
    # The sidecar builder sizes from the anchor, then HostKVCache rounds up
    # with one additional page. Only anchor indices are used for transfers;
    # the draft allocation must cover them, but may include that spare page.
    require(
        draft_host.logical_size >= host.logical_size
        and host.page_size == draft_host.page_size,
        "draft host slots covering every target host slot",
    )
    require(
        host.layer_num == target.layer_num and draft_host.layer_num == 1,
        "all target and draft layers in backup and restore",
    )
    require(
        tree.cache_controller.layer_num == target.layer_num,
        "a completion counter covering every target layer",
    )
    specs = [s for s in tree.sidecar_pool_specs if s.pool_name == sidecar.name]
    require(len(specs) == 1, "an attached draft sidecar specification")
    spec = specs[0]
    require(
        spec.indices_from_pool == anchor.name,
        "draft pages coupled to the target anchor indices",
    )
    require(
        str(getattr(spec.hit_policy, "value", spec.hit_policy)) == "all_pages",
        "all draft pages before a CPU cache hit is usable",
    )
