# Changes from SGLang

The engine is SGLang's `python/sglang` at upstream release v0.5.23 (commit `297e54fd10`), renamed to
`decoupled_engine`. Of its 6147 files, 85 are new and 139 are modified upstream files; the
rest differ from upstream only by the rename. Paths below are relative to `decoupled_engine/`.
[README.md](README.md#relation-to-sglang) says what the changes do.

To list the modified files again:

```bash
git grep -l 'Adapted from SGLang' -- evolving_infra/engine/decoupled_engine
```

## New files (85)

| Directory | Files |
|---|---|
| (top level) | `image_layout.py` |
| `srt/arg_groups/` | `symmetric_defaults.py` |
| `srt/disaggregation/` | `bootstrap_prefetch.py`, `decode_hicache_restore_log.py`, `decode_selection.py`, `jit_pull.py`, `late_d.py`, `late_d_link.py`, `late_d_need.py`, `lookahead_plan.py`, `native_eagle3_cache_layout.py`, `prebuilt_h2d.py`, `prefill_launch_thread.py`, `prefill_loop_unserialize.py`, `prefill_result_first.py`, `queued_prefix_guard.py` |
| `srt/disaggregation/pull/` | `__init__.py`, `api.py`, `importer.py`, `owner.py`, `protocol.py`, `read_engine.py` |
| `srt/disaggregation/symmetric/` | `README.md`, `__init__.py`, `admit_after_launch.py`, `consensus.py`, `export_table.py`, `fake_planes.py`, `import_queue.py`, `late_d_sym.py`, `launch_thread.py`, `marks.py`, `planes.py`, `publish_readback.py`, `scheduler_mixin.py` |
| `srt/entrypoints/openai/` | `encoding_dsv4.py` |
| `srt/layers/attention/dsv4/` | `visible_window.py` |
| `srt/managers/scheduler_components/` | `collective_guard.py`, `decode_load_consumer.py`, `deferred_write_submit.py`, `gate_pending_reorder.py`, `m2_lockstep_stats.py`, `next_batch_host_prefetch.py`, `phase_timers.py`, `prefill_routing_snapshot.py`, `prefill_state_push.py`, `prefill_timeline.py`, `prefill_work.py`, `sched_tracer.py` |
| `srt/mem_cache/` | `ce_2d_load.py`, `ce_loader.py`, `ce_peer.py`, `ce_submit.c`, `cpu_page_book.py`, `dedup_ipc_fanout.py`, `dsa_host_dedup.py`, `dsv4_host_dedup.py`, `dsv4_window_defaults.py`, `hicache_pipelined_sync.py`, `host_copy_bytes.py`, `load_quarantine.py`, `peer_gpu_export.py`, `peer_gpu_export_cache.py`, `peer_kv_demote.py`, `peer_kv_namespace.py`, `peer_kv_servicer.py`, `peer_kv_touch.py`, `staged_ce_load.py`, `swa_backfill.py` |
| `srt/mem_cache/allocator/` | `cpu_paged.py` |
| `srt/mem_cache/pool_host/` | `numa_split.py`, `numa_split_policy.py`, `shared_dedup.py`, `slot_allocator.py`, `vmm_share.py` |
| `srt/mem_cache/storage/peer_kv/` | `__init__.py`, `peer_kv_backend.py` |
| `srt/mem_cache/unified_cache/` | `retention_order.py`, `swa_ledger.py` |
| `srt/models/` | `deepseek_v4_vit.py`, `deepseek_v4_vl.py`, `glm5v.py` |
| `srt/multimodal/processors/` | `deepseek_v4_vl.py`, `glm5v.py` |
| `srt/speculative/dspark_components/` | `dspark_synthetic_oracle.py` |

## Modified upstream files (139)

Each carries the header `Adapted from SGLang <upstream path> (commit 297e54fd10)`.

| Directory | Files |
|---|---|
| (top level) | `__init__.py`, `launch_server.py` |
| `kernels/ops/attention/dsa/` | `transform_index.py` |
| `kernels/ops/attention/dsv4/` | `attn_metadata_kernels.py`, `sparse_prefill_kernels.py` |
| `kernels/ops/kvcache/` | `dsa_index_k_cache.py` |
| `kernels/ops/memory/` | `allocator.py`, `common.py` |
| `kernels/ops/speculative/dspark/` | `dspark_accept.py`, `dspark_verify_window.py` |
| `multimodal_gen/runtime/utils/` | `hf_diffusers_utils.py` |
| `srt/` | `environ.py`, `server_args.py` |
| `srt/arg_groups/` | `argparse_actions.py`, `attention_hook.py`, `deepseek_v4_hook.py`, `hicache_hook.py`, `kv_cache_hook.py`, `model_hook.py`, `overrides.py`, `pd_disaggregation_hook.py`, `pipeline.py`, `serving_hook.py`, `speculative_hook.py` |
| `srt/batch_overlap/` | `two_batch_overlap.py` |
| `srt/configs/` | `model_config.py` |
| `srt/disaggregation/` | `decode.py`, `decode_hicache_mixin.py`, `decode_schedule_batch_mixin.py`, `kv_events.py`, `prefill.py`, `utils.py` |
| `srt/disaggregation/base/` | `conn.py` |
| `srt/disaggregation/common/` | `conn.py` |
| `srt/disaggregation/mooncake/` | `conn.py` |
| `srt/disaggregation/nixl/` | `conn.py` |
| `srt/distributed/device_communicators/` | `mooncake_transfer_engine.py` |
| `srt/entrypoints/` | `engine.py`, `grpc_bridge.py`, `http_server.py` |
| `srt/entrypoints/openai/` | `protocol.py`, `serving_chat.py`, `serving_completions.py`, `utils.py` |
| `srt/function_call/` | `glm47_moe_detector.py`, `glm4_moe_detector.py`, `utils.py` |
| `srt/hardware_backend/npu/dsv4/` | `dsv4_memory_pool.py` |
| `srt/layers/attention/` | `deepseek_v4_backend.py`, `dsa_backend.py` |
| `srt/layers/attention/dsa/` | `dsa_indexer.py` |
| `srt/layers/attention/dsv4/` | `sparse_prefill_utils.py` |
| `srt/layers/moe/` | `hash_topk.py`, `mega_moe.py`, `topk.py` |
| `srt/managers/` | `cache_controller.py`, `detokenizer_manager.py`, `io_struct.py`, `load_snapshot.py`, `multi_tokenizer_mixin.py`, `schedule_batch.py`, `schedule_policy.py`, `scheduler.py`, `tokenizer_control_mixin.py`, `tokenizer_manager.py`, `tp_worker.py`, `utils.py` |
| `srt/managers/scheduler_components/` | `batch_result_processor.py`, `dp_attn.py`, `invariant_checker.py`, `kv_events_publisher.py`, `load_inquirer.py`, `metrics_reporter.py`, `output_streamer.py`, `request_receiver.py`, `weight_updater.py` |
| `srt/mem_cache/` | `allocation.py`, `base_prefix_cache.py`, `cache_init_params.py`, `common.py`, `deepseek_v4_memory_pool.py`, `events.py`, `hicache_storage.py`, `index_key_cache.py`, `kv_cache_builder.py`, `kv_cache_configurator.py`, `l2_transfer.py`, `memory_pool.py`, `memory_pool_host.py`, `radix_cache.py`, `registry.py`, `unified_radix_cache.py`, `utils.py` |
| `srt/mem_cache/allocator/` | `paged.py`, `swa.py` |
| `srt/mem_cache/cpp_utils/` | `native_hash.py` |
| `srt/mem_cache/hybrid_cache/` | `hybrid_cache_controller.py`, `hybrid_pool_assembler.py` |
| `srt/mem_cache/pool_host/` | `base.py`, `common.py`, `dsa.py`, `group.py`, `mla.py` |
| `srt/mem_cache/storage/mooncake_store/` | `mooncake_store.py` |
| `srt/mem_cache/unified_cache/` | `cache_action.py`, `session_ref_tracker.py`, `storage_attachment.py`, `unified_tree_core.py`, `unified_tree_core_interface.py` |
| `srt/mem_cache/unified_cache/components/` | `__init__.py`, `base.py`, `full.py`, `swa.py` |
| `srt/model_executor/` | `forward_batch_info.py`, `model_runner.py`, `pool_configurator.py` |
| `srt/models/` | `deepseek_v2.py`, `deepseek_v4.py`, `deepseek_v4_dspark.py`, `kimi_k25.py` |
| `srt/multimodal/processors/` | `kimi_k25.py` |
| `srt/observability/` | `metrics_collector.py`, `req_time_stats.py` |
| `srt/sampling/` | `sampling_batch_info.py`, `sampling_params.py` |
| `srt/speculative/` | `dflash_utils.py`, `eagle_disaggregation.py`, `eagle_info.py`, `eagle_utils.py`, `eagle_worker_common.py`, `eagle_worker_v2.py` |
| `srt/speculative/dspark_components/` | `dspark_verify.py`, `dspark_worker_v2.py` |
| `srt/utils/` | `common.py`, `device_timer.py`, `json_response.py`, `numa_utils.py`, `triton_load_watch.py` |
| `srt/utils/hf_transformers/` | `processor.py` |

## Fixes since the fork point

The files these fixes touch count as modified above.

- DeepSeek DSML: emit only complete invokes; reasoning ends only at a line-start `<｜DSML｜tool_calls>`; think tokens leave structural-tag excludes.
- GLM-4.7 tool parser keeps argument types across JSON Schema unions (`anyOf`/`oneOf`/`$ref`/`const`).
- GLM-4.5/4.7 streaming tool parsers scan only the new tail of an open call (was O(n²)).
- GLM non-streaming tool names are stripped of surrounding whitespace.
- DeepSeek V3.2/V4 encoders leave the tools out of the prompt when `tool_choice` is `none`.
- A prompt longer than the KV pool is rejected with OpenAI's "maximum context length" wording.
- With `--enable-cache-report`, usage carries `prompt_tokens_details.cached_tokens` even when it is 0.
- `reasoning_tokens` is clamped to the output kept after stop truncation.
- Tool schemas with draft-07 `$id` pass validation; an explicit `include_usage: false` overrides the server default.
- A streaming request whose client is gone is aborted at its next output; a closed chat stream closes its generator, and a closed or failed handler aborts the dispatched request it still owns.
- The DeepSeek V4 K norm/RoPE kernel reads its cache locations after the PDL wait.
- DeepSeek V4 routing: sqrtsoftplus no longer underflows or overflows; hash top-k renormalization is floored (no NaN).
- DSPARK verify rows without candidates keep their indices in range (no device assert on padded graph rows).
- HiCache write-through device-to-host copies wait for the forward stream.
- A split SWA node keeps its host lock and `host_uuid` on the new parent; host-locked halves stay out of the host LRU.
- The storage prefetch timeout is voted once per scheduler tick, not once per admission poll.
- Prefix hash values are collected iteratively (no RecursionError on deep chains).
