# strict_convert：trace 语料转严格回放 workload

该工具把本地 trace 转为 replay-trace schema v2 的 tier-A workload。每个
`generate` 都包含 `recorded_prompt_token_ids` 和
`recorded_output_token_ids`，且 `max_tokens` 与输出 token 数量严格相等。
回放时客户端直接发送 prompt token 数组并强制输出 token，不再重新套用 chat template。

当前项目的 BrowseComp 输入使用 GLM-5.2 源 trace，目标 tokenizer 为
DeepSeek-V4-Flash。MiniMax-M2.7 源语料和专用 adapter 已退役，不在本仓库中。

## 用法

```bash
python scripts/convert_traces_to_strict.py trie_text \
  --input /path/to/browsecomp-glm52.workload.jsonl \
  --target-tokenizer /path/to/deepseek-v4-flash \
  --chat-encoding dsv4 --thinking-mode thinking \
  --output /path/to/browsecomp-glm52.strict.jsonl \
  [--limit N] [--include SUBSTRING] [--workers N]

python scripts/convert_traces_to_strict.py trie_retokenize \
  --input /path/to/source-tier-a.jsonl \
  --source-tokenizer /path/to/glm-5.2 \
  --target-tokenizer /path/to/deepseek-v4-flash \
  --output /path/to/retokenized.strict.jsonl

python scripts/convert_traces_to_strict.py validate \
  --workload /path/to/browsecomp-glm52.strict.jsonl \
  --tokenizer /path/to/deepseek-v4-flash \
  --chat-encoding dsv4 --thinking-mode thinking --require-source-counts \
  [--deep-every N] [--workers N] [--skip-filter-check] [--report PATH]
```

每次转换同时写 `<output>.report.json`，记录输入选择、tokenizer 身份、转换计数、
prefix-chain 状态、长度分布、告警和输出哈希。`--limit` 以 adapter unit 为单位；
`--include` 强制纳入名称或 trace ID 含指定字符串的 unit。报告会冻结实际选择列表的
SHA-256，避免把切片误认为完整语料。
输出路径已存在时拒绝覆盖；修复语料必须另存，旧文件用于历史 run 复核。

## DSV4 编码

`--chat-encoding auto` 对目录名含 `deepseek-v4` 的 tokenizer 使用同一 SGLang checkout 内的
`engine/decoupled_engine/srt/entrypoints/openai/encoding_dsv4.py`，不经简版 Jinja。目录改名时显式指定 `dsv4`。
默认 `--thinking-mode thinking`；工具定义注入首个 system，DSML、角色、工具结果归并/排序由参考编码器处理。
`chat` 模式只适用于无录制 reasoning 的输出，有 reasoning 时拒绝丢弃或伪装为普通内容。
`add_generation_prompt=False` 关闭末尾待生成的 assistant 前缀；已完成的 assistant 不受影响。

历史模板产物使用 `--chat-encoding template` 复核。它保留紧凑 `<tool_call>` 代理格式，不等同原生 DSML。
新原生编码会改变 token 数与前缀结构，不能无说明地接替旧文件作为同口径性能对照。
这仍是将源模型的输出内容重新编码后强制回放，不是重新采集了目标模型的自然生成。

## 活动 adapter

| adapter | 输入 | 用途 |
| --- | --- | --- |
| `trie_text` | schema-v2、带结构化 assistant 文本的 tier-C JSONL | 用目标 tokenizer 生成完整 prompt/output token |
| `trie_retokenize` | 由另一 tokenizer 生成 token 的 tier-A JSONL | 先用源 tokenizer 恢复文本，再只生成目标 token；拒绝相同 tokenizer |
| `synthetic` | `generate_workload.py` shape-spec JSONL | 生成 token 数量精确、内容确定的合成严格回放 trace |

## 转换结构

```text
adapter -> CanonicalTrace -> builder -> schema-v2 JSONL + report
```

adapter 只负责把源格式翻译为统一的 `CanonicalTrace`。builder 维护消息状态、用目标
tokenizer 渲染 prompt/completed context、验证 token 解码无损，并生成最终事件。并行转换
使用 spawn 进程；每个 worker 写独立 part，父进程按原始 unit 顺序拼接，因此输出确定。
单个 JSONL 行可能很大；runner 会按源文件大小和物理内存估计 worker 上限，但无法消除单个
超大 unit 本身的峰值内存。

## 关键不变量

### Tokenizer 身份

- `tokenizer.json` SHA-256、vocab 大小和目标路径写入 report；紧凑身份同时写进每条 trace。
- `trie_retokenize` 只允许 source/target tokenizer 哈希不同。源 token 仅用于恢复文本，绝不
  写入目标 workload。
- 数值范围检查不能替代身份检查：来自另一词表的 ID 可能仍落在目标 vocab 范围内。
- 编码模式和参考编码器 SHA-256（或历史模板 SHA-256）写入 report 及每条 trace 的
  `strict_conversion.rendering`；validate 拒绝不匹配的编码配置。tokenizer 哈希不能替代模板身份。

### Prefix chain

对于 append-only 上下文，builder 要求下一轮 prompt 从
`prompt_ids[n] + output_ids[n]` 开始。若源 trace 重写历史、目标模板无法把 completed render
表示为 generation prompt 的直接延伸，或上一轮被截断，则明确记录 `context_rewrite` 并重新
编码完整 prompt，不伪造连续命中。

`metadata.strict_conversion.chain[]` 为每个 generate 记录：

- `chained` 或 `fresh`；
- break reason；
- output rule；
- 是否 truncated。

validator 会重新检查这些决定和实际 token prefix。

### 无损编码

所有生成 ID 的路径都必须 decode 回完全相同的文本，包括：

- fresh prompt；
- chained delta；
- prompt/output 边界修复后的输出。

若 tokenizer 规范化使文本无法逐字恢复，整条 trace 会以
`output_not_losslessly_tokenizable` 等明确原因丢弃，不能带着错误链继续转换。

### Reasoning 与截断

think tag 的拆分和前缀剥离只处理源 trace 已记录的文本，不假设某个模型的固定模板。
`TargetRenderer` 在启动时探测目标 generation tail、强制 suffix、角色保留以及
`reasoning_content` / `tool_calls` 是否被模板丢弃；builder 根据探测结果决定 role remap
与结构化字段的 re-inline。

长度截断的输出不补写模型没有生成的结束 tag 或终止 token，也不进入下一轮 chain。
空输出不满足 schema v2 的非空 token 约束，会降级为 `append_message` 并计数。

### Emission 与 history 的分离

每个 generate 拆成两种形态：**emission** 是完整录制生成
（reasoning + content + tool calls），决定 output ids；**history** 是源服务栈后续轮
真正回填的内容。默认 history 不回填 reasoning；源 token 链证明回填时保留，显式 replace_context
以源消息为准。对丢弃结构化
字段的模板，两种形态都把字段内联进 content：emission 内联 `<think>…</think>` 与
`serialize_tool_calls` 文本；history 只内联 tool calls。工作行的 `recorded_assistant`
在旧模板模式存 history 形态；DSV4 模式保留源 assistant 字段用于内容对账，回放以权威 prompt/output IDs 为准。
emission 与 history 不同的轮，chain 会用 history 渲染重新锚定
（`_rechain_on_history`），保持逐 token 前缀扩展性质；截断/降级轮照旧断链。
旧模板模式的 tool calls 使用确定性的紧凑形式；DSV4 模式直接使用参考 DSML。
DSV4 截断输出不补 EOS，只有 reasoning 的中途截断不补 `</think>`，避免人为闭合/重新打开 think 段。

### Role remap

若目标模板静默丢弃 `developer` 或非首个 `system`，renderer 会在探针证明丢失后才执行
确定性 remap：首个 `developer` 转 `system`，其他 `developer` 和非首个 `system` 转
`user`。每条 trace 和汇总 report 都记录 remap 计数。

## Synthetic adapter

`synthetic` 接受 `input_prompt_length`、`assistant_response_length[]`、`num_turns`、
`tool_call_latency[]`、`tool_call_output_length[]` 和
`final_assistant_response_length`。可选字段：

- `trace_id`；
- `reasoning_length`；
- `shared_prefix_length`；
- `start_offset_s`。

文本从经目标 tokenizer 验证为单 token 且边界稳定的词池生成；内容由
`seed_salt | trace_id | turn | role` 决定，不读取墙钟或全局随机状态。模板固定开销通过
实际渲染测量；无法满足目标 token 数的 shape 明确失败，不做静默截断。

## 验证

`validate` 分为两层：

1. 所有行执行 token 数、ID 范围、身份、prefix-chain 和逐请求源计数检查；原生 DSV4 还核对
   reasoning/content 全文及按序的每个 DSML 调用和参数，避免“总量接近但内容丢失”。
2. 默认所有行经当前 `ReplayTrace.from_jsonl_row` 的 schema/dataclass 准入检查（token 数组走 C primitives），
   每 `--deep-every N` 行额外检查 prompt 解码边界。

`--skip-filter-check` 只跳过全行 parser 准入，保留抽样深查，不再依赖已删除的独立 filter 脚本。
正式实验仍须由项目 endpoint conformance 验证服务端逐 token 强制输出，
离线转换通过不等于运行时已经准入。

**源保真对账**：当 `metadata.strict_conversion.original_counts` 携带采集侧
`recorded_output_tokens` / `expected_prompt_tokens` 或 Putnam 的 `completion_tokens` / `prompt_tokens`
时（trie adapter 从 `metadata.requests` 逐事件透传，也可按 source_request_id 回查），validate 逐 generate 比对 out_ids 数与采集自记
输出 token 数（跨 tokenizer 漂移带 [0.4, 2.5]，仅对 ≥32 token 的事件生效），并对
语料总量做 [0.7, 1.5]（输出）与 [0.6, 1.6]（prompt）比率门。chain 自洽通过不构成
保真声明；丢字段的转换在 chain 两侧一致地丢，只有对账能拦住。conversion report 的
`converter.code_files_sha256` 记录实际执行代码的源哈希（部署副本可能无 git 元数据）。
源 token IDs 的计数优先于采集自记计数。总量比率只对已配对的请求计算；发现部分覆盖时失败。
真实采集准入使用 `--require-source-counts`，确保 prompt/output 两侧都覆盖全部请求；没有计数时
明确报告 unavailable，不能将格式校验通过称为源保真通过。
