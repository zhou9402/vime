# 权重同步校验

移植自 miles@ba28991834(`backends/training_utils/weight_update/checksum_utils.py`、
`hf_weight_iterator/{bucketing,atomic_groups}.py`、`utils/weight_version.py`)。

vime 在全部三条权重同步路径(colocate IPC、NCCL distributed、disk-delta)上提供四层校验:

1. **传输 checksum** — `--check-weight-transfer-checksum`(`--ci-test` 下默认开):
   trainer 对每个 CPU 发送缓冲算 SHA-256;写入落地后通过 engine 的 `check_weights`
   RPC 取回逐 tensor 哈希,任一不一致则本次更新失败。
2. **同步后等值检查** — `--check-weight-update-equal`,配合
   `--check-weight-update-selector {all,target,draft}`(默认 auto = `target`,
   仅当配置了 `--vllm-speculative-config`/dspark draft 时才含 `draft`)、
   `--check-weight-update-skip-list`(按子串降级为非致命)、
   `--check-weight-update-allow-quant-error`(量化 tensor 在反量化空间比较,每侧 ≤1 ULP)。
   engine 侧 snapshot 默认**只存哈希**(不做 CPU 拷贝);
   `--check-weight-update-snapshot full` 才存 CPU 副本以支持 `reset_tensors` 恢复。
3. **原子更新组** — vLLM loader 会融合成单个 engine 参数的 HF 后缀组
   (如 DeepSeek `wq_a` + `wkv`)绝不跨桶拆分;fp8 weight+scale 永远作为一个单元发出。
4. **weight-version 过期守卫** — 断言每次更新后 driver 都发布 weight version,
   且 rollout 样本携带合法数字版本。v1 在样本提交级打版本戳;
   逐 token 的 engine 上报区间(类似 sglang 的 `meta_info["weight_versions"]`)
   是已知缺口,vLLM 侧尚未实现。

篡改检测由 `tests/test_weight_sync_tamper.py` 验证:同步后在 engine 侧翻转 1 bit,
`check_weights(action="compare")` 必须报错;不翻则必须通过。
