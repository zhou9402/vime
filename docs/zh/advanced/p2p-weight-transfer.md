# P2P / RDMA 权重传输

> 状态:进行中(从 miles @ ba28991834 移植,F01)。verifier 与训练侧传输包先行落地;
> vLLM 引擎后端与运行开关在后续 PR 合入。本文描述目标设计。

对于**分离部署**(非 colocate)的大规模训练,NCCL 广播权重同步的开销随引擎规模增长。
P2P 模式改为通过 RDMA(Mooncake transfer engine)直接写入各引擎。miles 的实测数据:
≤2 节点时比 NCCL 慢(+68–72%),≥4 节点时快 47–86%(GLM-5 744B:58.3 s → 8.5 s;
Kimi-K2 1T:53.3 s → 7.2 s 每次更新)。

## 设计(无副本,"Design A")

与 miles 不同(miles 在训练进程内通过 model replica 重建引擎的模型布局),vime 的移植不需要副本:

1. 每个引擎 worker 分配**一块固定(pinned)暂存缓冲**(大小为最大更新桶的 2 倍,双缓冲),
   注册到本地 Mooncake transfer engine,并通过 `get_p2p_transfer_info` RPC 发布
   `(session_id, staging_address, capacity)`。
2. 训练侧把 HF 布局的权重桶(带名称清单)打包进 pinned 传输缓冲,RDMA 写入所分配引擎的暂存缓冲。
3. 引擎**用自己的 vLLM loader 对暂存字节做重分片**(`load_weights_from_staged`),并返回逐参数的
   SHA-256 校验值;训练侧在释放写窗口前完成校验。

无需训练侧 model replica,也不要求训练/引擎的版本或 GPU 架构一致。

## 限制(与 miles 一致)

- 仅分离部署:与 `--colocate` 互斥(colocate 走 CUDA IPC)。
- v1 不支持 LoRA,不支持投机解码。
- Mooncake 为可选依赖(`pip install vime[p2p]`,即 `mooncake-transfer-engine`);
  不启用 P2P 时不会 import。
- 训练与引擎主机之间需要可路由的 RDMA(或 TCP 回退:`MOONCAKE_PROTOCOL=tcp`);
  `MOONCAKE_DEVICE` 选择 RDMA 网卡。

## 失败语义

- 权重写入**绝不重试**(weight-update session 设计上不可重试;miles #4402 的重试加固只覆盖
  Mooncake *对象存储* 操作)。写入失败会把对应引擎标记为 errored,更新继续在健康引擎上进行
  (partial-target 隔离)。
- 完成状态未知的写入("uncertain write")还会永久钉住已注册的传输缓冲内存:此后不再注销。
- 每个权重桶在到达时做 SHA-256 校验(与通用 `check_weights` 加固 F02 收敛)。

## 用法(目标形态,F01-P3 之后)

```bash
python train.py ... --update-weight-transport p2p --p2p-transfer-timeout 30
```

## 引擎后端(vLLM patch)

`docker/patch/latest/vllm-p2p-weight-transfer.patch` 为 vLLM 增加 `p2p` 权重传输后端
(`vllm/distributed/weight_transfer/p2p_engine.py`):每个 worker 分配一块注册到 Mooncake 的
暂存缓冲,并以 worker RPC(经 `/collective_rpc`)暴露 `get_p2p_transfer_info` /
`load_weights_from_staged`。后者先校验暂存字节的逐参数 SHA-256,再经模型自身的
`load_weights` 加载。Mooncake 仅在 `--build-arg ENABLE_P2P=1` 时安装。

## Verifier

`python -m pytest tests/test_p2p_weight_transfer.py -x -q` 用 fake 覆盖合约:
与直接拷贝的字节等价、篡改检测、瞬态重试、永久失败隔离。
