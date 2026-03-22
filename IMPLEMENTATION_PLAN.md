# EgoGazeVQA Proactive Implementation Plan

## Goal

把原本偏向多选视频问答的 EgoGazeVQA 管线，扩展成更贴近 `prompt.md` 的主动意图推理框架：

- 输入最近几秒视频
- 使用 gaze 作为物理锚点
- 输出 `Reasoning / Intention / Answer / Object Class / Box`
- 支持 `<SILENCE>`
- 对比 `baseline` 与 `gaze-aware`

## 当前数据现实

EgoGazeVQA 提供：

- `metadata.csv`
- `ego4d.json / egoexo.json / egtea.json`
- clip 级 `mp4`
- gaze 点和 narration

EgoGazeVQA 不提供：

- 真实目标框 GT

所以主动定位部分只能先做弱监督 / 代理评测。

## 已完成

1. 数据下载与本地适配

- `scripts/download_egogazevqa.py`
- 支持 gated Hugging Face 下载
- 默认排除 `SFT_model/**`

2. 数据层扩展

- 解析 `metadata.csv`
- 加载 gaze 序列
- 支持全片均匀抽帧
- 支持最近几秒尾部时间窗抽帧

3. Proactive runner

- `src/ego_rog/egogazevqa_runner.py`
- 新增 proactive prompt
- 支持 `<SILENCE>`
- 支持 `Reasoning / Intention / Answer / Object Class / Localization`

4. 代理定位评测

- 基于最后一帧 gaze 点构造 proxy box
- 统计：
  - `intent_accuracy`
  - `has_box_rate`
  - `point_hit_rate`
  - `proxy_iou_ge_0_3`
  - `proxy_iou_ge_0_5`
  - `mean_center_distance_norm`

5. 新配置

- `configs/egogazevqa_proactive_gaze.yaml`
- `configs/egogazevqa_proactive_baseline.yaml`

6. 验证

- `inspect` 已通过
- proactive smoke dry run 已通过

## 当前最重要的下一步

1. 运行真实 `proactive_gaze`

```bash
python run_experiment.py --config configs/egogazevqa_proactive_gaze.yaml run
```

2. 运行真实 `proactive_baseline`

```bash
python run_experiment.py --config configs/egogazevqa_proactive_baseline.yaml run
```

3. 比较：

- `outputs/egogazevqa_qwen_proactive_gaze/metrics.json`
- `outputs/egogazevqa_qwen_proactive_baseline/metrics.json`

## 后续增强方向

1. 从 narration 中抽取候选目标词
2. 结合 gaze 和目标词生成更强的 proxy GT
3. 自动生成结果对比表
4. 导出可视化 demo 视频
