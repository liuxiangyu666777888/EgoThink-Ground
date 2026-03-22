# SOTA Experiments

当前仓库支持两条实验线：

- `egogazevqa`：主线，基于 EgoGazeVQA 的第一视角视频意图推理 / 视频问答
- `egointention`：保留的 legacy grounding 管线

## 当前主线

现在的 EgoGazeVQA 后端已经支持两种运行模式：

1. `qa` 模式

- 输入多帧视频 + 可选 gaze 文本提示
- 输出多选题答案字母
- 指标是 `accuracy`

2. `proactive` 模式

- 输入最近 5 秒左右的多帧视频 + 当前 gaze 锚点
- 输出：
  - `Reasoning`
  - `Intention`
  - `Answer`
  - `Object Class`
  - `Localization: <box>...</box>`
- 支持 `<SILENCE>`
- 更接近 `prompt.md` 中的“注视驱动主动推理”

## 重要限制

EgoGazeVQA 原始数据没有真实目标框标注，只有：

- QA 标签
- gaze 坐标
- narration

所以当前 `proactive` 模式里的定位评测是代理评测，不是严格 GT box 评测。

当前使用的代理指标包括：

- `intent_accuracy`
- `has_box_rate`
- `point_hit_rate`
- `proxy_iou_ge_0_3`
- `proxy_iou_ge_0_5`
- `mean_center_distance_norm`

其中代理框来自“当前 gaze 点在最后一帧附近构造的 box”。

## 数据目录

默认目录：

```text
dataset/egogazevqa/
  metadata.csv
  ego4d.json
  egoexo.json
  egtea.json
  ego4d/<video_id>/<start>_<end>.mp4
  egoexo/<video_id>/<start>_<end>.mp4
  egtea/<video_id>/<start>_<end>.mp4
```

## 下载数据

EgoGazeVQA 是 Hugging Face gated dataset。先完成：

1. 在数据页接受访问条款  
   https://huggingface.co/datasets/taiyi09/EgoGazeVQA
2. 登录 Hugging Face

```bash
hf auth login
```

然后执行：

```bash
python scripts/download_egogazevqa.py
```

只下载标注和 narration：

```bash
python scripts/download_egogazevqa.py --metadata-only
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 统一命令

所有实验都通过同一入口运行：

```bash
python run_experiment.py --config <config_path> <command>
```

支持命令：

- `inspect`
- `sample`
- `run`
- `evaluate`

## 常用配置

视频问答：

- [configs/egogazevqa_textual.yaml](C:\Users\Administrator\Desktop\SOTA\configs\egogazevqa_textual.yaml)
- [configs/egogazevqa_baseline.yaml](C:\Users\Administrator\Desktop\SOTA\configs\egogazevqa_baseline.yaml)

主动推理：

- [configs/egogazevqa_proactive_gaze.yaml](C:\Users\Administrator\Desktop\SOTA\configs\egogazevqa_proactive_gaze.yaml)
- [configs/egogazevqa_proactive_baseline.yaml](C:\Users\Administrator\Desktop\SOTA\configs\egogazevqa_proactive_baseline.yaml)

Legacy：

- [configs/qwen_rog.yaml](C:\Users\Administrator\Desktop\SOTA\configs\qwen_rog.yaml)

## 推荐流程

### 1. 检查数据

```bash
python run_experiment.py --config configs/egogazevqa_proactive_gaze.yaml inspect
```

### 2. 生成平衡 manifest

```bash
python run_experiment.py --config configs/egogazevqa_proactive_gaze.yaml sample --per-task 50 --output outputs/egogazevqa_manifest.jsonl
```

### 3. 设置 API key

PowerShell:

```powershell
$env:QWEN_API_KEY="你的key"
```

### 4. 跑 proactive gaze

```bash
python run_experiment.py --config configs/egogazevqa_proactive_gaze.yaml run
```

### 5. 跑 proactive baseline

```bash
python run_experiment.py --config configs/egogazevqa_proactive_baseline.yaml run
```

### 6. 重算已有结果

```bash
python run_experiment.py --config configs/egogazevqa_proactive_gaze.yaml evaluate
```

## 输出目录

每次运行会生成：

```text
outputs/<exp_name>/
  resolved_config.json
  results.jsonl
  metrics.json
  _decoded_frames/
  visuals/
```

说明：

- `resolved_config.json`：实际生效配置
- `results.jsonl`：逐样本明细
- `metrics.json`：汇总指标
- `_decoded_frames/`：视频抽帧缓存
- `visuals/`：代理框可视化结果

## 关键代码

- 下载脚本：[scripts/download_egogazevqa.py](C:\Users\Administrator\Desktop\SOTA\scripts\download_egogazevqa.py)
- 数据层：[src/ego_rog/egogazevqa_data.py](C:\Users\Administrator\Desktop\SOTA\src\ego_rog\egogazevqa_data.py)
- 主 runner：[src/ego_rog/egogazevqa_runner.py](C:\Users\Administrator\Desktop\SOTA\src\ego_rog\egogazevqa_runner.py)
- 配置层：[src/ego_rog/config.py](C:\Users\Administrator\Desktop\SOTA\src\ego_rog\config.py)
- 统一 CLI：[src/ego_rog/cli.py](C:\Users\Administrator\Desktop\SOTA\src\ego_rog\cli.py)
