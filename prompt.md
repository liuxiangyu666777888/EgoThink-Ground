# 基于 EgoGazeVQA + Qwen3-VL 的注视驱动主动意图推理实验设计

## 1. 任务目标

在第一视角短视频中，利用最近几秒的视觉上下文和当前 gaze 坐标，推断佩戴者的隐式短期意图，并在当前帧中定位最相关的目标对象。

输入：

- 最近一段时间的多帧视频
- 当前帧对应的 gaze 坐标
- 多项选择问题与候选答案

输出：

- `Reasoning`
- `Intention`
- `Answer`
- `Object Class`
- `Localization: <box>[[...]]</box>`

## 2. 创新点

### 2.1 Gaze 作为物理锚点

不是把 gaze 仅仅当成附加文本，而是把它作为主动推理的物理锚点：

- 用 gaze 约束意图推理
- 用 gaze 缩小候选目标范围
- 用 gaze 评估定位是否贴近当前注意目标

### 2.2 从 QA 扩展到主动推理

EgoGazeVQA 原始形式更偏向视频问答。本项目将其改造成 `proactive reasoning` 管线：

- 不只回答选项
- 同时生成隐式意图解释
- 同时输出目标物框

### 2.3 原始框与修正框双口径评估

EgoGazeVQA 没有真实目标框标注，因此不能只看单一定位分数。本项目明确区分：

- `raw localization`：模型原始输出框
- `refined localization`：利用 gaze 几何锚点修正后的最终框

这样可以分别衡量：

- 模型原生定位能力
- gaze 后处理对最终系统效果的提升

## 3. 优势

- 数据获取门槛低，EgoGazeVQA 可直接通过 Hugging Face 下载
- 与 Qwen3-VL 的多图输入和框输出协议兼容
- 可在无额外训练的前提下快速验证“注视驱动主动推理”的可行性
- 评估设计更完整，避免把模型原始框能力和 gaze 后处理收益混为一谈

## 4. 实验方法

### 4.1 数据

主数据集使用 `taiyi09/EgoGazeVQA`，包含：

- `metadata.csv`
- `ego4d.json`
- `egoexo.json`
- `egtea.json`
- 对应视频片段

### 4.2 模型设置

支持两类配置：

- `thinking`：高成本、慢速、长推理
- `fast`：低成本、短上下文、适合大规模跑实验

当前主线推荐优先使用 `qwen3-vl-flash` 的 fast 配置完成大规模实验，再用 thinking 配置做高成本对照。

### 4.3 Prompt 设计

核心约束：

- 最近几帧按时间顺序输入
- 最后一帧是当前帧
- gaze 同时以归一化坐标和像素坐标写入 prompt
- 要求模型输出固定五段结构
- 要求框必须是当前帧绝对像素坐标
- 要求框优先贴近最新 gaze 附近的可见目标

### 4.4 定位后处理

由于模型经常输出：

- 偏离 gaze 的框
- 0-1000 归一化坐标
- 语义正确但几何位置错误的框

因此系统加入两步后处理：

1. 自动识别并缩放 `0-1` 或 `0-1000` 形式的坐标
2. 当原始框明显偏离 gaze 时，用 gaze 重新锚定框中心，保留原框宽高，生成 refined box

## 5. 评估指标

### 5.1 语义指标

- `accuracy`：答案是否正确
- `intent_accuracy`：语义一致口径下的意图正确率
- `strict_intent_accuracy`：严格词面匹配口径下的意图正确率

### 5.2 定位指标

由于无真实 GT box，使用 gaze proxy box 做弱监督评估。

原始框指标：

- `raw_has_box_rate`
- `raw_point_hit_rate`
- `raw_mean_proxy_iou`
- `raw_proxy_iou_ge_0_3`
- `raw_proxy_iou_ge_0_5`
- `raw_mean_center_distance_norm`

最终框指标：

- `has_box_rate`
- `point_hit_rate`
- `mean_proxy_iou`
- `proxy_iou_ge_0_3`
- `proxy_iou_ge_0_5`
- `mean_center_distance_norm`

后处理使用情况：

- `refinement_applied_rate`

## 6. 当前实验结论

在 `fast_eval20_v3` 小样本实验中：

- `accuracy = 0.90`
- `intent_accuracy = 0.90`
- `raw_mean_proxy_iou = 0.0581`
- `mean_proxy_iou = 0.3381`
- `raw_point_hit_rate = 0.05`
- `point_hit_rate = 1.00`
- `refinement_applied_rate = 0.95`

结论：

- 模型在答案与意图层面已经具备较强可用性
- 模型原始框能力仍然偏弱
- gaze-guided refinement 对最终定位性能贡献很大

## 7. 实验交付物

- 可运行的数据下载脚本
- EgoGazeVQA 主动推理 runner
- baseline / proactive / fast / thinking 多套配置
- 原始框与修正框双口径指标
- 可视化导出脚本
- 结果目录与复现实验命令
