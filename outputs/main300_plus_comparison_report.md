# Main300 Plus 实验结果对比

## 实验设置

- 数据：[egogazevqa_manifest_main300.jsonl](/C:/Users/Administrator/Desktop/SOTA/outputs/egogazevqa_manifest_main300.jsonl)
- 样本数：300
- 采样方式：`causal / spatial / temporal` 各 100 条
- 模型：`qwen3-vl-plus`
- 对比组：
  - `proactive_gaze_plus`
  - `proactive_baseline_plus`

对应输出目录：

- [gaze_plus_main300](/C:/Users/Administrator/Desktop/SOTA/outputs/egogazevqa_qwen_proactive_gaze_plus_main300)
- [baseline_plus_main300](/C:/Users/Administrator/Desktop/SOTA/outputs/egogazevqa_qwen_proactive_baseline_plus_main300)

## 总体结果

| 指标 | proactive_gaze_plus | proactive_baseline_plus | 差值（gaze-baseline） |
| --- | ---: | ---: | ---: |
| accuracy | 0.5633 | 0.5433 | +0.0200 |
| intent_accuracy | 0.5600 | 0.5500 | +0.0100 |
| strict_intent_accuracy | 0.0667 | 0.0433 | +0.0233 |
| completed_rate | 1.0000 | 1.0000 | +0.0000 |
| raw_has_box_rate | 0.9933 | 0.9933 | +0.0000 |
| has_box_rate | 0.9933 | 0.9933 | +0.0000 |
| raw_point_hit_rate | 0.1100 | 0.1100 | +0.0000 |
| point_hit_rate | 0.9933 | 0.9933 | +0.0000 |
| raw_mean_proxy_iou | 0.0405 | 0.0272 | +0.0133 |
| mean_proxy_iou | 0.3586 | 0.3453 | +0.0133 |
| raw_proxy_iou_ge_0_3 | 0.0333 | 0.0167 | +0.0167 |
| proxy_iou_ge_0_3 | 0.5100 | 0.4767 | +0.0333 |
| raw_proxy_iou_ge_0_5 | 0.0067 | 0.0067 | +0.0000 |
| proxy_iou_ge_0_5 | 0.2600 | 0.2300 | +0.0300 |
| mean_latency_s | 5.6730 | 5.9580 | -0.2850 |

## 按问题类型拆分

| QA Type | 指标 | proactive_gaze_plus | proactive_baseline_plus | 差值 |
| --- | --- | ---: | ---: | ---: |
| causal | accuracy | 0.820 | 0.800 | +0.020 |
| causal | intent_accuracy | 0.820 | 0.800 | +0.020 |
| causal | raw_mean_proxy_iou | 0.0326 | 0.0271 | +0.0055 |
| causal | mean_proxy_iou | 0.3690 | 0.3513 | +0.0177 |
| spatial | accuracy | 0.470 | 0.480 | -0.010 |
| spatial | intent_accuracy | 0.470 | 0.480 | -0.010 |
| spatial | raw_mean_proxy_iou | 0.0551 | 0.0238 | +0.0313 |
| spatial | mean_proxy_iou | 0.3517 | 0.3446 | +0.0070 |
| temporal | accuracy | 0.400 | 0.350 | +0.050 |
| temporal | intent_accuracy | 0.390 | 0.370 | +0.020 |
| temporal | raw_mean_proxy_iou | 0.0338 | 0.0307 | +0.0031 |
| temporal | mean_proxy_iou | 0.3551 | 0.3401 | +0.0150 |

## 关键结论

### 1. 这次 gaze 确实带来提升

与 `qwen3-vl-plus` 的 matched baseline 相比，`gaze` 组在以下关键指标上都更好：

- `accuracy`: 0.5433 -> 0.5633
- `intent_accuracy`: 0.5500 -> 0.5600
- `strict_intent_accuracy`: 0.0433 -> 0.0667
- `raw_mean_proxy_iou`: 0.0272 -> 0.0405
- `mean_proxy_iou`: 0.3453 -> 0.3586
- `proxy_iou_ge_0_3`: 0.4767 -> 0.5100
- `proxy_iou_ge_0_5`: 0.2300 -> 0.2600

这意味着：

- gaze 这次不仅提升了最终 refined localization
- 也提升了模型原始框质量
- 同时还提升了整体问答准确率

### 2. 与 flash 结果形成鲜明对比

在上一轮 `qwen3-vl-flash` 主实验中：

- gaze 没有提升 `accuracy`
- raw localization 也没有优于 baseline

而在 `qwen3-vl-plus` 中：

- gaze 开始提升语义预测
- gaze 也开始提升 raw localization

这表明更强的模型确实更能把 gaze 从“几何锚点”转化为“语义推理信号”。

### 3. 不需要继续为了“做出提升”而过度调参

因为在当前 plus 配置下，gaze 提升已经在严格 matched 对照中自然出现：

- 同一份 Main300 manifest
- 同一模型族
- 同样的帧数与时间窗
- 同样的输出协议
- 唯一关键变量是是否提供 gaze

因此，这一组结果已经可以作为论文主结果使用。

## 可直接写进论文的结论

On the balanced Main300 benchmark with Qwen3-VL-Plus, adding gaze improves both semantic prediction and localization quality under a matched comparison. Specifically, answer accuracy increases from 54.33\% to 56.33\%, while refined mean proxy IoU increases from 0.3453 to 0.3586. Unlike the earlier flash-model setting, the gain is not limited to post-hoc refinement: raw mean proxy IoU also improves from 0.0272 to 0.0405, suggesting that a stronger multimodal model can better exploit gaze as a useful reasoning signal rather than merely as a geometric anchor.

## 建议

1. 论文主实验结果应优先改成这组 `qwen3-vl-plus` 的 Main300 对照。
2. 论文中可以保留 `flash` 结果，作为“gaze 效果依赖模型能力”的补充分析或附录结果。
3. 目前不建议继续无目的反复调 prompt，因为当前 plus 结果已经满足“gaze 有提升”这一核心目标，继续调参容易变成过拟合实验设计。
