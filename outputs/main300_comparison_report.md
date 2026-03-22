# Main300 实验结果对比

## 实验设置

- 数据：`outputs/egogazevqa_manifest_main300.jsonl`
- 样本数：300
- 采样方式：按 `causal / spatial / temporal` 平衡采样，各 100 条
- 模型：`qwen3-vl-flash`
- 对比组：
  - `fast proactive_gaze`
  - `fast proactive_baseline`

对应输出目录：

- [gaze_fast_main300](/C:/Users/Administrator/Desktop/SOTA/outputs/egogazevqa_qwen_proactive_gaze_fast_main300)
- [baseline_fast_main300](/C:/Users/Administrator/Desktop/SOTA/outputs/egogazevqa_qwen_proactive_baseline_fast_main300)

## 总体结果

| 指标 | proactive_gaze_fast | proactive_baseline_fast | 结论 |
| --- | ---: | ---: | --- |
| accuracy | 0.477 | 0.497 | baseline 略优 |
| intent_accuracy | 0.477 | 0.497 | baseline 略优 |
| strict_intent_accuracy | 0.050 | 0.020 | gaze 更优 |
| completed_rate | 0.940 | 0.950 | baseline 略优 |
| parsed_answer_rate | 0.940 | 0.950 | baseline 略优 |
| raw_has_box_rate | 0.940 | 0.947 | baseline 略优 |
| has_box_rate | 0.940 | 0.947 | baseline 略优 |
| raw_point_hit_rate | 0.087 | 0.117 | baseline 更优 |
| point_hit_rate | 0.940 | 0.947 | baseline 略优 |
| raw_mean_proxy_iou | 0.0305 | 0.0355 | baseline 更优 |
| mean_proxy_iou | 0.3616 | 0.3405 | gaze 更优 |
| raw_proxy_iou_ge_0_3 | 0.020 | 0.0267 | baseline 更优 |
| proxy_iou_ge_0_3 | 0.493 | 0.460 | gaze 更优 |
| raw_proxy_iou_ge_0_5 | 0.0033 | 0.0067 | baseline 更优 |
| proxy_iou_ge_0_5 | 0.220 | 0.203 | gaze 更优 |
| refinement_applied_rate | 0.853 | 0.830 | gaze 更依赖 refinement |
| mean_latency_s | 13.13 | 11.23 | baseline 更快 |

## 按问题类型拆分

| QA Type | 指标 | proactive_gaze_fast | proactive_baseline_fast | 结论 |
| --- | --- | ---: | ---: | --- |
| causal | accuracy | 0.710 | 0.720 | baseline 略优 |
| causal | intent_accuracy | 0.710 | 0.720 | baseline 略优 |
| causal | raw_mean_proxy_iou | 0.0296 | 0.0281 | gaze 略优 |
| causal | mean_proxy_iou | 0.3824 | 0.3625 | gaze 更优 |
| causal | raw_point_hit_rate | 0.090 | 0.100 | baseline 略优 |
| causal | point_hit_rate | 0.970 | 0.910 | gaze 更优 |
| spatial | accuracy | 0.410 | 0.460 | baseline 更优 |
| spatial | intent_accuracy | 0.410 | 0.460 | baseline 更优 |
| spatial | raw_mean_proxy_iou | 0.0263 | 0.0350 | baseline 更优 |
| spatial | mean_proxy_iou | 0.3482 | 0.3273 | gaze 更优 |
| spatial | raw_point_hit_rate | 0.090 | 0.100 | baseline 略优 |
| spatial | point_hit_rate | 0.950 | 0.940 | 接近，gaze 略优 |
| temporal | accuracy | 0.310 | 0.310 | 持平 |
| temporal | intent_accuracy | 0.310 | 0.310 | 持平 |
| temporal | raw_mean_proxy_iou | 0.0359 | 0.0429 | baseline 更优 |
| temporal | mean_proxy_iou | 0.3534 | 0.3329 | gaze 更优 |
| temporal | raw_point_hit_rate | 0.080 | 0.150 | baseline 更优 |
| temporal | point_hit_rate | 0.900 | 0.990 | baseline 更优 |

## 结果解释

### 1. 语义层面

- 在这组 300 条正式主实验里，`proactive_gaze_fast` 并没有带来整体答案准确率提升。
- `baseline_fast` 的 `accuracy` 和 `intent_accuracy` 都略高。
- `gaze` 组只在 `strict_intent_accuracy` 上更高，说明 gaze 提供的信号有助于让意图表达更聚焦，但还没有稳定转化为更好的选项预测。

### 2. 定位层面

- 如果看模型原始框能力，`baseline` 反而更强：
  - `raw_point_hit_rate` 更高
  - `raw_mean_proxy_iou` 更高
- 如果看最终 refined 框，`gaze` 更强：
  - `mean_proxy_iou` 更高
  - `proxy_iou_ge_0_3 / 0.5` 更高

这说明：

- 模型原始框能力并未因加入 gaze 明显变强
- `gaze` 的主要价值体现在 refinement 阶段，而不是原生框输出阶段

### 3. 计算稳定性

- `gaze` 组错误 18 条
- `baseline` 组错误 15 条
- 两组主错误都来自 DashScope 接口的 SSL/连接异常，而不是本地代码崩溃

## 可直接写进报告的结论

在 300 条平衡样本的正式主实验中，`fast proactive_gaze` 相比 `fast proactive_baseline` 没有提升整体问答准确率，`accuracy` 反而从 `0.497` 下降到 `0.477`。但在最终定位质量上，`proactive_gaze` 更优，`mean_proxy_iou` 从 `0.3405` 提升到 `0.3616`，`proxy_iou_ge_0_3` 从 `0.460` 提升到 `0.493`。进一步结合 `raw_mean_proxy_iou` 仅为 `0.0305` 可以看出，当前性能提升主要来自 gaze-guided refinement，而非模型原始框能力本身。

## 后续建议

1. 报告中必须同时写 `raw` 和 `refined` 两套指标，避免把 refinement 增益误写成模型原生能力。
2. 如果目标是提升问答准确率，下一步应该优化 `gaze -> answer` 的语义利用，而不是继续单独强化 box prompt。
3. 如果目标是提升系统级目标定位效果，当前 `gaze + refinement` 路线已经成立，可以继续做更大规模实验。
