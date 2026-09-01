# 16 · Subgoal 词汇表 v1

> Subgoal 是慢层意图的短期执行目标，不是第二套需要人工逐帧标注的意图体系。
> v1 人工只标 `intents.md` 的 8 个顶层 intent；subgoal 由构建器根据状态、VG 检测和
> 输入事件半自动派生。

## 1. 与顶层 intent 对齐

顶层 intent 固定为：

```text
decipher / kite / rescue / rotate / travel / search / gate / idle
```

合法 subgoal 和候选范围定义在 `idv_agent/configs/subgoal.py` 的
`INTENT_SUBGOALS` 中。一个 subgoal 只能属于其 intent 的候选集合；未知或证据不足时
使用 `none` 或 `recover_target`，不能写自由文本。

## 2. v1 词汇

| subgoal | 主要 intent | 触发语义 |
|---|---|---|
| `none` | idle/任意 | 没有足够证据，安全等待 |
| `observe` | idle | 观察画面、等待事件或慢层刷新 |
| `search_area` | search/rescue/gate | 在当前区域寻找目标 |
| `find_cipher` | decipher/search | 尚未锁定密码机，寻找可用目标 |
| `approach_cipher` | decipher | 已找到密码机，向其靠近 |
| `face_cipher` | decipher | 调整视角/位置使密码机和交互方向对齐 |
| `start_decoding` | decipher | 出现交互提示，准备按 Q |
| `maintain_decoding` | decipher | 已进入破译状态，保持交互 |
| `handle_qte` | decipher | 破译中的校准/特殊 QTE 窗口 |
| `recover_target` | search/travel/decipher | 目标暂时丢失，重新观察或搜索 |
| `choose_destination` | rotate | 选择下一个安全点或密码机 |
| `move_to_target` | travel/rotate/rescue/gate | 向慢层指定目标移动 |
| `avoid_obstacle` | travel/rotate/kite | 进行短时绕行、脱困或避障 |
| `locate_safe_point` | kite | 寻找板窗/安全区域 |
| `maintain_distance` | kite | 保持与监管者的距离并调整路线 |
| `rescue_teammate` | rescue | 执行救援交互 |
| `open_gate` | gate | 到达出口并执行开门交互 |
| `escape` | kite/rescue/gate | 脱离危险区域或通过出口 |

## 3. 半自动派生规则

规则按以下优先级执行，输出一个 subgoal：

1. **事件/HUD 优先**：`decoding + qte` → `handle_qte`；交互提示出现且未破译 →
   `start_decoding`；已破译且无 QTE → `maintain_decoding`。
2. **VG 目标优先**：密码机已检测但未对齐 → `approach_cipher`；距离足够近但目标
   偏离视线中心 → `face_cipher`；没有可靠密码机 → `find_cipher` 或 `recover_target`。
3. **意图兜底**：`travel/rotate` → `move_to_target`，`kite` → `maintain_distance`，
   `rescue` → `rescue_teammate`，`gate` → `open_gate`，`idle` → `observe`。
4. 发生卡墙、连续无位移或目标方向突变时覆盖为 `avoid_obstacle`，恢复后回到原
   intent 的候选 subgoal。

派生器必须保留：`subgoal_source`（`rule`/`human_override`）、触发证据和规则版本。
人工可以只修改顶层 intent；只有规则明显错误时才允许填写 `human_override`，用于
后续评估规则质量，不把例外扩张成新的词汇。

## 4. 数据字段

```json
"slow_label": {
  "valid": true,
  "segment_id": "seg_003",
  "intent": "decipher",
  "subgoal": "face_cipher",
  "subgoal_source": "rule",
  "subgoal_rule_version": "subgoal.v1"
}
```

`intent` 是人工监督目标；`subgoal` 是半自动监督目标。训练时可以对
`subgoal_source=rule` 使用较低权重，对 `human_override` 使用正常权重，避免规则噪声
压过人工意图。

