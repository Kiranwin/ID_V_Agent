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
使用 `none`，不能写自由文本。

## 2. v1 词汇

| subgoal | 主要 intent | 触发语义 |
|---|---|---|
| `none` | idle/任意 | 没有足够证据，安全等待 |
| `approach_cipher` | decipher | 已找到密码机，向其靠近 |
| `start_decoding` | decipher | 出现交互提示，准备按 Q |
| `handle_qte` | decipher | 破译中的校准/特殊 QTE 窗口 |
| `locate_safe_point` | kite | 寻找板窗/安全区域 |
| `maintain_distance` | kite | 保持与监管者的距离并调整路线 |
| `vault_window` | kite | 执行翻窗 |
| `drop_pallet` | kite | 执行放板 |
| `disengage` | kite/rotate | 脱离当前危险路线 |
| `approach_chair` | rescue | 接近椅子或救援位置 |
| `search_area` | rescue | 在当前区域寻找救援目标 |
| `rescue_teammate` | rescue | 执行救援交互 |
| `heal_teammate` | rescue | 治疗队友 |
| `wait_opportunity` | rescue | 等待安全救援窗口 |
| `move_to_zone` | rotate | 移动到下一个区域 |
| `find_cipher` | travel | 扫视寻找密码机 |
| `move_to_target` | travel | 向目标移动 |
| `avoid_obstacle` | travel | 绕行或避障 |
| `find_chest` | search | 寻找箱子 |
| `open_chest` | search | 打开箱子 |
| `pickup_item` | search | 拾取道具 |
| `move_to_gate` | gate | 移动到大门 |
| `open_gate` | gate | 执行开门交互 |
| `escape` | gate | 通过出口逃脱 |
| `observe` | idle | 观察画面、等待事件或慢层刷新 |
| `hide` | idle | 躲藏或保持隐蔽 |

## 3. 半自动派生规则

规则按以下优先级执行，输出一个 subgoal：

1. **decipher**：QTE/校准 → `handle_qte`；交互提示或交互动作 → `start_decoding`；
   其余情况 → `approach_cipher`。
2. **travel**：移动信号优先 → `move_to_target`；无移动但相机有变化 → `find_cipher`；
   两者都无时安全回退到 `move_to_target`。
3. **其他 intent**：`kite` 默认 `maintain_distance`，`rescue` 默认
   `rescue_teammate`，`rotate` 默认 `move_to_zone`，`search` 默认 `find_chest`，
   `gate` 默认 `open_gate`，`idle` 默认 `observe`；动作类别命中专用 subgoal 时覆盖默认值。

派生器必须保留：`subgoal_source`（`rule`/`human_override`）、触发证据和规则版本。
人工可以只修改顶层 intent；只有规则明显错误时才允许填写 `human_override`，用于
后续评估规则质量，不把例外扩张成新的词汇。

## 4. 数据字段

```json
"slow_label": {
  "valid": true,
  "segment_id": "seg_003",
  "intent": "decipher",
  "subgoal": "approach_cipher",
  "subgoal_source": "rule",
  "subgoal_rule_version": "subgoal.v1"
}
```

`intent` 是人工监督目标；`subgoal` 是半自动监督目标。训练时可以对
`subgoal_source=rule` 使用较低权重，对 `human_override` 使用正常权重，避免规则噪声
压过人工意图。
