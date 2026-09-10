# 回放工具

回放工具读取一个 VLA 原始 session，将键盘、鼠标按键和 Raw Input 相对鼠标位移按原始时间间隔重放。默认是 dry-run；只有显式传入 `--send-input` 才会调用 `pydirectinput` 发送真实输入。

## 1. 基本用法

```powershell
python -m idv_agent.scripts.replay_mouse data/new_vla_raw_sessions/<session_id>
```

默认只解析并打印统计，不发送输入：

```text
[replay] events=128 keyboard=42 mouse_moves=86 duration=12.431s scale=1
[replay] dry-run：未发送输入；使用 --send-input 才会实际回放
```

真实回放：

```powershell
python -m idv_agent.scripts.replay_mouse data/new_vla_raw_sessions/<session_id> `
  --send-input --start-delay 5 --scale 1.0
```

真实发送只应在官方自定义剧本/训练营中使用。`--start-delay` 只是留出切回游戏窗口的时间，工具不会自动检查前台窗口。

## 2. 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `session` | 必填 | 原始 session 目录，必须包含 `events.csv` |
| `--send-input` | 关闭 | 实际发送输入；默认 dry-run |
| `--scale` | `1.0` | 鼠标位移缩放倍数，必须 `>= 0` |
| `--start-delay` | `5.0` | 真发送前等待秒数，必须 `>= 0` |
| `--mouse-deltas` | session 内文件 | 指定 Raw Input CSV，可传外部诊断文件 |

示例：

```powershell
# 只查看解析结果
python -m idv_agent.scripts.replay_mouse <session>

# 半速鼠标幅度；键盘时间间隔不变
python -m idv_agent.scripts.replay_mouse <session> --scale 0.5

# 使用外部鼠标诊断文件
python -m idv_agent.scripts.replay_mouse <session> --mouse-deltas tmp/mouse_deltas.csv
```

## 3. 依赖关系

```text
replay_mouse.py
├─ pandas                         读取 CSV
├─ agent/action_decoder.py        Command 数据结构
└─ agent/action_executor.py       dry-run 记录或 pydirectinput 发送

可选输入约定：
└─ capture/raw_input_mouse.py      MouseDelta 字段
```

真实发送还需要 `pydirectinput`。Raw Input 鼠标文件由 Windows `raw_input_mouse.py` 生成。

## 4. 输入文件

### `events.csv`

必需字段是 `timestamp_ns`、`kind`、`code`；采集工具还会写入 `value`。

```csv
timestamp_ns,kind,code,value
8123500000000,key_down,key:w,1.0
8123700000000,key_up,key:w,1.0
8123800000000,mouse_down,btn:left,1.0
8123900000000,mouse_up,btn:left,1.0
8124000000000,scroll,key:scroll,1.0
```

支持的事件转换：

- `key_down` → `press`
- `key_up` → `release`
- `mouse_down` → 鼠标 `press`
- `mouse_up` → 鼠标 `release`
- `scroll` 当前不转换为发送命令

连续重复的 `key_down` 会去重；没有匹配按下状态的 `key_up` 会忽略；`key:f9` 和 `key:f10` 永远不会回放。

代码示例：

```text
key:w       → 键盘 w
key:space   → 空格键
key:q       → 键盘 q
btn:left    → 鼠标左键
btn:right   → 鼠标右键
```

### `mouse_deltas.csv`

必需字段是 `timestamp_ns`、`dx`、`dy`。

```csv
timestamp_ns,dx,dy
8123512340000,4,-1
8123515670000,7,0
8123519000000,0,3
```

`dx`、`dy` 是相对位移增量，不是屏幕绝对坐标。零位移行不会产生回放命令；非零位移会生成 `mouse_move`，发送时乘以 `--scale` 并四舍五入为整数像素。

## 5. 时间规则

同一 session 的 `events.csv` 和 `mouse_deltas.csv` 使用同一个 `time.perf_counter_ns` 时钟。工具将两类输入合并、按时间排序，并以最早事件为零点计算相邻事件的 `delay_s`。

如果 `--mouse-deltas` 指向 session 外部文件，该文件会以自身第一条鼠标记录为零点单独归一化，再与 session 键盘事件合并。因此外部文件适合测试鼠标幅度，不代表与该 session 严格同步。

相同时间戳的事件延迟为 `0` 秒并依次发送；原始 CSV 不会被修改。

## 6. 执行流程

```text
读取 events.csv
    ↓
去重 key_down、配对有效 key_up、排除 F9/F10
    ↓
读取并缩放 mouse_deltas.csv
    ↓
合并、排序、转换为相邻 delay_s
    ↓
dry-run 打印统计，或逐条执行 Command
    ↓
finally 调用 ActionExecutor.shutdown() 释放仍 held 的键
```

统计示例：

```text
events=128 keyboard=42 mouse_moves=86 duration=12.431s scale=1
```

- `events`：解析后的事件总数。
- `keyboard`：`key:` 开头的按下/释放事件数。
- `mouse_moves`：非零 Raw Input 鼠标移动数。
- `duration`：相邻事件延迟之和。
- `scale`：鼠标位移缩放倍数。

## 7. 推荐顺序

先校验 session：

```powershell
python -m idv_agent.scripts.validate_vla_raw data/new_vla_raw_sessions/<session_id>
```

再 dry-run：

```powershell
python -m idv_agent.scripts.replay_mouse data/new_vla_raw_sessions/<session_id>
```

确认数量、持续时间和鼠标幅度后，再进行真实回放：

```powershell
python -m idv_agent.scripts.replay_mouse data/new_vla_raw_sessions/<session_id> `
  --send-input --start-delay 5
```

## 8. 与视觉回放的区别

该工具回放原始输入事件，不播放 JPEG、不驱动 VLA 推理。查看画面和动作标签应使用 session 中的 `frames/`、`frame_timestamps.csv`、`per_frame_actions.csv` 或数据工作台。
