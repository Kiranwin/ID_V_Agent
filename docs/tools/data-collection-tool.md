# 数据采集工具

本文介绍当前 VLA 原始数据采集工具的使用方式、脚本依赖和输出字段。采集工具只记录屏幕、键鼠事件和 Raw Input 鼠标相对位移，不向游戏注入输入。

> **当前 v6 流程更正（2026-09-08）**：以下旧提取/构建章节描述 v3/v4/v5 工具，
> 不是新 20 FPS MVP 样本的前置步骤。新流程权威入口是 [03 §0](../03-数据格式.md)。
> 原始校验仍保留；`prepare_mvp_v6 prepare` 已在内部调用它，并直接从原始时间戳、
> events.csv、mouse_deltas.csv 生成 200 ms 动作建议，无需 per_frame_actions.csv。
> X-AnyLabeling 完成后导入，再填写 navigation_review_with_state.csv，审核通过后导出
> v6。当前 train_vla/run_agent 尚未适配此新合同，不要再构建旧 v5 冒充兼容数据。

## 1. 工具入口

### 1.1 录制原始 session

```powershell
python -m idv_agent.scripts.record_vla `
  --window-title "第五人格" `
  --fps 20 `
  --output data/raw_sessions `
  --task-name find_cipher_and_decode `
  --task-instruction "找到密码机，靠近并进入破译" `
  --mode standard `
  --post-stop-seconds 0
```

录制流程：

1. 启动命令后，工具等待 F9。
2. 按 F9 开始一个录制段。
3. 在官方自定义剧本/训练营中完成示范。
4. 再按 F9 后继续记录默认 2 秒结果画面，再结束当前录制段。
5. 按 F10 退出程序；录制中的 F10 同时结束当前录制段。

`--max-seconds` 和 `--max-frames` 可用于短测试：

```powershell
python -m idv_agent.scripts.record_vla --max-seconds 20 --max-frames 600
```

默认参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--window-title` | `第五人格` | 模糊匹配目标窗口标题 |
| `--fps` | `20` | 目标采集帧率，不等于 Qwen 编码频率 |
| `--output` | `data/raw_sessions` | session 根目录 |
| `--post-stop-seconds` | `2` | F9 后继续采集时间；Q 专项不要求尾部结果 |
| `--trim-idle-boundaries` | 关闭 | 历史兼容；当前默认保留头尾无输入画面 |
| `--max-seconds` | `0` | 大于 0 时限制单段时长 |
| `--max-frames` | `0` | 大于 0 时限制单段帧数 |
| `--max-width` | `1334` | 超过此宽度时按比例缩小 |
| `--jpeg-quality` | `90` | JPEG 质量，范围 1..100 |
| `--task-name` | `find_cipher_and_decode` | 写入任务名 |
| `--task-instruction` | `找到密码机，靠近并进入破译` | 写入任务指令 |
| `--mode` | `standard` | 游戏模式条件 |
| `--note` | 训练营说明 | 写入备注 |


## 2. 脚本依赖关系

```text
record_vla.py
├─ capture/screen_capture.py       dxcam 屏幕帧
├─ capture/input_logger.py         pynput 键鼠事件
├─ capture/raw_input_mouse.py      Windows Raw Input 相对位移
└─ vla/action_chunk.py              schema 和动作常量

extract.py
├─ labels/extract.py                事件 → 逐帧动作
└─ labels/state_machine.py          按键区间和鼠标位移回放

build_vla_chunks.py
├─ extract_session()
├─ labels/state_machine.py
└─ vla/action_chunk.py              v3/v4/v5 记录校验
```

运行环境通常为 `idv312` conda 环境。采集需要 Windows、`dxcam`、`pynput`、`pygetwindow`、OpenCV；Raw Input 只支持 Windows。真实采集建议用管理员终端运行，以便捕获游戏内键盘事件。

## 3. session 目录结构

```text
<session_id>/
├─ frames/
│  ├─ 00000000.jpg
│  ├─ 00000001.jpg
│  └─ ...
├─ events.csv
├─ mouse_deltas.csv
├─ frame_timestamps.csv
├─ meta.json

```

F9 结束录制时，工具会清理首尾连续无输入帧；中间静止帧保留。没有任何有效键鼠输入的空 session 会被删除。保留帧会重新从 `00000000.jpg` 编号。

## 4. 原始输出字段

### 4.1 `meta.json`

```json
{
  "recording_type": "vla_raw",
  "vla_schema_version": "vla.action_chunk.v3",
  "session_id": "20260908_201530_123456",
  "source": "human",
  "mode": "standard",
  "task_name": "find_cipher_and_decode",
  "task_instruction": "找到密码机，靠近并进入破译",
  "target_fps": 30,
  "num_frames": 240,
  "effective_fps": 29.8,
  "start_ts_ns": 8123456789012,
  "end_ts_ns": 8131506789012,
  "window_title": "第五人格",
  "action_delay_frames": 1,
  "frame_timestamp_clock": "time.perf_counter_ns",
  "input_timestamp_clock": "time.perf_counter_ns",
  "max_width": 1334,
  "jpeg_quality": 90,
  "note": "只在官方自定义剧本/训练营录制",
  "mouse_input_source": "raw_input",
  "camera_motion_valid": true,
  "idle_boundary_trimmed": true,
  "had_input": true
}
```

字段说明：

- `recording_type`：固定为 `vla_raw`。
- `vla_schema_version`：当前原始采集版本。
- `mode`：游戏模式，例如 `standard`。
- `target_fps`：配置的目标帧率。
- `effective_fps`：按 session 时间范围计算的实际帧率。
- `num_frames`：保留的 JPEG 帧数。
- `start_ts_ns` / `end_ts_ns`：统一使用 `time.perf_counter_ns` 的 session 时间范围。
- `mouse_input_source`：当前固定为 `raw_input`。
- `idle_boundary_trimmed`：是否执行首尾无输入清理。
- `had_input`：是否检测到有效输入。

### 4.2 `frame_timestamps.csv`

```csv
frame_id,timestamp_ns
0,8123456789012
1,8123490122345
2,8123523455678
```

- `frame_id`：从 0 连续递增。
- `timestamp_ns`：该帧抓取时间。

### 4.3 `events.csv`

```csv
timestamp_ns,kind,code,value
8123500000000,key_down,key:w,1.0
8123700000000,key_up,key:w,1.0
8123800000000,mouse_down,btn:left,1.0
8123900000000,mouse_up,btn:left,1.0
8124000000000,scroll,key:scroll,1.0
```

`kind` 取值：

- `key_down` / `key_up`：键盘按下/抬起。
- `mouse_down` / `mouse_up`：鼠标按键按下/抬起。
- `scroll`：滚轮事件。

`code` 示例：

- `key:w`、`key:a`、`key:space`、`key:q`
- `btn:left`、`btn:right`、`btn:middle`
- `key:scroll`

F9/F10 控制键不会进入该文件。

### 4.4 `mouse_deltas.csv`

```csv
timestamp_ns,dx,dy
8123512340000,4,-1
8123515670000,7,0
8123519000000,0,3
```

- `timestamp_ns`：Raw Input 报告时间。
- `dx` / `dy`：相对鼠标位移，整数像素增量；不是屏幕绝对坐标。
- `dx=0,dy=0` 的报告可以存在，但不会被视为有效移动输入。




