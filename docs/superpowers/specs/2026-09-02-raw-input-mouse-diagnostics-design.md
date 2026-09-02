# Raw Input 鼠标诊断与相机动作采集设计

## 背景

《第五人格》会将系统鼠标光标锁定或重置到窗口中心。当前 `InputRecorder` 通过轮询
`pynput.mouse.Controller().position` 记录绝对桌面坐标，无法可靠表示游戏实际消费的
鼠标相对位移，导致 `cam_dx/cam_dy` 标签失真。

## 目标

1. 在不读取游戏内存、不注入 DLL、不做截帧 hook 的前提下，诊断 Windows Raw Input
   是否能收到真实鼠标相对位移。
2. 新增独立诊断 CLI，不改变现有正式录制脚本的默认行为。
3. 诊断结果写入 `mouse_deltas.csv`，格式为
   `timestamp_ns,dx,dy`，并输出采样率、非零率、累计位移和方向统计。

## 非目标

- 正式录制和提取统一使用 `mouse_deltas.csv`；不再采集绝对鼠标坐标。
- 不保证 Raw Input 在 NeAC 或特定全屏渲染模式下必然可用；必须通过训练营实测确认。
- 不在真人排位/匹配中运行。

## 方案

新增 `idv_agent.scripts.diagnose_raw_mouse`：

- 使用用户态 Windows Raw Input API 注册鼠标设备并读取 `WM_INPUT` 相对位移。
- 默认运行指定秒数，输出诊断统计；可选 `--output` 写入 CSV。
- 支持 `--window-title` 作为前台窗口提示/校验信息，但不读取游戏进程内部数据。
- 采集失败时给出管理员权限、交互式桌面和窗口焦点提示，并以非零状态退出。

后续正式接入（本阶段不实现）：

1. `InputRecorder` 增加 `raw-input` 后端和 `mouse_input_source` 元数据。
2. 帧动作提取按帧时间窗口累加 raw `dx/dy`。
3. 回放脚本优先使用 `mouse_deltas.csv`，旧绝对坐标仅作 dry-run 分析。
4. 旧 session 标记 `camera_motion_valid=false`，避免错误相机监督进入训练。

## 验收标准

- 在无游戏环境下，诊断模块可被导入、参数校验和 CSV 解析测试通过。
- 在训练营中移动镜头 10 秒，输出文件包含连续时间戳和明显非零 `dx/dy`；若非零率接近
  0，则报告 Raw Input 在当前环境不可用，不切换正式采集后端。
- 现有录制、提取、训练测试保持不受影响。

## 安全与合规

仅允许官方自定义剧本/训练营；诊断默认只记录输入，不发送任何输入。Raw Input 仅为
用户态系统输入 API，不读取内存、不注入模块、不拦截游戏画面。
