# Codex 全链路验证提示词

## 任务目标
验证 ID_V_Agent 从进入对局、找到密码机、到进入破译状态的完整代码链路，确保训练后的模型能在真实游戏中执行。

---

## 环境准备

### 1. 启动游戏
- 进入《第五人格》自定义剧本训练模式（单人模式，无监管者）
- 选择求生者角色，准备开始对局
- 保持游戏窗口标题为 "第五人格"（或记下实际标题）

### 2. 确认模型和资源
- 训练好的 VLA checkpoint：`checkpoints/M3_ACT/act_v2_500.pt` 或你指定的最新checkpoint
- M2_VG 初始化 checkpoint：`checkpoints/M2_VG/`
- Qwen3-VL 基础模型：`~/.cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/`
- 密码机检测资源（二选一或同时使用）：
  - YOLO 模型：`models/cipher_yolo_best.pt`（如果有）
  - 模板图片：`assets/cipher_template.jpg`（如果有）

---

## 执行步骤

### Step 1: 规则基线验证（确保环境正常）

```bash
# 在管理员权限的终端中运行
python -m idv_agent.scripts.run_agent \
  --mode rule \
  --title "第五人格" \
  --duration 60 \
  --fps 30 \
  --cipher-template assets/cipher_template.jpg \
  --cipher-detect-interval 3.0 \
  --cam-pixel-scale 120.0 \
  --send-input
```

**预期行为**：
1. Agent 自动截屏捕获游戏画面
2. 每 3 秒检测一次密码机位置
3. 检测到密码机后，朝密码机方向移动
4. 靠近密码机时，按 Q 键进入破译状态
5. 终端打印密码机检测日志和延迟统计

**验证点**：
- [ ] 能否正确捕获游戏画面（无黑屏）
- [ ] 密码机检测是否触发（查看日志 `[cipher] detections=...`）
- [ ] 角色是否朝密码机移动
- [ ] 是否成功按下 Q 键并进入破译状态

**如果失败**：
- 检查游戏窗口标题是否匹配 `--title` 参数
- 确认以管理员权限运行
- 检查密码机模板图片路径是否正确

---

### Step 2: 加载训练模型（代码链路验证）

**注意**：当前 VLA 模型接入尚在开发中，`run_agent.py` 只支持 `--mode rule`。以下是预期的接入方式（需要先实现 VLA Policy 适配器）。

#### 2.1 创建 VLA Policy 适配器

创建 `idv_agent/model/vla_policy.py`：

```python
"""VLA 动作块策略适配器"""
from __future__ import annotations
import torch
from pathlib import Path
from idv_agent.model.policy import Policy, PolicyOutput
from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter
from idv_agent.scripts.train_vla import _load_act_backbone

class VLAPolicy(Policy):
    """VLA 快慢双系统策略（action chunk 模式）"""
    
    def __init__(self, checkpoint_path: str, model_path: str, 
                 init_checkpoint: str, device: torch.device):
        self.device = device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        
        # 加载骨干网络和核心
        self.adapter, self.processor, _ = _load_act_backbone(
            model_path, init_checkpoint, dtype=dtype, device=device
        )
        self.core = SharedFastSlowVLA(
            self.adapter.hidden_size, temporal_dim=256
        ).to(device=device, dtype=torch.float32)
        
        # 加载训练好的权重
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.adapter.visual_projection.load_state_dict(saved["adapter"]["visual_projection"])
        self.adapter.condition_projection.load_state_dict(saved["adapter"]["condition_projection"])
        self.core.load_state_dict(saved["core"])
        
        self.adapter.eval()
        self.core.eval()
        
        # 维护历史帧和慢状态
        from collections import deque
        self.frame_history = deque(maxlen=8)
        self.condition = None
        self.action_buffer = deque()  # chunk 缓冲
        
    def decide(self, frame, state) -> PolicyOutput:
        """
        frame: BGR numpy array (H, W, 3)
        state: dict with keys: intent_vector, skeleton_idx, spatial, memory
        """
        import cv2
        from PIL import Image
        
        # BGR -> RGB -> PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # 更新帧历史
        self.frame_history.append(pil_image)
        
        # 如果 action buffer 非空，直接返回缓存的动作
        if self.action_buffer:
            return self.action_buffer.popleft()
        
        # 否则运行模型生成新的 chunk
        with torch.no_grad():
            # 编码当前窗口（需要至少 3 帧）
            if len(self.frame_history) < 3:
                # 还没积累足够历史，返回 NOOP
                return PolicyOutput(category_id=0, continuous=(0.0, 0.0, 0.0, 0.0))
            
            # TODO: 实现完整的编码和推理逻辑
            # 1. 将 frame_history 编码成特征
            # 2. 运行慢头更新 condition（如果需要）
            # 3. 运行快头生成 chunk
            # 4. 解码 chunk 到 PolicyOutput 并填充 action_buffer
            
            # 占位返回
            return PolicyOutput(category_id=0, continuous=(0.0, 0.0, 0.0, 0.0))
```

#### 2.2 修改 `run_agent.py` 支持 VLA 模式

在 `build_policy()` 函数中添加：

```python
def build_policy(args, device):
    if args.mode == "rule":
        from idv_agent.model.policy import RulePolicy
        rp = RulePolicy(RuleAgent(visual_servo=CipherVisualServo()))
        return rp, None
    elif args.mode == "vla":
        from idv_agent.model.vla_policy import VLAPolicy
        if not args.vla_checkpoint:
            raise ValueError("--mode vla 需要指定 --vla-checkpoint")
        policy = VLAPolicy(
            checkpoint_path=args.vla_checkpoint,
            model_path=args.model_path,
            init_checkpoint=args.init_checkpoint,
            device=device,
        )
        return policy, None
    raise ValueError(f"未知 mode: {args.mode}")
```

添加参数：

```python
p.add_argument("--mode", choices=["rule", "vla"], default="rule")
p.add_argument("--vla-checkpoint", type=Path, default=None)
p.add_argument("--model-path", type=Path, default=None)
p.add_argument("--init-checkpoint", type=Path, default=None)
```

---

### Step 3: VLA 模型全链路测试

```bash
# 先离线测试图片（验证模型加载）
python -m idv_agent.scripts.run_agent \
  --mode vla \
  --vla-checkpoint checkpoints/M3_ACT/act_v2_500.pt \
  --model-path ~/.cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master \
  --init-checkpoint checkpoints/M2_VG \
  --test-dir data/test_frames/ \
  --test-stride 5 \
  --device cuda
```

**预期输出**：
- 每张图片输出：`[test:N] file=... action=... continuous=... commands=...`
- 无模型加载错误或 CUDA OOM

**验证点**：
- [ ] 模型成功加载（无权重不匹配错误）
- [ ] 推理不崩溃
- [ ] 输出的 action 包含非 stop 的移动（如果测试图片有密码机）

---

### Step 4: 在线全链路测试

```bash
# 进入对局后运行（管理员权限）
python -m idv_agent.scripts.run_agent \
  --mode vla \
  --vla-checkpoint checkpoints/M3_ACT/act_v2_500.pt \
  --model-path ~/.cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master \
  --init-checkpoint checkpoints/M2_VG \
  --title "第五人格" \
  --duration 120 \
  --fps 30 \
  --cipher-template assets/cipher_template.jpg \
  --cipher-detect-interval 3.0 \
  --cam-pixel-scale 120.0 \
  --trajectory-log logs/vla_test_trajectory.jsonl \
  --trajectory-frames logs/vla_test_frames/ \
  --send-input \
  --device cuda
```

**预期行为**：
1. Agent 启动，开始实时推理
2. 移动头输出非 stop 的方向（验证移动塌缩是否修复）
3. 检测到密码机后，朝密码机移动
4. 靠近后按 Q 键进入破译状态
5. 轨迹日志记录每帧的观察和动作

**关键验证点**：
- [ ] 模型推理延迟 < 100ms（查看 `[latency]` 日志）
- [ ] 移动头预测分布：非 stop 占比 > 30%（事后分析轨迹日志）
- [ ] 意图头预测：出现 decipher 意图（在接近密码机时）
- [ ] 成功按下 Q 键并进入破译状态
- [ ] 无按键卡死（F12 可以正常退出）

---

## 事后分析

### 1. 分析轨迹日志

```python
import json
from collections import Counter

move_preds = []
intent_preds = []

with open("logs/vla_test_trajectory.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        # 根据实际 trajectory 格式提取
        # move_preds.append(...)
        # intent_preds.append(...)

move_counter = Counter(move_preds)
print(f"移动分布: {move_counter}")
print(f"非 stop 占比: {(len(move_preds) - move_counter.get(0, 0)) / len(move_preds):.1%}")
print(f"意图分布: {Counter(intent_preds)}")
```

### 2. 检查代码路径

- [ ] `RealtimeAgent.run()` → 主循环正常运行
- [ ] `Policy.decide()` → VLA 模型推理成功
- [ ] `ActionDecoder.decode()` → chunk 解码正确
- [ ] `ActionExecutor.execute()` → 键鼠命令发送成功
- [ ] 密码机检测 → 感知模块正常

### 3. 性能指标

- **推理延迟**: 目标 < 100ms（30 FPS 预算 33ms，留余量给编码）
- **帧率**: 实际运行 FPS ≥ 25
- **移动塌缩**: 非 stop 预测 > 30%
- **意图准确率**: 在密码机附近预测出 decipher

---

## 故障排查

### 问题 1: 模型加载失败
**症状**: `KeyError` 或权重不匹配
**排查**:
```bash
# 检查 checkpoint 内容
python -c "
import torch
ckpt = torch.load('checkpoints/M3_ACT/act_v2_500.pt', map_location='cpu')
print(ckpt.keys())
"
```
**解决**: 确认 checkpoint 包含 `adapter`, `core` 字典

### 问题 2: 移动头仍然全预测 stop
**症状**: 角色不动或只原地转视角
**排查**:
1. 检查训练是否启用了 `--move-direction-balance`
2. 运行诊断脚本确认模型状态：
```bash
python idv_agent/scripts/diagnose_act.py \
  --checkpoint checkpoints/M3_ACT/act_v2_500.pt \
  --data <your_data> --val-data <your_val_data> \
  --model-path <model_path> --init-checkpoint <init_ckpt> \
  --max-samples 256 --output reports/diagnose_online.json
```
**解决**: 如果确认塌缩，回到训练阶段用修复后的配置重新训练

### 问题 3: 推理延迟过高（> 200ms）
**症状**: 帧率低于 15 FPS
**排查**:
- 检查是否在 CPU 上运行（应该用 CUDA）
- 检查是否有重复编码（frame cache 应启用）
**解决**: 
- 改用 `--device cuda`
- 优化 VLAPolicy 实现，缓存任务编码

### 问题 4: 按键不响应
**症状**: 终端显示命令但游戏无反应
**排查**:
- 确认以管理员权限运行
- 检查 `pydirectinput` 是否安装
- 测试 `executor.dry_run` 是否为 False
**解决**:
```bash
pip install pydirectinput
# 以管理员权限重新运行
```

### 问题 5: 找不到密码机
**症状**: `[cipher] detections=0`
**排查**:
- 密码机模板图片分辨率是否匹配
- YOLO 模型路径是否正确
- 地图光照是否差异过大
**解决**:
- 采集当前分辨率和地图的密码机截图作为模板
- 或使用 YOLO 模型（泛化性更好）

---

## 成功标准

**最低标准**（代码链路正确）：
- [x] 模型成功加载，无权重错误
- [x] 推理不崩溃，延迟 < 200ms
- [x] 键鼠命令正确发送到游戏
- [x] 密码机检测触发（规则或 VLA 均可）
- [x] 角色能移动（非全 stop）

**理想标准**（模型有效）：
- [x] 非 stop 预测 > 30%
- [x] 能主动朝密码机移动
- [x] 在密码机附近预测出 decipher 意图
- [x] 成功按下 Q 键并进入破译状态
- [x] 推理延迟 < 100ms，帧率 ≥ 25 FPS

---

## 下一步

如果全链路验证通过：
1. 收集 VLA 在线轨迹作为新的训练数据（DAgger）
2. 分析失败案例（未找到密码机、撞墙等）
3. 针对性补充训练数据
4. 迭代训练，提升成功率

如果验证失败：
1. 先用规则基线确认环境和感知模块正常
2. 离线测试 VLA 模型加载和推理
3. 回到训练阶段，用修复后的配置重新训练
4. 重复 Step 3-4
