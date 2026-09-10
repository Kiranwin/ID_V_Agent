# Data Workbench V6 标注工作台

更新：2026-09-09。工作台直接在网页中标注 M29/V6 的阶段、目标、路径、动作和当前破译状态。
**不用手动编辑 CSV**：网页保存到每个 session 自己的 `navigation_review_with_state.csv`。
工作台只支持 V6，原 actions/intents/chunks/camera-control 路由已移除。

## 启动与当前工程

在项目根目录执行：

```powershell
conda activate idv312
python -m idv_agent.scripts.data_workbench --root data/raw_sessions --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。当前四个 session 已有独立审核表：

| session | 原始帧 | 审核端点 | CSV 所在位置 |
|---|---:|---:|---|
| 20260908_205631_985816 | 81 | 21 | 此 session 目录内 |
| 20260908_210127_149560 | 86 | 25 | 此 session 目录内 |
| 20260908_210720_785351 | 100 | 27 | 此 session 目录内 |
| 20260908_210912_841524 | 99 | 28 | 此 session 目录内 |

例如 `data/raw_sessions/20260908_210127_149560/navigation_review_with_state.csv`。
四个工程共 101 个端点；当前进度以网页顶部计数为准。视觉 Q 标签合计 48 正、318
负（366 帧）。当前合并 Q v2 文件为
`data/derived/prompt_q/20260909_xany_all4_v2/train.jsonl`。之前三条的 285 帧试验文件
保留作历史产物，不作为当前入口。

目录关系：

```text
data/raw_sessions/<session>/
  frames/ + events.csv + mouse_deltas.csv + frame_timestamps.csv + meta.json
  navigation_review_with_state.csv       ← 网页实际读写的唯一审核表
  review_history/                       ← 保存前快照、修改记录

data/annotations/mvp_v6/<session>_xany_import_v2/
  manifest.json + decisions.jsonl + visual_frames.jsonl
  source_annotations/                   ← X-AnyLabeling 原 JSON 快照
  navigation_review*.csv                ← 历史导入快照，不再直接编辑
  prompt_q_all_frames.jsonl             ← Q/NO_Q + 可用的 prompt bbox

data/annotations/mvp_v6/<session>_nav_review_<时间>/
  training_v6.jsonl                      ← 完整工程导出，新版本
```

图片、事件、时间戳和 X-AnyLabeling JSON 不修改。允许按用户要求在 session 中新增/更新
审核表；raw hash 只覆盖原始采集证据，审核表另行记录 revision/hash。

## 网页使用顺序

1. 左侧选择 session。工作台默认打开 `*_xany_import_v2` 工程；`v1` 只保留为旧快照。
2. 选择决策端点，左侧查看当前画面、历史缩略按钮与标注框。可逐帧/滑条查看，播放
   使用原始时间间隔，最多到本端点。当前状态标注不使用未来画面。
3. 右侧选择阶段、实际破译状态、目标、镜头方式和路径。点击图中的密码机框可选目标；
   多候选全部显示，工作台不会自行挑最大框。
4. 展开“查看演示动作建议”。prompt 可见端点表示已经足够接近，导航自动排除；其他端点
   才选择接受演示、纠正或排除。纠正时才显示方向与鼠标 counts。
5. 未确定的内容点击“保存草稿”；完成当前行点击“标注完成并下一条”。完整语义检查
   在这一步执行，失败会提示原因，不覆盖 CSV。Q 标签为只读，由交互提示生成。
6. “下一个未完成”跳过已完成行。刷新/重启后从 session CSV 恢复进度。页面顶部还会汇总
   四个工程的五阶段、decoding 正负、steering/path、有效导航和独立 val 缺口；这只是
   实时预检，最终仍以聚合审计和导出文件为准。
7. “检查工程”给出所有未完成/非法行；“完成工程并导出 V6”要求全部行完成，写新版本
   并显示导出位置。工程全部导航排除也可导出，但不表示具备完整导航训练覆盖。
8. “导出 Q 全帧样本”可随时导出已导入的全部视觉帧，不受导航完成率影响；不会只导出
   稀疏决策端点而丢掉人类反应前的 Q 正样本。

未保存修改会阻止切换工程/端点。若遇另一页面已保存的 revision 冲突，当前修改不会
覆盖服务器文件；点击“放弃未保存修改并重载”获取最新版本。不要同时开多个独立服务
写同一个 session；并发冲突保护针对同一服务的浏览器请求。

## 新 session 的准备与导入

在工作台展开“准备新工程 / 导入 X-AnyLabeling”：

- 点击“准备端点”：自动 raw 校验并直接按 events/mouse/timestamps 生成 200 ms 动作建议。
- 在 X-AnyLabeling 完成 `cipher_visible` / `cipher_highlight` / `interact_prompt` 三类矩形。
- 每张图保存同名 JSON。无目标也保存 `shapes: []`，缺文件不推断为负样本。
- 在网页填写标注人、用途 train/val/test、同场景分组，勾选已完成声明，然后“导入标注”。
- 打开导入工程开始网页标注。新 session 自动获得自己的审核 CSV，若已存在则保留其内容。

同场景/连续录制的 session 应使用同一个 scenario_group，不能分开伪造独立验证。
X-AnyLabeling 完成声明是用户审核来源，不会把自动检测 score/checked 标志当独立人工复核。

矩形 JSON 示例（软件通常生成 2 或 4 个角点，都支持）：

```json
{
  "imagePath": "00000075.jpg", "imageWidth": 1334, "imageHeight": 779,
  "shapes": [
    {"label": "cipher_visible", "shape_type": "rectangle", "points": [[700,300],[980,700]]},
    {"label": "interact_prompt", "shape_type": "rectangle", "points": [[820,410],[910,540]]}
  ]
}
```

## 字段与填写样例

| 网页字段 / CSV 字段 | 可选值或样例 | 用法 |
|---|---|---|
| 端点 `id` / `frame` | `session:00000075` / `75` | 自动生成，只读 |
| 图像 `image_path` | 原始 `frames/00000075.jpg` 路径 | 自动生成，只读 |
| 提示 `interact_prompt` / `q_target` | `1 / 1`，`0 / 0` | 有提示 Q、无提示 NO_Q；人类未按也仍为正样本 |
| prompt 与密码机候选关联 | Q 端点不要求候选；prompt 出现即视为已足够接近并自动排除导航 | prompt 框本身不是密码机框；候选/跟踪只标 prompt 出现前的搜索、靠近、对齐 |
| 候选 `candidate_summary` | `shape_0`, `shape_1` 的类别和 bbox | 源矩形，只读；与图中标识对应 |
| 阶段建议 `phase_proposal` | `interact` 或空 | 只读建议，最终 phase 由人选择 |
| 移动/相机建议 `move_proposal` / `camera_proposal` | `1` / `[25,0]` | 从事件积分生成，只读 |
| 限制 `eligibility_issues` | `insufficient_history` | 页面展示；不阻塞 Q 标注 |
| 范围 `scope` | `in_scope` / `out_of_scope` / `uncertain` | 本期 MVP、范围外、无法判断 |
| 阶段 `phase` | `search` / `approach` / `align` / `interact` / `maintain_decode` | 搜索、靠近、对齐、交互、维持破译；按 Q 后尚未确认的等待仍属于 interact，执行反馈单独记录 |
| 当前状态 `decoding` | `true` / `false` / `unknown` | 画面实际状态，不等于任务名，不由 Q 按键自动推断 |
| 证据 `decoding_evidence_frames` | `[70,75]` | 点“加入证据”自动填入；true 需要当前/过去证据；不能用未来帧 |
| 目标 `target_candidate` | `shape_0` / `none` / `unknown` / `occluded` | 当前帧目标；没有、不确定、已遮挡 |
| 目标 ID `target_track_id` | `cipher_1` | 仅在 prompt 出现前需要跨端点跟踪审计时填写；当前 M29 不把它喂成类别输入 |
| 镜头 `steering` | `target_center` / `path_follow` / `search_sweep` / `hold` | 对准目标、跟随路径、搜索、保持 |
| 路径 `path` | `direct` / `detour_left` / `detour_right` / `blocked` / `unknown` | 局部通行方式；无法判断就 unknown |
| 动作来源 `action_source` | `accept_replay` / `correction` / `exclude` | 接受自动建议、纠正动作、排除导航；Q 不受 exclude 影响 |
| 移动 `move` | 0 停止；1 W；2 W+D；3 D；4 S+D；5 S；6 S+A；7 A；8 W+A | 只在 correction 填写 |
| 镜头 `camera_dx/camera_dy` | `-110,-25,0,25,110` | 实际 Raw Input counts，不是桶 ID或图像像素；只在 correction 填写 |
| 原因 `reason` | `当前目标被遮挡，导航动作不确定` | 备注；correction 必须有理由 |
| 完成 `reviewed` | pending / reviewed | 保存草稿/标注完成按钮自动填写 |

示例：靠近密码机可以填 `in_scope + approach + shape_0 + cipher_1 + path_follow + direct
+ accept_replay + decoding=false`；不手填 move/camera，完成时自动取建议。
已有破译视觉证据时填 `maintain_decode + decoding=true + [当前证据帧]`。无法判断历史
或动作窗口不足时选 `exclude` 保留视觉/Q 监督，不能把未知强行写成“正在破译”。

Q、顶层状态、导航各有独立 mask：`interact_prompt=1` 时无论人类反应慢、刚按过 Q、
phase 是什么，都保留 Q；顶层另行学习 decoding 状态指导导航。允许连续新画面连续 Q。

边界端点采用分层监督，不需要放弃意图标注：

- `insufficient_history`、`visual_history_gap`：可以标 phase、decoding、目标和 Q；工作台
  自动锁定 `action_source=exclude`，因此只关闭 `masks.navigation`。
- `incomplete_action_window`：没有未来 200 ms replay，也仍可标当前 phase/状态/Q；导航
  自动排除，不再报 `no complete replay action window` 阻止完成。
- `insufficient_one_second_outcome_tail`：只表示无法确认动作后的 decoding outcome，不影响
  当前 phase、当前 decoding 事实或 Q；outcome 保持 unknown。

如果边界帧连 phase 本身也无法判断，phase 保持“未确定”即可完成；导出会令
`masks.phase=false`，不会把未知硬写成某个意图。目标无法判断时选择 `unknown`，当前
decoding 选择 `unknown`。这些未知项不作为负样本。

## 依赖的脚本子文件

| 文件 | 职责 |
|---|---|
| `scripts/data_workbench.py` | 本地 HTTP 服务、V6 路由、错误返回 |
| `data_tools/static/index.html` / `app.js` / `styles.css` | 网页工程、帧/框、表单、保存、检查、导出 |
| `data_tools/workbench.py` | session/工程匹配，准备、导入、完成和导出操作 |
| `data_tools/review_store.py` | CSV 加载，草稿/完成校验，revision 冲突、原子保存和快照 |
| `data_tools/session_store.py` | 只读 raw 帧/时间/事件浏览，不读取旧意图或 chunk |
| `scripts/import_mvp_anylabeling.py` | JSON 框导入、每 session CSV 初始化、共用单行语义校验和导出 |
| `scripts/prepare_mvp_v6.py` | 端点、200 ms replay、因果和 hash 校验、V6 标签导出 |
| `scripts/validate_vla_raw.py` | 原始采集数据完整性检查 |
| `vla/prompt_q.py` | 当前提示→Q/NO_Q 监督约定 |
| `training/m29_dataset.py` | Q 导出准入、M29 训练文件读取 |

上述路径均在 `idv_agent/` 下。标注运行使用 `idv312`，需要 Pillow 和项目依赖；无需
加载 Qwen 权重、无需 GPU。导出 Q 准入会加载 PyTorch 数据集代码，但不会训练。

## API（网页已调用，通常无需手动使用）

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/sessions` | session 列表 |
| GET | `/api/sessions/<name>/mvp-v6` | raw 摘要与工程列表 |
| GET | `/api/mvp-v6/readiness` | 全 session 实时标注覆盖与训练阻塞项 |
| GET | `/api/sessions/<name>/mvp-v6/rows?workspace=...` | 当前 CSV、revision、字段选项 |
| GET | `/api/sessions/<name>/frames/<frame>` | 只读图像 |
| POST | `/api/sessions/<name>/mvp-v6/save-row` | 保存草稿或完成行 |
| POST | `/api/sessions/<name>/mvp-v6/check-review` | 整表检查 |
| POST | `/api/sessions/<name>/mvp-v6/finish-review` | 完成并生成新 V6 导出 |
| POST | `/api/sessions/<name>/mvp-v6/export-q-all` | 导出全部图像 Q 标签 |
| POST | `/api/sessions/<name>/mvp-v6/prepare`、`import` | 新工程准备与导入 |

save-row 示例（revision 用 GET 返回值，不手造）：

```json
{
  "workspace": "C:/Codespace/ID_V_Agent/data/annotations/mvp_v6/<session>_xany_import_v2",
  "row_id": "<session>:00000075", "revision": "<GET返回的CSV SHA256>", "reviewer": "user",
  "patch": {"phase":"align", "decoding":"unknown", "reviewed":"pending"}
}
```

400 表示标签/请求非法，409 表示版本冲突或重复输出，404 表示路径/资源不存在。
旧 actions/intents/chunks/camera-control 路由不可用，不保留 V5 工作台兼容。

## 验证范围

2026-09-09 的 57 项相关测试通过，覆盖真实 HTTP 保存/重载、错误不落盘、冲突不覆盖、
每 session 独立 CSV、源文件保持、完整工程导出以及 Q 独立性；Node 语法、Python 编译和
diff 检查通过。隔离浏览器工程实测通过保存草稿、刷新恢复、完成跳转、缺字段拒绝、
候选框点击和 Q 全帧导出。真实用户标签没有被测试改写。

导出成功表示标签格式可用，不表示 M29 已训练或具备导航泛化；full 训练仍检查
五阶段、当前 decoding 正负、facts/route 类别多样性、prompt bbox 及有效导航覆盖。
现有四条同场景 pilot 不构成独立验证集。
