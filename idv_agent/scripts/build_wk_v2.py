# -*- coding: utf-8 -*-
"""Build the WK v2 training JSONL and knowledge gaps.

WK v2 只面向官方自定义剧本、训练营与沙盒环境，不涉及真人排位、内存读取、
DLL 注入或截帧 hook。所有条目必须有来源、来源类型与审核状态；answer 只写
直接知识陈述，不允许出现「候选 / 待核验 / 不得进入训练」等元描述。

Usage:
    python -m idv_agent.scripts.build_wk_v2

Outputs:
    idv_agent/data/wk/wk_train.jsonl
    idv_agent/data/wk/knowledge_gaps.json
"""

from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "wk"

MATCH_URL = "https://wiki.biligame.com/dwrg/%E5%8C%B9%E9%85%8D%E6%A8%A1%E5%BC%8F"
JOINT_URL = "https://wiki.biligame.com/dwrg/%E8%81%94%E5%90%88%E7%8B%A9%E7%8C%8E%E6%A8%A1%E5%BC%8F"
BLACKJACK_URL = "https://wiki.biligame.com/dwrg/%E9%BB%91%E6%9D%B0%E5%85%8B%E6%A8%A1%E5%BC%8F"
LAWYER_URL = "https://wiki.biligame.com/dwrg/%E5%BE%8B%E5%B8%88/%E8%BF%9B%E9%98%B6"
ITEM_URL = "https://wiki.biligame.com/dwrg/?curid=207"
HEAL_URL = "https://wiki.biligame.com/dwrg/?curid=20236"

ANCHOR_DOC_PATH = "docs/13-WK数据来源与锚点.md"
ANCHOR_TITLE = "WK v2 数据来源与人工审核锚点（项目记录）"


def _community(title: str, url: str, quote: str) -> dict:
    return {
        "title": title,
        "url_or_path": url,
        "source_type": "verified_community",
        "quote_or_excerpt": quote,
    }


def _anchor(quote: str) -> dict:
    return {
        "title": ANCHOR_TITLE,
        "url_or_path": ANCHOR_DOC_PATH,
        "source_type": "project_record",
        "quote_or_excerpt": quote,
    }


def _rec(
    idx: int,
    question: str,
    answer: str,
    mode: str,
    topic: str,
    episode_id: str,
    source: dict,
    status: str,
    review_note: str,
) -> dict:
    return {
        "id": f"wk_{idx:06d}",
        "question": question,
        "answer": answer,
        "mode": mode,
        "topic": topic,
        "episode_id": episode_id,
        "split": "train",
        "source": source,
        "verification": {"status": status, "review_note": review_note},
    }


MATCH = "第五人格 BWIKI：匹配模式"
JOINT = "第五人格 BWIKI：联合狩猎模式"
BLACKJACK = "第五人格 BWIKI：黑杰克模式"
LAWYER = "第五人格 BWIKI：律师/进阶"
ITEM = "第五人格 BWIKI：道具"


RECORDS: list[dict] = []

# ============================== rule ==============================
R_EP = "wk_rules_v2"

RECORDS.append(_rec(
    1,
    "匹配模式下每局双方各有多少名玩家？",
    "匹配模式为 1 名监管者对抗 4 名求生者的非对称竞技模式，双方共 5 名玩家。",
    "standard", "rule", R_EP,
    _community(MATCH, MATCH_URL, "匹配模式为1位监管者对抗4位求生者的非对称竞技模式。"),
    "reviewed", "人工审核确认锚点（1 名监管者 vs 4 名求生者），直接采信。",
))

RECORDS.append(_rec(
    2,
    "匹配模式下每张地图有几台密码机？破译多少台后大门通电？",
    "每张地图刷新 7 台密码机，求生者破译其中 5 台后大门通电。",
    "standard", "rule", R_EP,
    _community(MATCH, MATCH_URL, "每张地图都会刷新7台密码机，求生者需要破译其中的5台，并开启任意一个大门后从任意一个开启的大门逃脱。"),
    "reviewed", "人工审核确认锚点（每图 7 台、破译 5 台后大门通电），直接采信。",
))

RECORDS.append(_rec(
    3,
    "联合狩猎模式每局有多少名玩家？",
    "联合狩猎需要 10 名玩家同时参与，包括 2 名监管者和 8 名求生者。",
    "joint_hunt", "rule", R_EP,
    _community(JOINT, JOINT_URL, "联合狩猎模式需要10名玩家同时参与，包括两名监管者和8名求生者。"),
    "reviewed", "人工审核确认锚点（2 名监管者 vs 8 名求生者，共 10 人），直接采信。",
))

RECORDS.append(_rec(
    4,
    "联合狩猎模式有几台密码机？需要破译几台？",
    "联合狩猎模式密码机调整为 11 台，求生者需要破译 7 台才能开启大门。",
    "joint_hunt", "rule", R_EP,
    _community(JOINT, JOINT_URL, "密码机的数量调整为11台，求生者需要破译7台密码机才能开启大门。"),
    "reviewed", "人工审核确认锚点（密码机 11 台、需破译 7 台），直接采信。",
))

RECORDS.append(_rec(
    5,
    "联合狩猎模式如何判定胜负和平局？",
    "联合狩猎中求生者逃生 4 人为平局，逃生 4 人以上为胜利，逃生 4 人以下为失败。",
    "joint_hunt", "rule", R_EP,
    _community(JOINT, JOINT_URL, "求生者逃生4人双方平局，逃生4人以上为胜利，4人以下为失败。"),
    "reviewed", "人工审核确认锚点（逃生 4 人平局、4 人以上胜、4 人以下败），直接采信。",
))

RECORDS.append(_rec(
    6,
    "黑杰克模式需要几名玩家？玩家如何被淘汰？",
    "黑杰克模式 5 名玩家各自为战，每轮手牌点数超过 21 点的玩家被淘汰；坚持到最后者获胜，凑齐 21 点的玩家提前获得胜利。",
    "blackjack", "rule", R_EP,
    _community(BLACKJACK, BLACKJACK_URL, "五名玩家各自为战，通过控制各自手中卡牌的点数来对决。每轮超过21点的玩家将被淘汰，坚持到最后者将获得胜利。而凑齐21点的玩家将提前获得胜利。"),
    "needs_review", "单一社区来源，需官方说明或游戏内复核。",
))

RECORDS.append(_rec(
    7,
    "匹配模式下求生者被淘汰前会经历哪些阶段？",
    "求生者被击倒后可由监管者牵上气球并挂上狂欢之椅，椅上阶段结束后起飞淘汰；若倒地时间累计满 240 秒仍未被救助或被挂椅，也会被直接淘汰（俗称放血死）。",
    "standard", "rule", R_EP,
    _community(MATCH, MATCH_URL, "倒地状态下的求生者如果在倒地时间总计达到240秒后仍未被其他求生者救助恢复状态或牵上气球，挂上狂欢之椅（即倒地时间累积至4分钟），将直接被淘汰出局（即玩家所说的放血死）。"),
    "needs_review", "单一社区来源，淘汰时间线需官方说明复核。",
))

RECORDS.append(_rec(
    8,
    "破译密码机时的校准（QTE）机制是怎样的？",
    "破译时有概率触发校准；完美校准立即获得 1% 破译进度，校准失败则触电、破译进度回退 2%、打断破译进程并对监管者形成提示。",
    "standard", "rule", R_EP,
    _community(MATCH, MATCH_URL, "如果校准结果为完美校准，则会立即获得1%的破译进度。一般情况下，如果校准失败，校准失败的求生者将会触电，同时破译进度会回退2%，并打断破译进程、对监管者形成提示。"),
    "needs_review", "单一社区来源；数值随版本调整过，需复核当前版本。",
))

RECORDS.append(_rec(
    9,
    "箱子在匹配模式中如何运作？哪些求生者可以更换开箱得到的道具？",
    "箱子在地图固定点位刷新且只有求生者能交互，开启后随机获得手持物；只有支持手持物替换的求生者可更换箱子中获得的手持物，不支持的仍可开箱但无法替换。",
    "standard", "rule", R_EP,
    _community(MATCH, MATCH_URL, "只有求生者能与箱子交互。只有支持手持物替换的求生者可更换箱子获得的手持物，不支持手持物替换的求生者仍可开启箱子，但无法替换手持物。"),
    "needs_review", "单一社区来源，开箱可获道具列表随版本调整，需复核。",
))

RECORDS.append(_rec(
    10,
    "匹配模式下地窖在什么条件下刷新和开启？",
    "破译完成 2 台密码机后，地窖在地图 3-5 个固定刷新点之一随机刷新；当求生者只剩 1 名时地窖盖打开，该求生者可从中逃脱；地窖开启期间约每 3 分钟重新刷新位置。",
    "standard", "rule", R_EP,
    _community(MATCH, MATCH_URL, "地窖会在密码机破译完成2台的情况下在地图的3-5个固定地窖刷新点之一随机刷新。当求生者只剩1名时，地窖盖将会打开，该求生者可从地窖逃脱。同时，在地窖开启的情况下，地窖每隔3分钟会重新刷新位置。"),
    "needs_review", "单一社区来源，刷新/开启条件需官方说明复核。",
))

RECORDS.append(_rec(
    11,
    "联合狩猎模式地窖的刷新和开启规则有什么不同？",
    "联合狩猎破译完成 3 台密码机后地窖刷新，求生者剩余 1 人时自动开启；求生者也可在电话亭购买撬棍主动开启地窖，撬棍对全图求生者显示位置。",
    "joint_hunt", "rule", R_EP,
    _community(JOINT, JOINT_URL, "破译完成3台密码机刷新地窖，求生者剩余1人时地窖自动开启，求生者也可购买撬棍主动开启地窖。"),
    "needs_review", "单一社区来源，撬棍刷新/购买规则需官方说明复核。",
))

RECORDS.append(_rec(
    12,
    "黑杰克模式的回合制流程是怎样的？",
    "黑杰克为回合制，每回合分为发牌、分散、变身、追逃、抽牌与结算六个阶段；普通回合追逃阶段限时 100 秒，特殊回合限时 60 秒；每回合会重置求生者的道具数量。",
    "blackjack", "rule", R_EP,
    _community(BLACKJACK, BLACKJACK_URL, "普通回合追逃阶段限时100秒，特殊回合追逃阶段限时60秒。每个回合可以分为发牌、分散、变身、追逃、抽牌与结算共六个阶段。每回合会重置求生者的道具数量，但是道具不会累加。"),
    "needs_review", "单一社区来源；回合时长 2023 年调整过，需复核当前版本。",
))

RECORDS.append(_rec(
    13,
    "黑杰克模式中监管者造成的伤害是多少？",
    "黑杰克追逃阶段中，技能伤害固定为 500 点恐惧值（相当于匹配模式普通攻击命中一次），攻击伤害固定为 1000 点恐惧值（一次攻击即可令普通求生者倒地）。",
    "blackjack", "rule", R_EP,
    _community(BLACKJACK, BLACKJACK_URL, "追逃阶段中，技能伤害固定为500点恐惧值（等效于匹配模式普通监管者普通攻击命中一次），攻击伤害固定为1000点恐惧值（一次攻击即可令普通求生者倒地）。"),
    "needs_review", "单一社区来源，伤害数值需官方说明复核。",
))

RECORDS.append(_rec(
    14,
    "多人同时破译同一台密码机时速度如何变化？",
    "多人合作破译时每个求生者的破译速度都会下降：双人、三人、四人合作时分别降低 25%、35%、40%，整体速度相当于单人破译的 1.5 倍、1.95 倍、2.4 倍。",
    "standard", "rule", R_EP,
    _community(MATCH, MATCH_URL, "双人、三人、四人合作破译时，每个求生者的破译速度分别降低25%、35%、40%。即，假若求生者均未拥有破译增益/减益效果，双人、三人、四人破译速度等同于单人破译速度的1.5倍、1.95倍、2.4倍。"),
    "needs_review", "单一社区来源，合作惩罚数值需官方说明复核。",
))

# ============================== object ==============================
O_EP = "wk_objects_v2"

RECORDS.append(_rec(
    15,
    "密码机是什么？破译它的作用是什么？",
    "密码机是求生者破译的核心对象：每张地图 7 台，破译 5 台后大门通电；靠近密码机可以看到破译进度条，未被完全破译的密码机可作为监管者传送或渡鸦的目标点。",
    "standard", "object", O_EP,
    _community(MATCH, MATCH_URL, "每张地图一共七台密码机，求生者仅需破译其中五台密码机。求生者和监管者在靠近密码机时，可以看到密码机的破译进度条。"),
    "needs_review", "单一社区来源，对象语义需游戏内/官方资料复核。",
))

RECORDS.append(_rec(
    16,
    "狂欢之椅的作用是什么？",
    "狂欢之椅是监管者淘汰求生者的装置：倒地被牵上气球的求生者被挂上狂欢之椅，椅上阶段结束后起飞被淘汰；求生者可用工具箱拆除狂欢之椅。",
    "standard", "object", O_EP,
    _community(JOINT, JOINT_URL, "战斗内求生者第一次被放上狂欢之椅会进入禁锢阶段，在该阶段内求生者无法被救下，禁锢阶段结束后进入短暂的营救阶段，该阶段求生者可被救。"),
    "needs_review", "BWIKI 狂欢之椅独立页面为归宿家具页，机制以上椅阶段描述为准，需复核。",
))

RECORDS.append(_rec(
    17,
    "木板有什么功能？",
    "地图中有若干木板，求生者可放下木板，放下后形成碰撞体积阻挡监管者；若放下时监管者位于判定区域内会被砸晕；木板放下后无法再次翻起，被破坏后无法复原。",
    "standard", "object", O_EP,
    _community(MATCH, MATCH_URL, "求生者可以放下木板，放下后木板将存在碰撞体积，可阻挡监管者，如放下时监管者位于木板判定区域内则将被击晕。木板被放下后，无法再次翻起。"),
    "needs_review", "单一社区来源，对象语义需游戏内复核。",
))

RECORDS.append(_rec(
    18,
    "窗户在追击中起什么作用？",
    "求生者和监管者都可以翻越窗户，通常监管者翻越速度低于求生者；正在翻越窗户时，其他玩家不能同时翻越该窗户。",
    "standard", "object", O_EP,
    _community(MATCH, MATCH_URL, "求生者和监管者都可以翻越窗户，但通常情况下监管者翻越窗户的速度低于求生者。在求生者或监管者正在翻越窗户时，其他玩家均不可以翻越该窗户。"),
    "needs_review", "单一社区来源，对象语义需游戏内复核。",
))

RECORDS.append(_rec(
    19,
    "箱子是什么？它提供什么？",
    "箱子在地图固定点位刷新，只有求生者可交互，开启后按概率随机获得手持物；开关箱子的动作约 1.5 秒，搜寻物品约 10 秒。",
    "standard", "object", O_EP,
    _community(MATCH, MATCH_URL, "在地图的固定点位刷新。只有求生者能与箱子交互。开启箱子有不同概率获得不同道具。一般情况下，开关箱子的动作持续时间为1.5秒，搜寻物品的持续时间为10秒。"),
    "needs_review", "单一社区来源，道具概率未公开，仅语义条目进训练。",
))

RECORDS.append(_rec(
    20,
    "电话亭是什么？在哪个模式出现？",
    "电话亭是联合狩猎模式新增的场景设施，共 5 个，求生者和监管者可通过电话亭使用积分购买道具；距离任意玩家最近的电话亭会对其高亮显示。",
    "joint_hunt", "object", O_EP,
    _community(JOINT, JOINT_URL, "场景内新增5个电话亭，求生者和监管者均可以通过电话亭与庄园主通话并使用获得的积分购买道具。距离任意玩家最近的电话亭将会对其高亮显示。"),
    "needs_review", "单一社区来源，电话亭购买规则需官方说明复核。",
))

RECORDS.append(_rec(
    21,
    "地窖是什么？",
    "地窖是求生者的额外逃脱通道：匹配模式破译 2 台密码机后刷新、仅剩 1 名求生者时开启；联合狩猎需破译 3 台，且可用撬棍主动开启。",
    "standard", "object", O_EP,
    _community(MATCH, MATCH_URL, "地窖会在密码机破译完成2台的情况下在地图的3-5个固定地窖刷新点之一随机刷新。当求生者只剩1名时，地窖盖将会打开，该求生者可从地窖逃脱。"),
    "needs_review", "同一 WIKI 匹配模式页与联合狩猎页交叉支撑，仍需官方说明复核。",
))

RECORDS.append(_rec(
    22,
    "逃生门（大门）是什么？开启需要什么条件？",
    "每张地图有两个大门，求生者破译完 5 台密码机后可开启，从任意一个大门出去均判定为逃脱；一般情况下单人破译大门需要 18 秒。",
    "standard", "object", O_EP,
    _community(MATCH, MATCH_URL, "每张地图一共两个大门，分别是金属门扇的正门（大门）与木质门扇的侧门（小门），求生者破译完五台密码机后可开启，从任意一个大门出去均判定为逃脱。一般情况下，单人破译大门所需时间为18秒。"),
    "needs_review", "单一社区来源，开门时间随角色/天赋变化，需复核。",
))

RECORDS.append(_rec(
    23,
    "律师随身携带的手绘地图能查看什么？",
    "律师的手绘地图可查阅全图范围内未破译完成的密码机、已通电的大门、求生者、监管者和巡视者轮廓；不能透视机械师玩偶与记者幻影，轮廓出现与消失约有 1 秒延迟。",
    "standard", "object", O_EP,
    _community(LAWYER, LAWYER_URL, "随身携带地图，可查阅全图范围内未破译完成的密码机、已通电的大门、求生者、监管者、巡视者轮廓；不能透视机械师的玩偶与记者的幻影；轮廓的出现与消失有约1s延迟。"),
    "needs_review", "单一社区来源，律师地图功能需官方说明复核。",
))

RECORDS.append(_rec(
    24,
    "律师使用手绘地图的耐久消耗是多少？",
    "律师使用手绘地图时耐久消耗 1.25%/秒（其他求生者使用为 2.5%/秒），打开地图瞬间固定消耗 4% 耐久。",
    "standard", "object", O_EP,
    _community(ITEM, ITEM_URL, "手绘地图：耐久消耗：1.25%/秒（其他求生者使用：2.5%/秒）；打开地图耐久消耗：4%。"),
    "needs_review", "单一社区来源，耐久数值需官方说明复核。",
))

RECORDS.append(_rec(
    25,
    "镇静剂的作用是什么？",
    "镇静剂是求生者受伤状态下的自我治疗道具：使用后减少自己 1/2 恐惧值，一般治疗时间为 15 秒，道具可使用时间为 30 秒（医生为无消耗）。",
    "standard", "object", O_EP,
    _community(MATCH, MATCH_URL, "在获得手持物镇静剂后，可以通过使用进行自我治疗。减少自己1/2的恐惧值。"),
    "needs_review", "匹配模式页与联合狩猎页交叉支撑，仍需官方说明复核。",
))

RECORDS.append(_rec(
    26,
    "工具箱的作用是什么？",
    "工具箱用于拆除场景内的狂欢之椅，标准可拆除时间为 3 秒。",
    "standard", "object", O_EP,
    _community(ITEM, ITEM_URL, "工具箱：可用于拆除场景内的狂欢之椅。标准可拆除时间为3s。"),
    "needs_review", "单一社区来源，拆除时间需官方说明复核。",
))

RECORDS.append(_rec(
    27,
    "黑杰克模式的牌库由哪些牌组成？",
    "黑杰克模式所有玩家共享同一个牌库，共 26 张牌：1-10、J、Q、K 各两张，红色/黑色各一张；J、Q、K 分别代表点数 1、2、3。",
    "blackjack", "object", O_EP,
    _community(BLACKJACK, BLACKJACK_URL, "牌库中总共有26张牌，1、2、3、4、5、6、7、8、9、10以及J、Q、K各两张，红色/黑色各一张。J、Q、K代表点数1、2、3。"),
    "needs_review", "单一社区来源，牌库组成需官方说明复核。",
))

RECORDS.append(_rec(
    28,
    "黑杰克模式的道具卡有哪些类型？",
    "黑杰克模式可用筹码在道具卡盒购买道具卡，包括窥牌卡、换牌卡、洗牌卡、递牌卡、弃牌卡、加速卡、透视卡和幸运卡等；变身为监管者后只可使用弃牌卡、窥牌卡和透视卡。",
    "blackjack", "object", O_EP,
    _community(BLACKJACK, BLACKJACK_URL, "玩家可以通过消耗一定量筹码，在游戏内战斗界面的道具卡盒商城购买道具。变身为监管者后，只可以使用弃牌卡、窥牌卡和透视卡。"),
    "needs_review", "单一社区来源，道具卡可用范围需官方说明复核。",
))

# ============================== state ==============================
S_EP = "wk_states_v2"

RECORDS.append(_rec(
    29,
    "求生者的健康、受伤和倒地状态如何判定？",
    "恐惧值满值为 1（相当于 2 滴血量）：恐惧值低于 1/2 为健康状态，达到 1/2 及以上为受伤状态，达到 1（满值）为倒地状态。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "当求生者的恐惧值达到1/2及以上时，该求生者被判定处于受伤状态。当求生者的恐惧值达到1（满值）时，求生者将成为倒地状态。"),
    "needs_review", "单一社区来源，状态判定需官方说明复核。",
))

RECORDS.append(_rec(
    30,
    "倒地状态下的求生者能做什么？",
    "倒地状态下求生者行动受限，只能缓慢爬行（约 0.44 米/秒）、进行一定程度的自我治疗、被其他求生者治疗，或从已开启的大门/地窖逃脱；除逃离大门、从地窖逃生等少数行为外不能进行其他交互。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "倒地状态下的求生者行动受限，移动变为缓慢地爬行。一般情况下，倒地爬行速度为0.44米/秒。倒地状态下的求生者除逃离大门、从地窖逃生、响应求生者祭司的超长通道以外，不能进行其它任何交互。"),
    "needs_review", "单一社区来源，倒地交互限制需官方说明复核。",
))

RECORDS.append(_rec(
    31,
    "倒地求生者多长时间会被直接淘汰？",
    "一般情况下，倒地时间累计达到 240 秒（4 分钟）仍未被救助恢复状态或挂上狂欢之椅的求生者会被直接淘汰（俗称放血死）；每次倒地时间会累积计算。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "倒地状态下的求生者如果在倒地时间总计达到240秒后仍未被其他求生者救助恢复状态或牵上气球，挂上狂欢之椅（即倒地时间累积至4分钟），将直接被淘汰出局。每次倒地后倒地时间会累积计算。"),
    "needs_review", "单一社区来源，淘汰时间需官方说明复核。",
))

RECORDS.append(_rec(
    32,
    "倒地自愈需要多长时间？",
    "一般情况下，倒地的求生者突破自愈上限自愈完成需要 30 秒。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "一般情况下，倒地的求生者突破自愈上限自愈完成需要30秒。"),
    "needs_review", "单一社区来源，自愈时长需官方说明复核。",
))

RECORDS.append(_rec(
    33,
    "治疗受伤或倒地的队友一般需要多长时间？效果如何？",
    "可行动的求生者可治疗受伤和倒地的其他求生者，治疗完成后减少其 1/2 恐惧值；正常情况下治疗受伤与倒地的其他求生者需要 15 秒。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "可行动的求生者可以对受伤和倒地状态的其他求生者进行治疗，治疗完成后会减少其1/2的恐惧值。正常情况下，治疗受伤与倒地的其他求生者所需时间为15秒。"),
    "needs_review", "匹配模式页与治疗页交叉支撑，仍需官方说明复核。",
))

RECORDS.append(_rec(
    34,
    "治疗校准失败会怎样？",
    "治疗时有概率触发校准；校准失败则治疗进度回退 10%、打断治疗进程并对监管者形成提示；完美校准立即获得 1% 治疗进度。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "如果校准失败，则治疗进度回退10%，并打断治疗进程、对监管者形成提示。如果校准为完美校准，则会立即获得1%的治疗进度。"),
    "needs_review", "单一社区来源，治疗校准数值需官方说明复核。",
))

RECORDS.append(_rec(
    35,
    "监管者被木板砸晕是怎么回事？",
    "求生者放下木板时，若监管者位于木板判定区域内则会被木板击晕；不同监管者被击晕后的恢复时间各不相同，部分技能/状态（如兴奋、梦之女巫本体）可免疫木板击晕。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "如放下时监管者位于木板判定区域内则将被击晕。一般情况下。监管者被木板击晕后的恢复时间各不相同。"),
    "needs_review", "单一社区来源，眩晕/免疫机制需官方说明复核。",
))

RECORDS.append(_rec(
    36,
    "监管者擦刀（攻击恢复）期间是什么状态？",
    "监管者普通攻击或蓄力攻击命中求生者后进入攻击恢复动作（攻击后摇加擦刀）；期间监管者不可行动、可以转变视角，且位置对所有在场求生者可见。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "命中求生者的攻击恢复动作期间，监管者不可行动，可以转变视角，位置对所有在场求生者可见。"),
    "needs_review", "单一社区来源，擦刀机制需官方说明复核。",
))

RECORDS.append(_rec(
    37,
    "心跳提示代表什么？",
    "求生者进入监管者警戒半径后会出现心跳，监管者越近心跳越剧烈；一般情况下警戒半径为 32.07 米；警戒半径不等于耳鸣生效半径（耳鸣为 36 米）。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "在进入监管者的警戒半径后，求生者会出现心跳。监管者距离求生者越近，则求生者心跳越剧烈。一般情况下，监管者的警戒半径为32.07米。警戒半径不等于耳鸣生效半径（耳鸣生效半径为36米）。"),
    "needs_review", "单一社区来源，警戒半径数值需官方说明复核。",
))

RECORDS.append(_rec(
    38,
    "求生者视角下的红光代表什么？",
    "求生者视角下，监管者模型正面有一片扇形的红色区域，称为视线红光，用于提示监管者的朝向；红光的亮度会根据情况变化。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "在求生者视角下，监管者的模型正面（并非监管者玩家的视野方向）有一片扇形的红色区域，称为视线红光。"),
    "needs_review", "单一社区来源，红光语义需官方说明/本地画面复核。",
))

RECORDS.append(_rec(
    39,
    "受击加速是什么？",
    "求生者受到监管者攻击或绝大部分技能伤害后获得受击加速：移速提升 65%，持续 2 秒；倒地状态下的求生者无法受到受击加速效果增益。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "在触发受击加速之后，求生者移速提升65%，持续2秒。求生者倒地后无法受到受击加速效果增益。"),
    "needs_review", "单一社区来源，受击加速数值需官方说明复核。",
))

RECORDS.append(_rec(
    40,
    "三鸦聚顶（挂机乌鸦）如何触发？",
    "求生者 80 秒不进行破译、治疗、开门或破坏等交互动作，头顶会出现三只乌鸦盘旋（三鸦聚顶），约每 4.5 秒向监管者爆点提示；场上仅剩 1 名求生者时，28.5 秒不交互即触发。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "如果求生者80秒不进行破译、治疗、开门或破坏等交互动作，则无论该地图场景有无场景乌鸦，求生者的头顶将会有三只乌鸦盘旋（又被称为“三鸦聚顶”），每隔约4.5秒向监管者给予爆点提示。"),
    "needs_review", "单一社区来源，挂机触发时长需官方说明复核。",
))

RECORDS.append(_rec(
    41,
    "恐惧震慑是什么？哪些行为会触发？",
    "监管者攻击击中正在交互的求生者会触发恐惧震慑，恐惧值额外上升 500 点（相当于半血）；翻越板窗、开/关箱子、破译、开门、治疗、救援等交互都会触发；律师在任何交互时都不会被恐惧震慑（使用魔术棒隐身期间除外）。",
    "standard", "state", S_EP,
    _community(MATCH, MATCH_URL, "假如监管者攻击或一些监管者的特定技能击中正在交互的求生者，将触发恐惧震慑，恐惧震慑会使求生者的恐惧值额外上升500点。"),
    "needs_review", "匹配模式页与律师页交叉支撑，仍需官方说明复核。",
))

RECORDS.append(_rec(
    42,
    "联合狩猎中求生者第一次上椅后的禁锢/营救阶段是怎样的？",
    "联合狩猎中求生者第一次被放上狂欢之椅会进入禁锢阶段，该阶段内无法被救下；禁锢阶段结束后进入短暂的营救阶段，此阶段才可被救下；常规情况下禁锢阶段为 40 秒。",
    "joint_hunt", "state", S_EP,
    _community(JOINT, JOINT_URL, "战斗内求生者第一次被放上狂欢之椅会进入禁锢阶段，在该阶段内求生者无法被救下，禁锢阶段结束后进入短暂的营救阶段，该阶段求生者可被救。求生者第一次挂上狂欢之椅的禁锢阶段常规情况为40秒。"),
    "needs_review", "单一社区来源，禁锢/营救时长需官方说明复核。",
))

# ============================== interaction ==============================
I_EP = "wk_interactions_v2"

RECORDS.append(_rec(
    43,
    "PC 端求生者如何移动？",
    "使用 WASD 键移动：W 前进、S 后退、A 左移、D 右移；求生者可切换跑动、走路与蹲行三种行走方式。",
    "standard", "interaction", I_EP,
    _anchor("求生者键位：WASD 移动"),
    "reviewed", "人工审核确认锚点（WASD 移动），直接采信。",
))

RECORDS.append(_rec(
    44,
    "空格键在 PC 端有哪些用途？",
    "空格（Space）用于翻越窗户/木板、破译校准（QTE）以及倒地状态自我治疗。",
    "standard", "interaction", I_EP,
    _anchor("求生者键位：空格翻越/QTE/倒地自愈"),
    "reviewed", "人工审核确认锚点（空格翻越/QTE/倒地自愈），直接采信。",
))

RECORDS.append(_rec(
    45,
    "Q 键的用途是什么？",
    "Q 是场景交互键，用于破译、治疗、救援、开门、开箱等场景交互；靠近密码机后按一次 Q 即进入自动破译状态。",
    "standard", "interaction", I_EP,
    _anchor("求生者键位：Q 场景交互"),
    "reviewed", "人工审核确认锚点（Q 场景交互）；自动破译语义与项目 schema 的 INTERACT_HOLD 约定一致。",
))

RECORDS.append(_rec(
    46,
    "E 键的作用是什么？",
    "E 是治疗键，用于治疗受伤或倒地的队友。",
    "standard", "interaction", I_EP,
    _anchor("求生者键位：E 治疗"),
    "reviewed", "人工审核确认锚点（E 治疗），直接采信。",
))

RECORDS.append(_rec(
    47,
    "F 键的作用是什么？",
    "F 是使用道具键，用于使用手持物/道具（部分角色亦用作一技能键）。",
    "standard", "interaction", I_EP,
    _anchor("求生者键位：F 使用道具"),
    "reviewed", "人工审核确认锚点（F 使用道具），直接采信。",
))

RECORDS.append(_rec(
    48,
    "Shift 键的作用是什么？",
    "Shift 用于跑步/走路切换。",
    "standard", "interaction", I_EP,
    _anchor("求生者键位：Shift 跑/走切换"),
    "reviewed", "人工审核确认锚点（Shift 跑/走切换），直接采信；项目 keymap.py 尚未登记该键，已记入知识缺口。",
))

RECORDS.append(_rec(
    49,
    "Ctrl 键的作用是什么？",
    "Ctrl 用于蹲下（蹲行）。",
    "standard", "interaction", I_EP,
    _anchor("求生者键位：Ctrl 蹲下"),
    "reviewed", "人工审核确认锚点（Ctrl 蹲下），直接采信；项目 keymap.py 尚未登记该键，已记入知识缺口。",
))

RECORDS.append(_rec(
    50,
    "T 键的作用是什么？",
    "T 用于拆除监管者的创生物（如窥视者、傀儡、触手等）。",
    "standard", "interaction", I_EP,
    _anchor("求生者键位：T 拆除监管者创生物"),
    "reviewed", "人工审核确认锚点（T 拆除监管者创生物），直接采信；项目 keymap.py 尚未登记该键，已记入知识缺口。",
))

RECORDS.append(_rec(
    51,
    "X 键的作用是什么？",
    "X 用于涂鸦。",
    "standard", "interaction", I_EP,
    _anchor("求生者键位：X 涂鸦"),
    "reviewed", "人工审核确认锚点（X 涂鸦），直接采信；项目 keymap.py 尚未登记该键，已记入知识缺口。",
))

RECORDS.append(_rec(
    52,
    "破译密码机的前置条件是什么？",
    "靠近密码机并出现交互提示（扳手图标）后按 Q 开始破译；破译中会持续锁定破译状态，期间有概率触发校准，校准按空格完成。",
    "standard", "interaction", I_EP,
    _community(MATCH, MATCH_URL, "求生者在破译密码机时，会有一定概率触发校准（俗称QTE）。"),
    "needs_review", "交互提示（扳手图标）来自项目数据规范，破译触发条件来自 BWIKI，建议游戏内复核。",
))

RECORDS.append(_rec(
    53,
    "救援被挂上狂欢之椅的队友需要什么条件？",
    "队友被挂上狂欢之椅后（联合狩猎需先经过不可救援的禁锢阶段），靠近椅子并出现交互提示后按 Q 交互即可救援；救援属于会触发恐惧震慑的交互行为。",
    "standard", "interaction", I_EP,
    _community(MATCH, MATCH_URL, "求生者会触发恐惧震慑的交互行为，包含：解救狂欢之椅之上的求生者。"),
    "needs_review", "单一社区来源，救援触发条件需官方说明复核。",
))

RECORDS.append(_rec(
    54,
    "翻越窗户或木板的前置条件是什么？",
    "靠近窗户或板子并出现翻越提示后按空格（Space），角色翻越障碍；翻越过程中无法进行其他交互。",
    "standard", "interaction", I_EP,
    _community(MATCH, MATCH_URL, "求生者进行与窗户的交互可分为快翻（撑翻）与慢翻（踩翻）。求生者进行与木板的交互可分为快翻（滚翻）、中速翻板（走翻）与慢翻（爬翻）。"),
    "needs_review", "单一社区来源，翻越触发条件需官方说明复核。",
))

RECORDS.append(_rec(
    55,
    "开启大门需要什么条件？",
    "破译完 5 台密码机后大门通电，求生者靠近大门出现交互提示后按 Q 交互即可开启；单人一般情况下需要 18 秒。",
    "standard", "interaction", I_EP,
    _community(MATCH, MATCH_URL, "求生者破译完五台密码机后可开启，从任意一个大门出去均判定为逃脱。一般情况下，单人破译大门所需时间为18秒。"),
    "needs_review", "单一社区来源，开门前置条件需官方说明复核。",
))

RECORDS.append(_rec(
    56,
    "开箱子的交互规则是什么？",
    "靠近箱子按交互键开/关箱子（约 1.5 秒），继续交互进入搜寻物品（约 10 秒）；开关箱子时被打会触发恐惧震慑，搜寻物品期间被打不会触发。",
    "standard", "interaction", I_EP,
    _community(MATCH, MATCH_URL, "在开关箱子时被监管者攻击将触发恐惧震慑，但是在搜寻物品期间被监管者攻击不会触发恐惧震慑。一般情况下，开关箱子的动作持续时间为1.5秒，搜寻物品的持续时间为10秒。"),
    "needs_review", "单一社区来源，开箱交互细节需官方说明复核。",
))

# ============================== mode ==============================
M_EP = "wk_modes_v2"

RECORDS.append(_rec(
    57,
    "standard 和 joint_hunt 的人数有什么差异？",
    "匹配模式为 1 名监管者 vs 4 名求生者；联合狩猎为 2 名监管者 vs 8 名求生者（共 10 人）。",
    "joint_hunt", "mode", M_EP,
    _community(JOINT, JOINT_URL, "联合狩猎模式需要10名玩家同时参与，包括两名监管者和8名求生者。"),
    "reviewed", "人工审核确认锚点（2v8 共 10 人），与匹配模式 1v4 对比，直接采信。",
))

RECORDS.append(_rec(
    58,
    "standard 和 joint_hunt 的密码机目标有什么差异？",
    "匹配模式每图 7 台密码机、破译 5 台后大门通电；联合狩猎 11 台密码机、需破译 7 台。",
    "joint_hunt", "mode", M_EP,
    _community(JOINT, JOINT_URL, "密码机的数量调整为11台，求生者需要破译7台密码机才能开启大门。"),
    "reviewed", "人工审核确认锚点（11 台/7 台），与匹配模式 7 台/5 台对比，直接采信。",
))

RECORDS.append(_rec(
    59,
    "联合狩猎模式中求生者可以携带几个手持物？",
    "联合狩猎中求生者可同时携带两个手持物，监管者可携带一个道具。",
    "joint_hunt", "mode", M_EP,
    _community(JOINT, JOINT_URL, "求生者可同时携带两个手持物，监管者可携带一个道具。"),
    "reviewed", "人工审核确认锚点（求生者可同时携带两个手持物），直接采信。",
))

RECORDS.append(_rec(
    60,
    "联合狩猎的电话亭如何运作？",
    "联合狩猎场景新增 5 个电话亭，求生者和监管者通过游戏表现获得积分，在电话亭用积分购买道具。",
    "joint_hunt", "mode", M_EP,
    _community(JOINT, JOINT_URL, "场景内新增5个电话亭，求生者和监管者均可以通过电话亭与庄园主通话并使用获得的积分购买道具。"),
    "reviewed", "人工审核确认锚点（5 个电话亭、积分购买道具），直接采信。",
))

RECORDS.append(_rec(
    61,
    "standard 和 joint_hunt 的破译速度参数有什么差异？",
    "匹配模式单人基础破译约 81 秒，200 秒后触发破译加速（速度 +30%）；联合狩猎基础破译时间增加为 95 秒，且没有破译加速阶段。",
    "joint_hunt", "mode", M_EP,
    _community(JOINT, JOINT_URL, "密码机基础破译时间增加为95秒，且没有破译加速阶段。"),
    "needs_review", "匹配模式页与联合狩猎页交叉支撑，破译参数需官方说明复核。",
))

RECORDS.append(_rec(
    62,
    "standard 和 joint_hunt 的大门开启时间有什么差异？",
    "匹配模式单人开门一般 18 秒；联合狩猎开门更慢，单个大门输入时间增加至 32 秒。",
    "joint_hunt", "mode", M_EP,
    _community(JOINT, JOINT_URL, "求生者开启大门速度更慢，单个大门输入时间增加至32秒。"),
    "needs_review", "匹配模式页与联合狩猎页交叉支撑，开门时间需官方说明复核。",
))

RECORDS.append(_rec(
    63,
    "standard 和 joint_hunt 的地窖规则有什么差异？",
    "匹配模式破译 2 台密码机后地窖刷新、仅剩 1 名求生者时开启；联合狩猎破译 3 台后刷新，求生者也可购买撬棍主动开启地窖。",
    "joint_hunt", "mode", M_EP,
    _community(JOINT, JOINT_URL, "破译完成3台密码机刷新地窖，求生者剩余1人时地窖自动开启，求生者也可购买撬棍主动开启地窖。"),
    "needs_review", "匹配模式页与联合狩猎页交叉支撑，地窖规则需官方说明复核。",
))

RECORDS.append(_rec(
    64,
    "联合狩猎中求生者需要被击中几次才倒地？",
    "联合狩猎中求生者需要被击中三次才会倒地（匹配模式一般两次）；监管者获得 1500 存在感解锁 1 阶技能，4000 存在感解锁 2 阶技能。",
    "joint_hunt", "mode", M_EP,
    _community(JOINT, JOINT_URL, "求生者需要被击中三次才会倒地，监管者获得1500存在感解锁1阶技能，4000存在感解锁2阶技能。"),
    "needs_review", "单一社区来源，存在感/击倒次数需官方说明复核。",
))

RECORDS.append(_rec(
    65,
    "黑杰克模式场景中缺少哪些传统模式设施？",
    "黑杰克模式场景中没有狂欢之椅和箱子（正常开放的地下室内两者均被移除），玩家也不可打开柜子。",
    "blackjack", "mode", M_EP,
    _community(BLACKJACK, BLACKJACK_URL, "场景中没有狂欢之椅，没有箱子（正常开放的地下室内两者均被移除）。玩家不可打开柜子。"),
    "needs_review", "单一社区来源，模式差异需官方说明复核。",
))

RECORDS.append(_rec(
    66,
    "黑杰克模式的破译速度是多少？",
    "黑杰克模式中求生者的密码机破译所需时间缩短为 40 秒（约 2.5%/秒）。",
    "blackjack", "mode", M_EP,
    _community(BLACKJACK, BLACKJACK_URL, "求生者的密码机破译所需时间缩短为40秒。求生者破译一台密码机的基础速度约2.5%/秒。"),
    "needs_review", "单一社区来源，模式差异需官方说明复核。",
))

RECORDS.append(_rec(
    67,
    "黑杰克模式的双身份和回合制是什么？",
    "黑杰克模式 5 名玩家各自为战，每名玩家在准备阶段选择 1 名求生者和 1 名监管者作为双身份；每回合扑克牌点数最高的玩家变身为监管者进入追逃阶段。",
    "blackjack", "mode", M_EP,
    _community(BLACKJACK, BLACKJACK_URL, "每名玩家将在准备阶段挑选1名求生者以及1名监管者作为自己的“双身份”加入到游戏当中。发牌之后，在场玩家中扑克牌诅咒点数最高的玩家将变身异化为在准备阶段选择的监管者。"),
    "needs_review", "单一社区来源，模式机制需官方说明复核。",
))

RECORDS.append(_rec(
    68,
    "黑杰克模式中凑齐 21 点会怎样？",
    "任意角色在任意时间达到 21 点将直接中断该回合、跳过结算阶段进入狙击 21 点回合；若该角色在回合结束后仍保持 21 点，将直接获得游戏胜利；多个 21 点可同时获得胜利。",
    "blackjack", "mode", M_EP,
    _community(BLACKJACK, BLACKJACK_URL, "当任意角色在任意时间达到21点时，将直接中断该回合，跳过结算阶段，进入狙击21点回合。假如该角色在回合结束后仍保持21点，将直接获得游戏胜利。多个21点可以同时获得胜利。"),
    "needs_review", "单一社区来源，21 点规则需官方说明复核。",
))

RECORDS.append(_rec(
    69,
    "黑杰克决胜回合的胜负如何判定？",
    "当仅剩 2 名玩家时进入决胜回合；两人点数均小于 21 点时点数更大者胜，均大于 21 点时点数更小者胜，点数相同时同时获胜；普通获胜身份为“继承者”。",
    "blackjack", "mode", M_EP,
    _community(BLACKJACK, BLACKJACK_URL, "在决胜回合中，两人手牌点数均小于21点时，点数更大。在决胜回合中，两人手牌点数均大于21点时，点数更小。在决胜回合中，两人手牌点数相同时，同时获胜。"),
    "needs_review", "单一社区来源，决胜规则需官方说明复核。",
))

RECORDS.append(_rec(
    70,
    "standard 和 blackjack 的胜负目标有什么根本差异？",
    "匹配模式是 1 名监管者 vs 4 名求生者的团队对抗，求生者破译 5 台密码机后开启大门逃脱以争取团队胜利；黑杰克是 5 名玩家各自为战的 21 点对抗，控制手牌点数、避免超过 21 点并争取凑齐 21 点或坚持到最后。",
    "blackjack", "mode", M_EP,
    _community(BLACKJACK, BLACKJACK_URL, "黑杰克玩法基于21点扑克牌的规则：五名玩家各自为战，通过控制各自手中卡牌的点数来对决。"),
    "needs_review", "匹配模式页与黑杰克页交叉支撑，需官方说明复核；模式差异独立采证，不从 standard 推断。",
))


KNOWLEDGE_GAPS: dict = {
    "generated_at": "2026-09-01",
    "purpose": "WK v2 知识缺口清单；这些主题缺少可靠来源或需项目内部确认，不进入训练集。",
    "gaps": [
        {
            "id": "gap_001",
            "topic": "interaction",
            "question": "项目 keymap 与人工锚点键位不一致",
            "description": "人工审核锚点包含 Shift（跑/走切换）、Ctrl（蹲下）、T（拆除监管者创生物）、X（涂鸦），但 idv_agent/configs/keymap.py 当前只登记 WASD/Q/Space/F/E/Tab/V/1-4，未定义 Shift/Ctrl/T/X。",
            "required_action": "以游戏内实际键位设置为准同步 keymap 或建立 keymap.json，并在官方自定义剧本/训练营复核后再纳入 WK/ACT。",
        },
        {
            "id": "gap_002",
            "topic": "object",
            "question": "狂欢之椅独立机制页面缺失",
            "description": "BWIKI 的「狂欢之椅」独立页面是归宿家具页面，不是对局机制页；上椅/起飞/淘汰参数只能从匹配模式页相关段落和联合狩猎页禁锢/营救阶段间接采证，完整的椅阶段时间线（含挣扎、起飞）缺少独立可靠来源。",
            "required_action": "用官方说明或训练营录制复核上椅起飞时间、挣扎与救援窗口。",
        },
        {
            "id": "gap_003",
            "topic": "mode",
            "question": "黑杰克当前版本与自定义黑杰克规则一致性",
            "description": "黑杰克页面标注了地图限制（自定义可选择更多地图）与大量角色修正；匹配黑杰克与自定义黑杰克是否完全同规则需要额外确认。",
            "required_action": "在官方自定义剧本中实测自定义黑杰克的可选地图、回合时长与道具卡可用性。",
        },
        {
            "id": "gap_004",
            "topic": "object",
            "question": "各角色技能/道具效果完整列表",
            "description": "本轮聚焦律师与核心通用对象，未覆盖全部角色专属道具数值与地图点位数据。",
            "required_action": "按后续 VG/ACT 需要分批采集，优先律师相关的要点记录与地图机制。",
        },
        {
            "id": "gap_005",
            "topic": "state",
            "question": "交互提示视觉形态（扳手图标/黄色高亮）的官方或本地证据",
            "description": "「靠近密码机出现交互提示（扳手图标）后按 Q」中的提示图标描述来自项目数据规范；密码机被墙体遮挡时的黄色高亮语义尚未在本地录制或官方文档中单独取证。",
            "required_action": "在官方自定义剧本/训练营录制可见密码机、墙后高亮、交互提示、破译状态样本，建立 VG 图像证据。",
        },
        {
            "id": "gap_006",
            "topic": "rule",
            "question": "官方版本化规则来源",
            "description": "当前模式规则主要来自 BWIKI（玩家自建 wiki，页面声明与官方有合作但非官方站点）；缺少官方公告原文或游戏内说明的版本化存档。",
            "required_action": "优先补充第五人格官网/版本公告/游戏内说明的原文摘录，替换或交叉验证社区来源。",
        },
        {
            "id": "gap_007",
            "topic": "mode",
            "question": "联合狩猎混沌效应等限时机制",
            "description": "联合狩猎页记录了 2025 年 3-4 月限时的混沌效应（高温/尘暴/塌陷）；该机制是否会在训练/自定义环境出现、是否纳入 WK 需要额外确认。",
            "required_action": "确认当前版本自定义联合狩猎是否包含限时机制，避免模型把限时规则当常驻规则。",
        },
    ],
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "wk_train.jsonl"
    gaps_path = OUT_DIR / "knowledge_gaps.json"

    ids = [r["id"] for r in RECORDS]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicated record id")
    if not 50 <= len(ids) <= 100:
        raise SystemExit(f"record count out of range: {len(ids)}")

    with train_path.open("w", encoding="utf-8") as handle:
        for record in RECORDS:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with gaps_path.open("w", encoding="utf-8") as handle:
        json.dump(KNOWLEDGE_GAPS, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"wrote {len(ids)} records -> {train_path}")
    print(f"wrote knowledge gaps -> {gaps_path}")


if __name__ == "__main__":
    main()
