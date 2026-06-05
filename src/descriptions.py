"""Chinese-facing labels and explanations for the QC report.

Centralises the human-readable (Chinese) name, description, and manual-review
hint for every alarm type, plus severity and input-artifact labels. Keeping this
in one place lets the HTML report stay friendly for reviewers without scattering
strings across the QC modules.
"""
from __future__ import annotations

from typing import Dict, Tuple

from .findings import HIGH, MEDIUM, WARNING

# severity -> (中文名, 颜色)
SEVERITY_ZH: Dict[str, Tuple[str, str]] = {
    HIGH: ("高危", "#c0392b"),
    MEDIUM: ("中等", "#e67e22"),
    WARNING: ("警告", "#f1c40f"),
}

# alarm type -> (中文名, 说明, 人工复核建议)
ALARM_ZH: Dict[str, Tuple[str, str, str]] = {
    "person_missing": (
        "主播人像缺失",
        "检测到人物，但最终合成 alpha 在该人物区域的覆盖率过低，主播可能被抠掉。",
        "查看该时段 foreground/matanyone2/combined_alpha 是否完整包住主播。",
    ),
    "human_alpha_missing": (
        "人像 alpha 偏低",
        "MatAnyone 人像 alpha 对该人物的覆盖不足（仅作辅助诊断）。",
        "查看 matting/person/source_person/pha 在该帧的抠图质量。",
    ),
    "second_person_missing": (
        "第二人像缺失",
        "画面中第二个人（助播）区域的最终 alpha 覆盖率过低。",
        "确认助播是否应保留；检查 combined_alpha 与 assistant_person。",
    ),
    "staff_or_extra_person_risk": (
        "多余人员风险",
        "画面中出现超过 2 人，疑似工作人员或路人入画。",
        "人工确认是否有不该出现的人物，是否需要遮挡。",
    ),
    "assistant_mask_missing": (
        "助播 mask 缺失/偏弱",
        "检测到第二个人，但 assistant_person mask 很弱或不存在。",
        "检查 masks/combined/assistant_person 是否覆盖该人物。",
    ),
    "face_alpha_missing": (
        "人脸 alpha 缺失",
        "人脸区域被最终 alpha 切掉，脸部可能出现残缺。",
        "重点查看脸部边缘在 combined_alpha 的完整性。",
    ),
    "hand_alpha_missing": (
        "手部 alpha 缺失",
        "手部区域被最终 alpha 切掉；手边商品通常也容易被一起切掉。",
        "优先复核：手部及手边商品是否被保留。",
    ),
    "product_missing_near_hand": (
        "手边商品缺失",
        "手边检测到疑似商品，但最终 alpha 对其覆盖率过低，商品可能被切掉。",
        "查看该商品在 combined_alpha / final_keep 是否保留。",
    ),
    "host_product_drop": (
        "主播手持商品掉面积",
        "host_hand_products mask 面积相对前后骤降，商品可能丢失或被遮挡。",
        "检查 masks/combined/host_hand_products 该时段连续性。",
    ),
    "assistant_product_drop": (
        "助播手持商品掉面积",
        "assistant_hand_products mask 面积骤降，商品可能丢失。",
        "检查 masks/combined/assistant_hand_products 该时段连续性。",
    ),
    "table_product_drop": (
        "桌面商品掉面积",
        "table_products mask 面积骤降，桌面商品可能丢失或被遮挡。",
        "检查 masks/combined/table_products 该时段连续性。",
    ),
    "final_keep_object_drop": (
        "保留物体消失",
        "final_keep 中的非字幕连通块持续存在后突然消失。",
        "确认该物体是否应保留，检查 final_keep。",
    ),
    "sam2_object_drop": (
        "SAM2 对象掉面积",
        "同一 SAM2 对象面积相对前后 1 秒中位数骤降，跟踪可能丢失。",
        "查看 masks/sam2/<对象> 该时段掩膜。",
    ),
    "sam2_background_leak": (
        "SAM2 背景泄漏",
        "SAM2 对象面积异常增大，可能错误吃进背景。",
        "查看 masks/sam2/<对象> 是否扩散到背景。",
    ),
    "sam2_drift": (
        "SAM2 漂移",
        "SAM2 对象 bbox 中心突然跳远，超过画面尺寸阈值，跟踪可能漂移。",
        "查看该对象前后帧位置是否突变。",
    ),
    "possible_object_person_merge": (
        "对象与人粘连",
        "对象 mask 与人物 mask 出现异常大面积重叠变化，可能粘连。",
        "检查对象与人物边界是否混在一起。",
    ),
    "matanyone_alpha_drop": (
        "人像 alpha 掉面积",
        "human_alpha 面积相对前后 1 秒中位数骤降，人像可能闪断。",
        "查看 matting/person/source_person/pha 连续性。",
    ),
    "matanyone_person_missing": (
        "人像缺失(MatAnyone)",
        "人稳定存在但 human_alpha 覆盖明显偏低。",
        "检查 MatAnyone 该时段是否丢人。",
    ),
    "matanyone_flicker": (
        "人像闪烁",
        "连续帧 human_alpha IoU 过低，人像 alpha 抖动/闪烁。",
        "查看人像边缘是否逐帧跳变。",
    ),
    "subtitle_mask_missing": (
        "字幕 mask 缺失",
        "未提供上游字幕 mask，已退化为固定区域忽略。",
        "如需更准，提供 masks/combined/subtitles。",
    ),
    "composited_video_missing": (
        "成片缺失",
        "未找到合成成片，已仅运行 alpha 质检。",
        "确认 outputs/manifest.json 与成片是否生成。",
    ),
}

# input_status key -> 中文名
ARTIFACT_ZH: Dict[str, str] = {
    "source_video": "源视频 source.mp4",
    "frames_dir": "源帧目录 frames/source",
    "combined_alpha_dir": "最终合成 alpha combined_alpha",
    "human_alpha_dir": "人像 alpha (MatAnyone)",
    "final_keep_dir": "最终保留区 final_keep",
    "table_products_dir": "桌面商品 table_products",
    "host_hand_products_dir": "主播手持商品 host_hand_products",
    "assistant_person_dir": "助播人像 assistant_person",
    "assistant_hand_products_dir": "助播手持商品 assistant_hand_products",
    "subtitles_dir": "字幕 mask subtitles",
    "sam2_dir": "SAM2 掩膜 masks/sam2",
    "sam2_prompts": "SAM2 prompts",
    "auto_candidates": "auto_candidates",
    "outputs_manifest": "成片清单 outputs/manifest.json",
    "composited_video": "合成成片",
}


def alarm_zh(type_code: str) -> Tuple[str, str, str]:
    return ALARM_ZH.get(type_code, (type_code, "", ""))


def severity_zh(sev: str) -> Tuple[str, str]:
    return SEVERITY_ZH.get(sev, (sev, "#777"))


def artifact_zh(key: str) -> str:
    return ARTIFACT_ZH.get(key, key)
