"""
离线效果评估脚本 —— 评估诊断 agent 的核心质量（无需数据库 / 服务运行）。

评估两个维度（对应"成熟 agent 必须能量化效果"的标准）：
  1. 主要诊断命中率：模型 primary_diagnosis 与专家标注是否一致（模糊匹配）。
  2. 分诊紧急度一致率：模型 urgency 与专家标注是否一致（精确匹配）。

运行前提：已在 backend/.env 配置 GOOGLE_API_KEY。

用法（在 backend 目录下执行）：
    python scripts/eval_agent.py                 # 跑内置评测集，结果打印到控制台
    python scripts/eval_agent.py --cases scripts/eval_cases.json --out eval_report.json
    python scripts/eval_agent.py --list-cases   # 仅查看评测集条目，不调用模型
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# 确保无论从哪个目录运行都能 import 到 backend 下的 app 包。
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 复用生产诊断/分诊的结构化 schema，保证评估口径与线上一致。
from app.services.agents.nodes.diagnostician import DiagnosisAssessment
from app.services.agents.nodes.triage import TriageResponse
from app.services.llm.gemini_client import gemini_client

DEFAULT_CASES = Path(__file__).resolve().parent / "eval_cases.json"
STOPWORDS = {"and", "the", "of", "with", "for", "to", "a", "an", "in", "on", "or", "is", "are"}


# ---------------------------------------------------------------------------
# 评分逻辑（纯函数，可独立测试）
# ---------------------------------------------------------------------------
def _normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def diagnosis_hit(predicted: str, expected_list: list[str]) -> bool:
    """专家标注任一候选的关键词，在预测诊断中出现 >=50% 即视为命中。"""
    pred = _normalize_tokens(predicted)
    for expected in expected_list:
        tokens = _normalize_tokens(expected) - STOPWORDS
        if not tokens:
            continue
        coverage = sum(1 for t in tokens if t in pred) / len(tokens)
        if coverage >= 0.5:
            return True
    return False


def urgency_hit(predicted: str, expected: str) -> bool:
    return (predicted or "").strip().upper() == (expected or "").strip().upper()


# ---------------------------------------------------------------------------
# Prompt 构建（与 diagnostician / triage 节点保持同一口径，但不查 RAG，降低评估门槛）
# ---------------------------------------------------------------------------
def _diagnosis_prompt(case: dict[str, Any]) -> str:
    return (
        "You are a senior clinical diagnostician. Analyze the patient data and provide a "
        "structured diagnosis.\n\n"
        f"[PATIENT DATA]\nAge: {case.get('age', 'Unknown')}\n"
        f"Gender: {case.get('gender', 'Unknown')}\n"
        f"Symptoms: {case['symptoms']}\n\n"
        "Instructions:\n"
        "- Prioritize clinical safety and list reasonable differential diagnoses.\n"
        "- Your output is decision support only and does NOT replace a licensed physician."
    )


def _triage_prompt(case: dict[str, Any]) -> str:
    return (
        "You are an expert triage nurse in a rural clinic. Assess the urgency of the "
        "following patient case:\n\n"
        f"Symptoms: {case['symptoms']}\n"
        f"Age: {case.get('age', 'Unknown')}\n"
        "Return urgency one of EMERGENCY / URGENT / ROUTINE / SELF-CARE."
    )


# ---------------------------------------------------------------------------
# 单条评估
# ---------------------------------------------------------------------------
async def eval_one_case(case: dict[str, Any]) -> dict[str, Any]:
    diagnosis = await gemini_client.generate_structured(
        prompt=_diagnosis_prompt(case), response_model=DiagnosisAssessment
    )
    triage = await gemini_client.generate_structured(
        prompt=_triage_prompt(case), response_model=TriageResponse
    )

    predicted_dx = diagnosis.primary_diagnosis
    predicted_urgency = triage.urgency

    return {
        "id": case["id"],
        "symptoms": case["symptoms"],
        "expected_diagnoses": case["expected_diagnoses"],
        "predicted_diagnosis": predicted_dx,
        "diagnosis_correct": diagnosis_hit(predicted_dx, case["expected_diagnoses"]),
        "expected_urgency": case["expected_urgency"],
        "predicted_urgency": predicted_urgency,
        "urgency_correct": urgency_hit(predicted_urgency, case["expected_urgency"]),
        "confidence": round(float(getattr(diagnosis, "confidence", 0.0) or 0.0), 4),
    }


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def _build_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}
    dx_correct = sum(1 for r in results if r["diagnosis_correct"])
    urg_correct = sum(1 for r in results if r["urgency_correct"])
    conf = [r["confidence"] for r in results if r.get("confidence") is not None]
    return {
        "total": total,
        "diagnosis_accuracy": round(dx_correct / total, 4),
        "urgency_accuracy": round(urg_correct / total, 4),
        "mean_confidence": round(sum(conf) / len(conf), 4) if conf else None,
        "diagnosis_correct": dx_correct,
        "urgency_correct": urg_correct,
    }


def print_report(results: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    print("\n===== 诊断 Agent 效果评估报告 =====\n")
    for r in results:
        mark = "PASS" if r["diagnosis_correct"] else "FAIL"
        print(f"[{mark}] 案例#{r['id']}")
        print(f"    症状: {r['symptoms'][:60]}{'...' if len(r['symptoms']) > 60 else ''}")
        print(f"    期望诊断: {' / '.join(r['expected_diagnoses'])}")
        print(f"    预测诊断: {r['predicted_diagnosis']}")
        print(f"    分诊: 预测 {r['predicted_urgency']} vs 期望 {r['expected_urgency']} "
              f"{'✓' if r['urgency_correct'] else '✗'}  |  置信度 {r['confidence']}")
        print()
    print("----- 汇总指标 -----")
    print(f"诊断命中率 : {metrics.get('diagnosis_accuracy', 0) * 100:.1f}% "
          f"({metrics.get('diagnosis_correct', 0)}/{metrics.get('total', 0)})")
    print(f"分诊一致率 : {metrics.get('urgency_accuracy', 0) * 100:.1f}% "
          f"({metrics.get('urgency_correct', 0)}/{metrics.get('total', 0)})")
    print(f"平均置信度 : {metrics.get('mean_confidence')}")
    print()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"评测集不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"评测集为空或格式错误: {path}")
    return data


async def run(cases_path: str, out_path: str | None) -> None:
    cases = load_cases(Path(cases_path))

    if not _has_api_key():
        raise SystemExit("未配置 GOOGLE_API_KEY，无法调用模型。请先配置 backend/.env。")

    results = [await eval_one_case(c) for c in cases]
    metrics = _build_metrics(results)
    print_report(results, metrics)

    if out_path:
        out = Path(out_path)
        out.write_text(
            json.dumps({"metrics": metrics, "cases": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"报告已写入: {out.resolve()}")


def _has_api_key() -> bool:
    from app.core.config import settings

    provider = (settings.LLM_PROVIDER or "gemini").strip().lower()
    if provider in {"openai", "dashscope", "qwen", "tongyi", "openai_compatible", "openai-compatible"}:
        return bool(settings.LLM_API_KEY or os.environ.get("DASHSCOPE_API_KEY"))
    return bool(settings.GOOGLE_API_KEY)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线评估诊断 agent 效果")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="评测集 JSON 路径")
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径（可选）")
    parser.add_argument("--list-cases", action="store_true", help="仅列出评测集条目")
    args = parser.parse_args()

    if args.list_cases:
        for c in load_cases(Path(args.cases)):
            print(f"#{c['id']} [{c['expected_urgency']}] {c['symptoms']} -> "
                  f"{' / '.join(c['expected_diagnoses'])}")
        return

    asyncio.run(run(args.cases, args.out))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)