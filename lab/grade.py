"""Answer grading: rule-based normalization first, LLM judge as fallback for near-misses."""
from __future__ import annotations
import re
from llm import chat, Meter

def _norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
def _has(hay, needle):
    """word-boundary containment on normalized strings ('no' must not match inside 'now')."""
    return bool(needle) and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", hay) is not None

def extract_answer(text: str) -> str:
    if not text: return ""
    m = re.findall(r"ANSWER\s*:\s*(.+)", text, flags=re.I|re.S)
    return (m[-1] if m else text).strip()

def rule_grade(answer_text: str, q) -> bool | None:
    """True/False when confident, None when the judge should decide."""
    a = _norm(extract_answer(answer_text))
    gold = q.answer
    if isinstance(gold, list):
        items = [_norm(x) for x in gold]
        if not gold: return ("none" in a or "no service" in a or a == "")
        present = all(_has(a, it) for it in items)
        # extra entities of the same shape as the gold items (service names, team names) count as wrong
        okset = set(items) | {_norm(x) for x in q.aliases}
        extra = [w for w in re.findall(r"[a-z0-9]+ (?:service|api|worker|engine|svc|core|gateway|daemon)\b", a) if w not in okset]
        if all(it in [t.lower() for t in TEAM_NAMES] for it in items):
            extra += [t for t in (x.lower() for x in TEAM_NAMES) if _has(a, t) and t not in items]
        if present and not extra: return True
        if not present: return False
        return None
    if isinstance(gold, (int, float)):
        nums = re.findall(r"-?\d+(?:\.\d+)?", a)
        if not nums: return False
        vals = {float(n) for n in nums}
        if len(vals)==1: return float(gold) in vals
        return float(gold) in vals and None
    golds = [_norm(gold)] + [_norm(x) for x in q.aliases]
    if any(_has(a, g) for g in golds):
        # a second, different service name in a single-entity answer is wrong unless it is a listed alias (ties)
        extra = [w for w in re.findall(r"[a-z0-9]+ (?:service|api|worker|engine|svc|core|gateway|daemon)\b", a) if w not in golds]
        return True if not extra else None
    if isinstance(gold, str) and gold.lower() in ("yes", "no") and _has(a, "yes" if gold.lower() == "no" else "no"): return False
    return None if len(a) > 80 else False

TEAM_NAMES = ["Payments","Identity","Search","Growth","Platform","Data","Messaging","Commerce"]

JUDGE_SYS = "You grade answers to factual questions. Reply with exactly CORRECT or INCORRECT. The answer is correct if it conveys the same fact(s) as the gold answer (ignore formatting, extra explanation, and alias names like 'foo-api' vs 'the Foo API'). For list answers, all gold items must be present and no wrong items added. For numeric answers, the number must match."

JUDGE_MODEL = "openai/gpt-5.6-luna"  # pinned so the judge does not change when the agent model does

def grade(answer_text: str, q, judge_model=JUDGE_MODEL) -> tuple[bool, str]:
    r = rule_grade(answer_text, q)
    if r is not None: return r, "rule"
    m = Meter()
    msg = chat([{"role":"system","content":JUDGE_SYS},{"role":"user","content":f"Question: {q.question}\nGold answer: {q.answer} (aliases: {q.aliases})\nCandidate answer: {extract_answer(answer_text)[:800]}"}], m, max_tokens=5, model=judge_model)
    return (msg.content or "").strip().upper().startswith("CORRECT"), "judge"
