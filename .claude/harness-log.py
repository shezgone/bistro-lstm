#!/usr/bin/env python3
"""하네스/루프 관찰용 로거. 훅이 stdin으로 넘기는 JSON을 받아
harness.log에 사람이 읽는 한 줄을 append 한다.
사용: harness-log.py <EVENT>   (EVENT = PRE | POST | SESSION 등)
"""
import sys, json, time, os

EVENT = sys.argv[1] if len(sys.argv) > 1 else "?"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness.log")

try:
    d = json.load(sys.stdin)
except Exception:
    d = {}

tool = d.get("tool_name", "-")
ti = d.get("tool_input", {}) or {}

# 툴 입력 요약: 의미 있는 키 우선
summary = ""
for k in ("command", "file_path", "pattern", "path", "skill", "subagent_type",
          "description", "url", "query", "prompt"):
    v = ti.get(k)
    if v:
        summary = f"{k}={v}"
        break
if not summary and ti:
    summary = json.dumps(ti, ensure_ascii=False)
summary = " ".join(str(summary).split())[:110]

# POST면 에러 여부 표시
extra = ""
if EVENT == "POST":
    resp = d.get("tool_response", d.get("tool_output", ""))
    is_err = False
    if isinstance(resp, dict):
        is_err = bool(resp.get("is_error") or resp.get("error"))
    extra = "  [ERROR]" if is_err else "  [ok]"

ts = time.strftime("%H:%M:%S")
line = f"{ts}  {EVENT:4}  {str(tool):14}  {summary}{extra}\n"

try:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)
except Exception:
    pass

# 훅 흐름을 막지 않도록 항상 정상 종료
sys.exit(0)
