"""Combined review: run the existing FNO and MGN adapters once and report the 4 combinations.

  python run_review.py

Reuses fno_adapter.run and mgn_adapter.run as they are (no score or decision logic here).
Each run writes review_results/<run time>/ with report.html, summary.json and
the two adapter output folders.
A missing input or a failing adapter is shown as a failure for that model in THIS run;
past results are never read.
FNO and MGN Chamfer definitions differ, so the two models are not ranked or compared.
Does not train or replace any model. Standard library only;
report.html is a static file (no server, no scripts).
"""

from __future__ import annotations

import argparse
import html
import json
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import fno_adapter
import mgn_adapter

ROOT = Path(__file__).resolve().parent
MODES = ("onestep", "rollout")
ACTION_KO = {
    "KEEP_CURRENT": "현재 모델 유지",
    "RETRAIN_REQUEST": "재학습 요청",
    "WAIT_FOR_TRAINING_DATA": "학습 데이터 대기",
    "DATA_REVIEW": "데이터 재검토",
    "HOLD": "판단 보류",
    "APPROVE_CANDIDATE": "교체 승인 권고",
}
NO_RANKING = (
    "FNO와 MGN의 Chamfer 계산 방식이 다릅니다. "
    "두 모델의 점수로 우열이나 순위를 매기지 않으며, "
    "이 보고서의 표는 모델 간 비교표가 아닙니다."
)


def run_model(name, required_paths, call):
    """Run one adapter. Returns dict(status, reason, output_dir, result)."""
    missing = [str(p) for p in required_paths if not Path(p).exists()]
    if missing:
        return dict(
            status="failed",
            reason=f"입력 파일/폴더 없음: {', '.join(missing)}",
            output_dir=None,
            result=None,
        )
    try:
        out, _text, result = call()
    except Exception as exc:  # recorded, never hidden
        return dict(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            output_dir=None,
            result=None,
            traceback=traceback.format_exc(limit=5),
        )
    return dict(status="ok", reason=None, output_dir=str(out), result=result)


def exceed_by_metric(per_recipe):
    """Normalize per-recipe over-limit records.

    FNO keys them by metric; MGN holds the Chamfer entry directly.
    """
    if not per_recipe:
        return {}
    if "recipes_over_provisional_max" in per_recipe:
        return {"chamfer_um_recipe_mean": per_recipe}
    return per_recipe


def combo(model, mode, run_info, extra_unmeasured):
    base = dict(model_family=model, eval_mode=mode, split="validation")
    if run_info["status"] != "ok":
        return dict(base, status="failed", reason=run_info["reason"])
    r = run_info["result"]
    p, d, c = r["inputs"].get(mode), r["decisions"].get(mode), r["comparisons"].get(mode)
    if p is None:
        problems = r["inspection"]["problems"]
        decision = next(iter(r["decisions"].values()), None)
        return dict(
            base,
            status="failed",
            reason=f"{mode} 입력을 만들지 못함 (데이터 문제 {len(problems)}건)",
            problems=problems[:20],
            decision=decision,
        )
    exceed = exceed_by_metric((c or {}).get("per_recipe"))
    rows = []
    for row in (c or {}).get("rows", []):
        e = exceed.get(row["metric"])
        if row["metric"] == "invalid_geometry_count" and p.get("geometry"):
            g = p["geometry"]
            limit = row["provisional_max"]
            counts = Counter(x["rid"] for x in g["affected_frames"])
            ids = sorted(
                rid
                for rid, count in counts.items()
                if isinstance(limit, (int, float)) and not isinstance(limit, bool) and count > limit
            )
            e = dict(recipes_over_provisional_max=len(ids), rids=ids) if g["complete"] else None
        rows.append(
            dict(
                row,
                recipes_over=e["recipes_over_provisional_max"] if e else None,
                recipes_over_ids=e["rids"] if e else None,
            )
        )
    return dict(
        base,
        status="ok",
        model_version=p["model_version"],
        recipe_count=p["recipe_count"],
        evaluation_dataset_id=p["evaluation_dataset_id"],
        evaluation_contract_id=p["evaluation_contract_id"],
        data_origin=p["data_origin"],
        data_check=p["data_check"],
        metric_units=p["metric_units"],
        metric_aggregation=p["metric_aggregation"],
        metrics=p["metrics"],
        reported_not_judged=p.get("reported_not_judged"),
        geometry=p.get("geometry"),
        provisional=dict(
            label=(c or {}).get("label"),
            policy_version=(c or {}).get("policy_version"),
            policy_status=(c or {}).get("policy_status"),
            rows=rows,
        ),
        decision=dict(
            action=d["action"],
            reasons=d["reasons"],
            policy_version=d["policy_version"],
            policy_status=d["policy_status"],
            remaining_blockers_even_if_approved=d.get("remaining_blockers_even_if_approved", []),
            training_started=d["training_started"],
            active_model_changed=d["active_model_changed"],
        ),
        unmeasured=sorted(set(p.get("unmeasured") or p.get("not_provided") or [])),
        unmeasured_detail=extra_unmeasured,
        known_issues=p.get("known_issues", []),
    )


def model_block(name, run_info):
    block = dict(
        status=run_info["status"],
        reason=run_info["reason"],
        adapter_output_dir=run_info["output_dir"],
    )
    if run_info.get("traceback"):
        block["traceback"] = run_info["traceback"]
    if run_info["status"] == "ok":
        insp = run_info["result"]["inspection"]
        block["data_check"] = insp["data_check"]
        block["problem_counts"] = insp["problem_counts"]
        if name == "MGN":
            block["geometry"] = run_info["result"].get("geometry")
            block["unverified"] = run_info["result"]["unverified"]["items"]
            block["test_reference"] = {
                k: dict(
                    recipe_count=v["recipe_count"],
                    chamfer_um=v["chamfer_um"],
                    aggregation=v["aggregation"]["per_recipe"],
                    use=v.get("use"),
                )
                for k, v in run_info["result"]["summary"].items()
                if k.startswith("test/")
            }
    return block


def build_summary(started, args, fno, mgn):
    combos = []
    for model, info, extra in [
        ("FNO", fno, fno_adapter.UNMEASURED),
        ("MGN", mgn, mgn_adapter.NOT_PROVIDED),
    ]:
        for mode in MODES:  # fixed order; never sorted by score
            combos.append(combo(model, mode, info, extra))
    return dict(
        generated_at=started.isoformat(timespec="seconds"),
        notice=NO_RANKING,
        inputs=dict(
            fno_metrics=args.fno_metrics,
            fno_samples=args.fno_samples,
            fno_thresholds=args.fno_thresholds,
            mgn_input=args.mgn_input,
            mgn_thresholds=args.mgn_thresholds,
        ),
        chamfer_definitions=dict(
            FNO=fno_adapter.CHAMFER_DEFINITION,
            MGN=next(
                (
                    c["metric_aggregation"].get("chamfer_definition")
                    for c in combos
                    if c["model_family"] == "MGN" and c["status"] == "ok"
                ),
                None,
            ),
        ),
        models=dict(FNO=model_block("FNO", fno), MGN=model_block("MGN", mgn)),
        combinations=combos,
        all_decisions=[
            dict(
                model=c["model_family"],
                mode=c["eval_mode"],
                status=c["status"],
                action=(c.get("decision") or {}).get("action"),
            )
            for c in combos
        ],
        training_started=any((c.get("decision") or {}).get("training_started") for c in combos),
        active_model_changed=any(
            (c.get("decision") or {}).get("active_model_changed") for c in combos
        ),
    )


# ---------- HTML (static, escaped) ----------


def e(v):
    return html.escape("" if v is None else str(v))


def num(v):
    if v is None:
        return '<span class="na">미측정</span>'
    return e(f"{v:.4g}") if isinstance(v, float) else e(v)


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#5f5f5a;--line:#d9d8d2;--card:#ffffff;
--warn-bg:#fff4d6;--warn-fg:#6b4a00;--bad-bg:#fde7e7;--bad-fg:#8a1c1c;
--hold-bg:#e8eefc;--hold-fg:#1f3c88;--ok:#1d6b3a;--over:#a3361b}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a6a59d;
--line:#3a3935;--card:#1f1f1d;--warn-bg:#3a2f12;--warn-fg:#f3d48a;--bad-bg:#3d1b1b;
--bad-fg:#f4b4b4;--hold-bg:#1d2a4a;--hold-fg:#b9cbff;--ok:#7fd19b;--over:#ff9b7d}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.15rem;margin:32px 0 8px}
h3{font-size:1rem;margin:0 0 8px}.muted{color:var(--muted)}
.banner{padding:10px 14px;border-radius:8px;margin:12px 0}
.warn{background:var(--warn-bg);color:var(--warn-fg)}.bad{background:var(--bad-bg);color:var(--bad-fg)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:12px}
.grid>*{min-width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;
overflow-x:auto;min-width:0}
.card p,.card li,td,.banner{overflow-wrap:anywhere}
th:first-child,td:first-child{white-space:nowrap;overflow-wrap:normal}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;font-size:.85rem;
white-space:nowrap}
.HOLD{background:var(--hold-bg);color:var(--hold-fg)}
.failed,.DATA_REVIEW{background:var(--bad-bg);color:var(--bad-fg)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.na{color:var(--muted);font-style:italic}
.r-이내{color:var(--ok)}.r-초과{color:var(--over);font-weight:600}.r-미측정{color:var(--muted)}
ul{margin:4px 0 0;padding-left:20px}
code{font-size:.85em}
"""


def combo_card(c):
    head = f"<h3>{e(c['model_family'])} · {e(c['eval_mode'])} · {e(c['split'])}</h3>"
    if c["status"] != "ok":
        return (
            f'<div class="card">{head}<p><span class="pill failed">실행 실패</span></p>'
            f'<p>{e(c["reason"])}</p><p class="muted">이번 실행에서 결과를 만들지 못했습니다. '
            "과거 결과를 대신 표시하지 않습니다.</p></div>"
        )
    d = c["decision"]
    return (
        f'<div class="card">{head}<p><span class="pill {e(d["action"])}">{e(d["action"])} '
        f"({e(ACTION_KO.get(d['action'], ''))})</span></p>"
        f'<p class="muted">모델 {e(c["model_version"])}<br>'
        f"레시피 {e(c['recipe_count'])}개 · 평가표 점검 "
        f"{'통과' if c['data_check']['passed'] else '실패'}</p></div>"
    )


def combo_detail(c):
    title = f"{e(c['model_family'])} · {e(c['eval_mode'])} · {e(c['split'])}"
    if c["status"] != "ok":
        extra = ""
        if c.get("problems"):
            extra = (
                "<ul>"
                + "".join(f"<li>{e(p['kind'])}: {e(p['detail'])}</li>" for p in c["problems"])
                + "</ul>"
            )
        return f'<h2>{title}</h2><div class="banner bad">실행 실패: {e(c["reason"])}</div>{extra}'
    agg, d, pv, dc = c["metric_aggregation"], c["decision"], c["provisional"], c["data_check"]
    units = ", ".join(f"{k}={v}" for k, v in c["metric_units"].items())
    rows = "".join(
        f"<tr><td><code>{e(r['metric'])}</code></td><td>{e(r['unit'])}</td>"
        f'<td class="n">{num(r["observed"])}</td>'
        f'<td class="n">≤ {num(r["provisional_max"])}</td>'
        f'<td class="r-{e(r["result"])}">{e(r["result"])}</td>'
        f'<td class="n">{"" if r["recipes_over"] is None else e(r["recipes_over"])}</td>'
        f"<td>{e(', '.join(r['recipes_over_ids'] or []))}</td></tr>"
        for r in pv["rows"]
    )
    reported = ""
    if c.get("reported_not_judged"):
        reported = (
            '<p class="muted">기록만(비교 안 함): '
            + ", ".join(
                f"<code>{e(k)}</code> {num(v)}" for k, v in c["reported_not_judged"].items()
            )
            + "</p>"
        )
    blockers = "".join(f"<li>{e(b)}</li>" for b in d["remaining_blockers_even_if_approved"])
    unmeasured = "".join(
        f"<li><code>{e(k)}</code>: {e(c['unmeasured_detail'].get(k, ''))}</li>"
        for k in c["unmeasured"]
    )
    issues = "".join(f"<li>{e(x)}</li>" for x in c["known_issues"])
    return f"""<h2>{title}</h2>
<div class="card"><table>
<tr><th>모델 버전</th><td>{e(c["model_version"])}</td></tr>
<tr><th>데이터</th><td>{e(c["split"])} · 레시피 {e(c["recipe_count"])}개 ·
<code>{e(c["evaluation_dataset_id"])}</code> · 출처 <code>{e(c["data_origin"])}</code></td></tr>
<tr><th>평가 계약</th><td><code>{e(c["evaluation_contract_id"])}</code></td></tr>
<tr><th>레시피 값 집계</th><td>{e(agg.get("per_recipe"))}</td></tr>
<tr><th>레시피 간 집계</th><td>{e(agg.get("over_recipes"))}</td></tr>
<tr><th>Chamfer 정의</th><td>{e(agg.get("chamfer_definition"))}</td></tr>
<tr><th>단위</th><td>{e(units)}</td></tr>
<tr><th>평가표 점검</th><td>passed={e(dc["passed"])},
invalid_recipe_count={e(dc["invalid_recipe_count"])},
reference_verified={e(dc["reference_verified"])}</td></tr>
</table></div>
<h3 style="margin-top:14px">[임시 비교] 공식 판정 아님 ·
기준 {e(pv["policy_version"])} (status={e(pv["policy_status"])})</h3>
<div class="card"><table><tr><th>지표</th><th>단위</th><th>관측값</th><th>임시 상한</th>
<th>결과</th><th>상한 초과 레시피 수</th><th>초과 ID</th></tr>{rows}</table>{reported}</div>
<h3 style="margin-top:14px">[최종 판정]</h3>
<div class="card"><p><span class="pill {e(d["action"])}">{e(d["action"])}
({e(ACTION_KO.get(d["action"], ""))})</span>
 기준 {e(d["policy_version"])} · status={e(d["policy_status"])}</p>
<p>보류 이유: {e(" / ".join(d["reasons"]))}</p>
<p>기준을 승인해도 남는 보류 사유:</p><ul>{blockers or "<li>없음</li>"}</ul>
<p class="muted">학습 시작 {e(d["training_started"])} ·
모델 교체 {e(d["active_model_changed"])}</p></div>
<h3 style="margin-top:14px">[미측정]</h3><div class="card"><ul>{unmeasured or "<li>없음</li>"}</ul>
{("<p>제공된 제한 사항:</p><ul>" + issues + "</ul>") if issues else ""}</div>"""


def render_html(s):
    models = s["models"]
    status_rows = "".join(
        f'<tr><td>{e(m)}</td><td><span class="pill {"HOLD" if b["status"] == "ok" else "failed"}">'
        f"{'실행됨' if b['status'] == 'ok' else '실패'}</span></td>"
        f"<td>{e(b.get('reason') or '')}</td>"
        f"<td><code>{e(b.get('adapter_output_dir') or '-')}</code></td></tr>"
        for m, b in models.items()
    )
    failed = [m for m, b in models.items() if b["status"] != "ok"]
    fail_banner = (
        (
            f'<div class="banner bad">실패한 연결: {e(", ".join(failed))}. '
            "해당 조합은 이번 실행 결과가 없으며 "
            "과거 결과를 표시하지 않습니다.</div>"
        )
        if failed
        else ""
    )
    mgn = models["MGN"]
    selfx = ""
    if mgn["status"] == "ok":
        selfx = (
            '<div class="banner warn">MGN 제공 설명: '
            "교차(자기교차) 있는 레시피가 남아 있음 — "
            "Chamfer만으로 합격 판단 금지. "
            "자기교차 건수는 제공되지 않아 미측정입니다.</div>"
        )
        if mgn.get("geometry") is not None:
            selfx = (
                '<div class="banner warn">MGN 제공 설명: 교차(자기교차) 있는 레시피가 남아 있음. '
                "이번 실행에서 선분 교차·중첩·구조 이상을 검사했습니다. "
                "아래 형상 검사에서 프레임 수와 레시피 수를 구분해 확인하세요.</div>"
            )
    geometry_section = ""
    if mgn.get("geometry") is not None:
        geo = mgn["geometry"]
        geo_rows = ""
        for key, g in geo["groups"].items():
            role = "참고 기록만" if key.startswith("test/") else "validation 임시 비교에 반영"
            affected = ", ".join(g["affected_recipe_ids"])
            steps = ", ".join(f"{x['rid']}:t{x['step']:03d}" for x in g["affected_frames"])
            geo_rows += (
                f"<tr><td>{e(key)}<br>{e(role)}</td>"
                f"<td>{e(g['frames_checked'])}/{e(g['frames_expected'])}</td>"
                f"<td>{num(g['invalid_geometry_count'])}</td>"
                f"<td>{e(g['self_intersection_frame_count'])}</td>"
                f"<td>{e(g['overlap_frame_count'])}</td>"
                f"<td>{e(g['duplicate_edge_frame_count'])}</td>"
                f"<td>{e(g['affected_recipe_count'])}</td>"
                f"<td><details><summary>레시피·단계 보기</summary><p>{e(affected)}</p>"
                f"<p>{e(steps)}</p></details></td></tr>"
            )
        geometry_section = (
            '<h2>MGN 선분 형상 검사</h2><div class="card">'
            "<p>이상 건수는 문제가 있는 예측 프레임 수입니다. 교차와 중첩이 함께 있는 "
            "프레임도 한 번만 셉니다. 끝점 접촉은 별도 진단입니다. "
            "동일한 두 노드의 중복 연결은 출력 구조 문제로 분리하고 "
            "서로 다른 선분의 중첩에서는 제외합니다. "
            "물리 정확도와 정답 검증은 포함하지 않습니다.</p>"
            "<table><tr><th>구분·용도</th><th>검사 프레임</th><th>이상 프레임</th>"
            "<th>교차 프레임</th><th>서로 다른 선분 중첩</th><th>중복 연결 프레임</th>"
            "<th>이상 레시피 수</th><th>ID·단계</th></tr>"
            f"{geo_rows}</table><p>읽기 오류 {e(len(geo['errors']))}건. "
            "검사가 미완료인 집계는 미측정으로 남깁니다.</p></div>"
        )
    test_rows = ""
    for k, v in (mgn.get("test_reference") or {}).items():
        ch = v["chamfer_um"]
        test_rows += (
            f'<tr><td>{e(k)}</td><td class="n">{e(v["recipe_count"])}</td>'
            f'<td class="n">{num(ch["mean"])}</td>'
            f'<td class="n">{num(ch["median"])}</td>'
            f'<td class="n">{num(ch["max"])} ({e(ch.get("worst_rid"))})</td>'
            f"<td>{e(v['aggregation'])}</td></tr>"
        )
    test_section = (
        (
            "<h2>참고 정보: MGN test (validation과 분리)</h2>"
            '<div class="banner warn">test 점수는 참고 기록만 합니다. '
            "임시 기준 비교·임계치 조정·판정에 사용하지 않습니다.</div>"
            '<div class="card"><table><tr><th>구분</th>'
            "<th>레시피</th><th>Chamfer 평균 (um)</th><th>중앙값</th><th>최대</th>"
            "<th>레시피 값 집계</th></tr>"
            f"{test_rows}</table></div>"
        )
        if test_rows
        else ""
    )
    unverified = "".join(f"<li>{e(x)}</li>" for x in (mgn.get("unverified") or []))
    defs = "".join(
        f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in s["chamfer_definitions"].items()
    )
    cards = "".join(combo_card(c) for c in s["combinations"])
    details = "".join(combo_detail(c) for c in s["combinations"])
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FNO MGN 평가 현황</title><style>{CSS}</style></head>
<body><main>
<h1>FNO · MGN 평가 현황</h1>
<p class="muted">생성 {e(s["generated_at"])} · 이번 실행에서 읽은 실제 평가 점수만 표시 ·
학습 시작 {e(s["training_started"])} · 모델 교체 {e(s["active_model_changed"])}</p>
<div class="banner warn">{e(s["notice"])}</div>
{fail_banner}{selfx}
{geometry_section}
<h2>연결 상태</h2><div class="card"><table><tr><th>모델</th><th>상태</th>
<th>실패 이유</th><th>연결기 결과 폴더</th></tr>{status_rows}</table></div>
<h2>4개 조합 판정</h2><div class="grid">{cards}</div>
<h2>Chamfer 정의 (모델별로 다름)</h2><div class="card"><table>{defs}</table></div>
{details}
{test_section}
<h2>미확인 사항 (판정과 별개)</h2>
<div class="card"><ul>{unverified or "<li>MGN 결과 없음</li>"}</ul>
<p class="muted">임시 상한은 근거 없는 시작값이며 팀 승인 기준이 아닙니다.
판정은 보호 규칙에 따라 HOLD입니다.</p></div>
</main></body></html>
"""


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(args, output_root):
    started = datetime.now()
    out = Path(output_root) / started.strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    fno = run_model(
        "FNO",
        [args.fno_metrics, args.fno_samples, args.fno_thresholds],
        lambda: fno_adapter.run(
            args.fno_metrics, args.fno_samples, args.fno_thresholds, out / "fno"
        ),
    )
    mgn = run_model(
        "MGN",
        [args.mgn_input, args.mgn_thresholds],
        lambda: mgn_adapter.run(
            args.mgn_input, args.mgn_thresholds, out / "mgn", check_geometry=args.check_mgn_geometry
        ),
    )
    summary = build_summary(started, args, fno, mgn)
    if summary["training_started"] or summary["active_model_changed"]:
        raise AssertionError("The review must only decide and record.")
    write_json(out / "summary.json", summary)
    (out / "report.html").write_text(render_html(summary), encoding="utf-8")
    return out, summary


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fno-metrics", default=str(fno_adapter.DEFAULT_METRICS))
    p.add_argument("--fno-samples", default=str(fno_adapter.DEFAULT_SAMPLES))
    p.add_argument("--fno-thresholds", default=str(fno_adapter.DEFAULT_THRESHOLDS))
    p.add_argument("--mgn-input", default=str(mgn_adapter.DEFAULT_INPUT))
    p.add_argument("--mgn-thresholds", default=str(mgn_adapter.DEFAULT_THRESHOLDS))
    p.add_argument("--output-root", default=str(ROOT / "review_results"))
    p.add_argument(
        "--check-mgn-geometry",
        action="store_true",
        help="inspect MGN segment crossings/overlaps and record frame-level diagnostics",
    )
    return p


def main():
    args = parser().parse_args()
    out, s = run(args, args.output_root)
    for c in s["combinations"]:
        action = (c.get("decision") or {}).get("action")
        print(
            f"{c['model_family']:<4} {c['eval_mode']:<8} {c['split']:<10} "
            + (
                f"{action} ({ACTION_KO.get(action, '')})"
                if c["status"] == "ok"
                else f"실패: {c['reason']}"
            )
        )
    print("보고서:", out / "report.html")
    print("요약:", out / "summary.json")
    if any(b["status"] != "ok" for b in s["models"].values()):
        raise SystemExit(1)
    geometry = s["models"]["MGN"].get("geometry")
    if args.check_mgn_geometry and (geometry is None or geometry["errors"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
