#!/usr/bin/env python3
"""Build a single-file HTML dashboard from lightspeed-eval result folders.

Usage:
    python3 build_dashboard.py <results_dir> [-o dashboard.html]

Every nested run (<results_dir>/eval_*/<agent>/run_N/evaluation_*_detailed.csv)
becomes one column in the dashboard. If no nested runs exist, flat
evaluation_*_detailed.csv files in <results_dir> are used instead.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Optional

PERF_FIELDS = (
    "time_to_first_token",
    "streaming_duration",
    "tokens_per_second",
    "agent_latency",
    "api_input_tokens",
    "api_output_tokens",
)


def to_float(value: str) -> Optional[float]:
    """Convert a CSV cell to float, or None when empty or invalid."""
    try:
        return float(value) if value not in ("", None) else None
    except ValueError:
        return None


def find_csvs(root: Path) -> list[tuple[str, Path]]:
    """Return (label, path) for each run CSV under root."""
    nested = sorted(root.glob("eval_*/*/run_*/evaluation_*_detailed.csv"))
    if nested:
        return [
            (f"{p.parents[2].name}/{p.parents[1].name}/{p.parent.name}", p)
            for p in nested
        ]
    flat = sorted(root.glob("evaluation_*_detailed.csv"))
    return [(re.sub(r"_detailed\.csv$", "", p.name), p) for p in flat]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file, returning an empty dict when it is missing or invalid."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def parse_judge_scores(text: str) -> list[dict[str, Any]]:
    """Parse the CSV judge_scores column (a JSON array) into a list; empty if absent."""
    try:
        value = json.loads(text) if text else []
    except ValueError:
        return []
    return value if isinstance(value, list) else []


def build_tags(configuration: dict[str, Any], meta: dict[str, Any]) -> dict[str, str]:
    """Derive the comparable config dimensions (x-axis labels and constants filter)."""
    agents = configuration.get("agents", {}).get("agents", {})
    default = configuration.get("agents", {}).get("default", {}).get("agent")
    name = (default[0] if isinstance(default, list) and default else None) or next(
        iter(agents), None
    )
    agent = agents.get(name, {}) if name else {}
    panel = configuration.get("judge_panel", {})
    pool = configuration.get("llm_pool", {}).get("models", {})
    judge_models = [
        pool.get(j, {}).get("model") or j for j in panel.get("judges", [])
    ]
    judge_label = " + ".join(judge_models) if judge_models else None
    dims = {
        "name": meta.get("name"),
        "model": meta.get("backend_model") or "unknown (no run_meta.json)",
        "instance": meta.get("aws_instance"),
        "cores": f"{meta['cpu_cores']} cores" if "cpu_cores" in meta else None,
        "thinking": (
            None if "thinking" not in meta else ("thinking on" if meta["thinking"] else "thinking off")
        ),
        "dataset": meta.get("dataset"),
        "judge": judge_label,
    }
    return {
        k: ("n/a" if v is None else str(v))
        for k, v in dims.items()
        if k not in ("name", "thinking", "dataset") or v is not None
    }


def load_run(label: str, path: Path) -> dict[str, Any]:
    """Load one detailed CSV into a run record."""
    summary_path = path.with_name(path.name.replace("_detailed.csv", "_summary.json"))
    summary = load_json(summary_path)
    configuration = summary.get("configuration", {})
    json_judge_scores = {
        (r.get("conversation_group_id"), r.get("turn_id"), r.get("metric_identifier")): r.get(
            "judge_scores"
        )
        for r in summary.get("results", [])
    }
    meta = load_json(path.parent / "run_meta.json")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    scored = [
        {
            "conversation": r["conversation_group_id"],
            "turn": r["turn_id"],
            "metric": r["metric_identifier"],
            "result": r["result"],
            "score": to_float(r["score"]),
            "threshold": to_float(r["threshold"]),
            "reason": r["reason"],
            "query": r["query"],
            "judge_scores": parse_judge_scores(r.get("judge_scores", ""))
            or (
                json_judge_scores.get(
                    (r["conversation_group_id"], r["turn_id"], r["metric_identifier"])
                )
                or []
            ),
            "response": r["response"],
            "expected": r["expected_response"],
        }
        for r in rows
    ]

    turns: dict[tuple[str, str], dict[str, Optional[float]]] = {}
    for r in rows:
        key = (r["conversation_group_id"], r["turn_id"])
        turns.setdefault(key, {f: to_float(r.get(f, "")) for f in PERF_FIELDS})
    perf = [
        {"conversation": k[0], "turn": k[1], **values} for k, values in turns.items()
    ]
    return {
        "label": label,
        "tags": build_tags(configuration, meta),
        "meta": meta,
        "configuration": configuration,
        "scored": scored,
        "perf": perf,
    }


def percentile(values: list[float], pct: float) -> Optional[float]:
    """Return a nearest-rank percentile, or None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def judge_view(run: dict[str, Any]) -> dict[str, Any]:
    """Score and pass rate for each judge and for each way of combining the judges.

    Built from per-judge scores. Runs without them (single-judge, older runs) use the
    recorded score as the one judge's score. Judges that errored are left out.
    """
    pool = (run["configuration"].get("llm_pool") or {}).get("models") or {}
    models = {alias: (entry or {}).get("model") or alias for alias, entry in pool.items()}
    configured = (run["configuration"].get("judge_panel") or {}).get("judges") or []
    fallback = models.get(configured[0], configured[0]) if configured else "judge"
    per_judge: dict[str, list[tuple[float, Optional[float]]]] = {}
    combined: dict[str, list[tuple[float, Optional[bool]]]] = {
        "average": [], "max": [], "majority": []
    }
    for row in run["scored"]:
        entries = row["judge_scores"] or [{"judge_id": None, "score": row["score"]}]
        limit = row["threshold"]
        scores: list[float] = []
        for entry in entries:
            score = entry.get("score")
            if score is None:
                continue
            judge_id = entry.get("judge_id")
            model = fallback if judge_id is None else models.get(judge_id, judge_id)
            per_judge.setdefault(model, []).append((score, limit))
            scores.append(score)
        if not scores:
            continue
        mean_score = statistics.fmean(scores)
        top = max(scores)
        majority = None if limit is None else sum(x >= limit for x in scores) > len(scores) / 2
        combined["average"].append((mean_score, None if limit is None else mean_score >= limit))
        combined["max"].append((top, None if limit is None else top >= limit))
        combined["majority"].append((mean_score, majority))

    def stats(pairs: list[tuple[float, Optional[bool]]]) -> dict[str, Any]:
        if not pairs:
            return {"score": None, "pass_rate": None, "n": 0}
        verdicts = [ok for _, ok in pairs if ok is not None]
        return {
            "score": statistics.fmean(score for score, _ in pairs),
            "pass_rate": 100 * sum(verdicts) / len(verdicts) if verdicts else None,
            "n": len(pairs),
        }

    each = {
        model: stats([(score, None if limit is None else score >= limit) for score, limit in pairs])
        for model, pairs in per_judge.items()
    }
    return {"each": each, **{name: stats(pairs) for name, pairs in combined.items()}}


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    """Compute headline numbers for one run."""
    scored = run["scored"]
    passed = sum(1 for r in scored if r["result"] == "PASS")
    scores = [r["score"] for r in scored if r["score"] is not None]
    summary: dict[str, Any] = {
        "label": run["label"],
        "tags": run["tags"],
        "turns": len(run["perf"]),
        "evaluations": len(scored),
        "pass_rate": 100 * passed / len(scored) if scored else None,
        "errors": sum(1 for r in scored if r["result"] == "ERROR"),
        "score_mean": statistics.fmean(scores) if scores else None,
        "judge_view": judge_view(run),
    }
    for field in PERF_FIELDS:
        values = [p[field] for p in run["perf"] if p[field] is not None]
        summary[field] = {
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "sum": sum(values) if values else None,
        }
    return summary


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hybrid Eval Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{--bg:#000;--card:#151515;--line:#2e2e2e;--fg:#f2f2f2;--muted:#a3a3a3;--red:#ee0000;--redhi:#ff3b3b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 'Red Hat Text','Red Hat Display',system-ui,sans-serif}
header{padding:18px 20px 0;border-top:4px solid var(--red)}
.title{font-size:22px;font-weight:700;letter-spacing:.2px}
.title b{color:var(--red)}
.subtitle{font-size:12px;color:var(--muted);margin:2px 0 12px}
nav{display:flex;flex-wrap:wrap;border-bottom:1px solid var(--line);padding:0 12px}
nav button{background:none;border:none;border-bottom:3px solid transparent;color:var(--muted);padding:10px 14px;cursor:pointer;font:inherit;font-size:13px}
nav button:hover{color:var(--fg)}
nav button.active{color:#fff;border-bottom-color:var(--red);font-weight:600}
main{padding:16px 20px 40px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:6px;margin-bottom:16px}
.panel h4{margin:0;padding:8px 12px;border-bottom:1px solid var(--line);font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.consts{display:flex;flex-wrap:wrap;gap:10px 16px;padding:12px;align-items:flex-end}
.consts label{display:flex;flex-direction:column;gap:3px;font-size:11px;color:var(--muted)}
select,button.btn{background:#0b0b0b;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:5px 8px;font:inherit;font-size:12px}
.consts select{min-width:140px}
select.pinned{background:var(--red);border-color:var(--red);color:#fff}
button.btn{cursor:pointer}button.btn:hover{border-color:var(--red)}
.count{padding:0 12px 10px;font-size:11px;color:var(--muted)}
.chk{flex-direction:row!important;align-items:center;gap:6px!important;color:var(--fg)!important;font-size:12px!important;cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:14px}
.chartcard{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px}
.chartcard .plot{height:400px}
.big{height:560px}
.desc{padding:12px 14px;border-left:3px solid var(--red);background:#0d0d0d;margin-bottom:12px;font-size:13px}
.desc h3{margin:0 0 4px;font-size:15px}
.subtabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.subtabs button{background:#0b0b0b;color:var(--muted);border:1px solid var(--line);border-radius:14px;padding:4px 12px;cursor:pointer;font:inherit;font-size:12px}
.subtabs button.active{background:var(--red);border-color:var(--red);color:#fff}
.hint{font-size:11px;color:var(--muted);margin:8px 2px 0}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;vertical-align:top}
th{color:var(--muted);font-weight:600;background:#0d0d0d}
th:first-child,td:first-child,td.l,th.l{text-align:left}
.pass{color:#6fdc8c;font-weight:600}.fail{color:var(--redhi);font-weight:700}
.scroll{overflow-x:auto}
a{color:var(--redhi)}
h3.sec{font-size:14px;margin:22px 0 8px}
dialog{background:var(--card);color:var(--fg);border:1px solid var(--red);border-radius:8px;width:min(900px,94vw);max-height:88vh;padding:16px}
dialog::backdrop{background:rgba(0,0,0,.75)}
dialog h2{margin-top:0}dialog h3{color:var(--redhi);font-size:13px;text-transform:uppercase;letter-spacing:.6px}
details{margin-left:14px}summary{cursor:pointer;color:var(--muted)}
.kv{display:grid;grid-template-columns:minmax(120px,max-content) 1fr;gap:2px 12px;margin-left:14px;font-size:12px}
.kv span:nth-child(odd){color:var(--muted)}.kv span:nth-child(even){word-break:break-all}
.txt{white-space:pre-wrap;background:#0d0d0d;border:1px solid var(--line);border-radius:4px;padding:10px;font-size:13px;max-height:260px;overflow:auto}
.metadata{padding:12px;font-size:11px;color:var(--muted);border-top:1px solid var(--line);margin-top:24px}
.metadata pre{margin:6px 0 0;white-space:pre-wrap}
</style>
</head>
<body>
<header>
  <div class="title">Hybrid Eval <b>Dashboard</b></div>
  <div class="subtitle" id="sub"></div>
</header>
<nav id="nav"></nav>
<main>
  <div class="panel">
    <h4>Constants: pin a parameter to compare only runs that share it (pinned parameters leave the x-axis labels)</h4>
    <div class="consts" id="consts"></div>
    <div class="count" id="count"></div>
    <div class="consts" id="display">
      <label class="chk"><input type="checkbox" id="showId"> Add the run ID (timestamp) to the x-axis labels</label>
    </div>
    <div class="consts" id="judgeview">
      <b>Judge view (quality charts):</b>
      <label class="chk"><input type="radio" name="jv" value="panel" checked> As recorded by the eval</label>
      <label class="chk"><input type="radio" name="jv" value="average"> Average of the judges</label>
      <label class="chk"><input type="radio" name="jv" value="max"> Max of the judges</label>
      <label class="chk"><input type="radio" name="jv" value="majority"> Majority vote</label>
      <label class="chk"><input type="radio" name="jv" value="each"> Each judge separately</label>
    </div>
  </div>
  <div id="page"></div>
  <div class="metadata"><b>Generation metadata</b><pre id="meta"></pre></div>
</main>
<dialog id="dlg"><div style="float:right"><button class="btn" id="close">Close</button></div><div id="dlgbody"></div></dialog>

<script>
const DATA = __DATA__;
const S=DATA.summaries, R=DATA.runs, NL=String.fromCharCode(10);
const fmt=(v,d=2)=>v==null?"-":Number(v).toFixed(d);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const DIMS=[["name","Run name"],["model","Served model"],["instance","AWS instance"],["cores","CPU cores"],["thinking","Thinking"],["dataset","Dataset"],["judge","Judge"]].filter(([k])=>S.some(s=>k in s.tags));
const nm=i=>S[i].tags.name||S[i].tags.model;
const C={fg:"#f2f2f2",mut:"#a3a3a3",grid:"#2b2b2b",red:"#ee0000"};
const PASTELS=["#a8d8ea","#ffd6a5","#caffbf","#e0bbf5","#ffadad","#fdffb6","#9bf6ff","#ffc6ff","#b9fbc0","#cfd8ff"];
document.getElementById("sub").textContent=R.length+" run(s) from "+DATA.source;
document.getElementById("meta").textContent=["Generated: "+DATA.generated,"Source: "+DATA.source,"Runs: "+R.length,""].concat(R.map(r=>r.label)).join(NL);

const pinned={};
const consts=document.getElementById("consts");
consts.innerHTML=DIMS.map(([k,name])=>{
  const vals=[...new Set(S.map(s=>s.tags[k]))].sort();
  return `<label>${name}<select data-dim="${k}"><option value="">(vary)</option>${vals.map(v=>`<option>${esc(v)}</option>`).join("")}</select></label>`;
}).join("")+`<button class="btn" id="clear">Clear all</button>`;
consts.addEventListener("change",e=>{
  const k=e.target.dataset.dim; if(!k) return;
  if(e.target.value) pinned[k]=e.target.value; else delete pinned[k];
  e.target.classList.toggle("pinned",!!e.target.value);
  refresh();
});
document.getElementById("clear").onclick=()=>{
  for(const k in pinned) delete pinned[k];
  consts.querySelectorAll("select").forEach(x=>{x.value="";x.classList.remove("pinned")});
  refresh();
};
let visible=[];
function applyFilter(){
  visible=S.map((s,i)=>i).filter(i=>Object.entries(pinned).every(([k,v])=>S[i].tags[k]===v));
  const pins=Object.entries(pinned).map(([k,v])=>k+"="+v).join(", ");
  document.getElementById("count").textContent=visible.length+" of "+S.length+" runs shown"+(pins?" - pinned: "+pins:"");
}

const T=(f,st)=>s=>s[f][st];
const TABS=[
 {id:"quality",title:"Quality",desc:"Higher is better. Judge-scored answer quality across the questions in each run.",
  subs:[["Pass rate (%)",s=>s.pass_rate,"%",1,"pass_rate"],["Mean score",s=>s.score_mean,"",2,"score"]]},
 {id:"ttft",title:"TTFT",desc:"Lower is better. Client-side time from request start until the first streamed token (includes routing and network).",
  subs:[["Mean",T("time_to_first_token","mean"),"s",2],["Median",T("time_to_first_token","p50"),"s",2],["P95",T("time_to_first_token","p95"),"s",2]]},
 {id:"tps",title:"Throughput",desc:"Higher is better. Output tokens per second during generation, excluding time to first token.",
  subs:[["Mean",T("tokens_per_second","mean"),"tok/s",1],["Median",T("tokens_per_second","p50"),"tok/s",1],["P95",T("tokens_per_second","p95"),"tok/s",1]]},
 {id:"lat",title:"Request Latency",desc:"Lower is better. Total time for each request as measured by the eval client.",
  subs:[["Mean",T("agent_latency","mean"),"s",2],["Median",T("agent_latency","p50"),"s",2],["P95",T("agent_latency","p95"),"s",2]]},
 {id:"dur",title:"Streaming Duration",desc:"Lower is better. Time from request start to end of the stream.",
  subs:[["Mean",T("streaming_duration","mean"),"s",2],["Median",T("streaming_duration","p50"),"s",2],["P95",T("streaming_duration","p95"),"s",2]]},
 {id:"tok",title:"Tokens",desc:"Output and input token counts reported by OLS per turn.",
  subs:[["Output (mean)",T("api_output_tokens","mean"),"tokens",0],["Input (mean)",T("api_input_tokens","mean"),"tokens",0],["Output (total)",T("api_output_tokens","sum"),"tokens",0]]},
];
const HOME=[
 ["Pass rate (%)",s=>s.pass_rate,"%",1,"pass_rate"],
 ["Mean quality score",s=>s.score_mean,"",2,"score"],
 ["TTFT mean (s)",T("time_to_first_token","mean"),"s",2],
 ["TTFT P95 (s)",T("time_to_first_token","p95"),"s",2],
 ["Throughput mean (tok/s)",T("tokens_per_second","mean"),"tok/s",1],
 ["Request latency mean (s)",T("agent_latency","mean"),"s",2],
 ["Output tokens per turn (mean)",T("api_output_tokens","mean"),"tokens",0],
];
const PAGES=[{id:"home",title:"Home"}].concat(TABS.map(t=>({id:t.id,title:t.title}))).concat([{id:"runs",title:"Runs"}]);
let page="home",sub=0;
const nav=document.getElementById("nav");
nav.innerHTML=PAGES.map(p=>`<button data-id="${p.id}">${p.title}</button>`).join("");
nav.onclick=e=>{if(e.target.dataset.id){page=e.target.dataset.id;sub=0;render();}};

const show={id:false};
function xLabel(i){
  const t=S[i].tags;
  const base=t.name||t.model||("#"+(visible.indexOf(i)+1));
  return show.id?base+" | "+S[i].label.split("/")[0].replace("eval_",""):base;
}
document.getElementById("showId").onchange=e=>{show.id=e.target.checked;render();};
let judgeView="panel";
document.querySelectorAll('input[name="jv"]').forEach(r=>{r.onchange=e=>{judgeView=e.target.value;render();};});
function leftMargin(){
  const first=xLabel(visible[0]).replace(/<[^>]+>/g,"").length;
  return Math.max(56,Math.min(240,Math.round(first*4.3)-30));
}
function baseLayout(title,unit,bottom){
  const axis={gridcolor:C.grid,zerolinecolor:C.grid,linecolor:C.grid,tickfont:{color:C.mut}};
  return {paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",
    font:{family:"Red Hat Text, system-ui, sans-serif",size:11,color:C.fg},
    title:{text:title,font:{size:13},x:0.02,xanchor:"left"},
    margin:{l:leftMargin(),r:16,t:44,b:bottom},dragmode:"zoom",bargap:0.4,
    hoverlabel:{bgcolor:"#111",bordercolor:C.red,font:{color:C.fg}},
    xaxis:Object.assign({},axis,{tickmode:"array",tickvals:visible.map((_,n)=>n),ticktext:visible.map(xLabel),
      tickangle:-40,showgrid:false,zeroline:false,automargin:true,tickfont:{color:C.fg,size:12}}),
    shapes:visible.slice(0,-1).map((_,n)=>({type:"line",xref:"x",yref:"paper",x0:n+0.5,x1:n+0.5,y0:-0.05,y1:1,
      line:{color:"#6a6a6a",width:1,dash:"dot"}})),
    yaxis:Object.assign({rangemode:"tozero",title:{text:unit,font:{color:C.mut}}},axis)};
}
function labelMargin(){
  const longest=Math.max(1,...visible.map(i=>xLabel(i).replace(/<[^>]+>/g,"").length));
  return Math.min(380,30+Math.round(longest*3.9));
}
function finishPlot(divId,traces,layout){
  const el=document.getElementById(divId);
  const height=Math.max(divId==="one"?560:400,320+layout.margin.b);
  el.style.height=height+"px";
  layout.height=height;
  Plotly.react(el,traces,layout,{responsive:true,displaylogo:false,scrollZoom:true});
  if(el.removeAllListeners) el.removeAllListeners("plotly_click");
  el.on("plotly_click",ev=>openRun(visible[ev.points[0].pointIndex]));
}
function drawBar(divId,title,get,unit,d){
  const ys=visible.map(i=>get(S[i]));
  const trace={type:"bar",x:visible.map((_,n)=>n),y:ys,
    text:ys.map(v=>v==null?"n/a":fmt(v,d)),textposition:"outside",cliponaxis:false,textfont:{color:C.fg},
    marker:{color:visible.map(i=>PASTELS[i%PASTELS.length])},
    customdata:visible.map(i=>[S[i].label,Object.entries(S[i].tags).map(([k,v])=>k+": "+v).join("<br>")]),
    hovertemplate:"%{customdata[1]}<br><b>"+title+": %{y:."+d+"f} "+unit+"</b><br>%{customdata[0]}<extra></extra>"};
  finishPlot(divId,[trace],baseLayout(title,unit,labelMargin()));
}
function drawJudgeBars(divId,title,q,unit,d){
  const models=[...new Set(visible.flatMap(i=>Object.keys(S[i].judge_view.each)))].sort();
  const traces=models.map((m,j)=>{
    const ys=visible.map(i=>{const e=S[i].judge_view.each[m];return e?e[q]:null;});
    return {type:"bar",name:m,x:visible.map((_,n)=>n),y:ys,
      text:ys.map(v=>v==null?"":fmt(v,d)),textposition:"outside",cliponaxis:false,textfont:{color:C.fg},
      marker:{color:PASTELS[j%PASTELS.length]},
      customdata:visible.map(i=>[S[i].label,Object.entries(S[i].tags).map(([k,v])=>k+": "+v).join("<br>")]),
      hovertemplate:"%{customdata[1]}<br><b>"+m+" - "+title+": %{y:."+d+"f} "+unit+"</b><br>%{customdata[0]}<extra></extra>"};});
  const layout=Object.assign(baseLayout(title,unit,labelMargin()),{barmode:"group",bargap:0.25,showlegend:true,
    legend:{orientation:"h",y:1.14,x:0,font:{color:C.fg}}});
  layout.margin.t=70;
  finishPlot(divId,traces,layout);
}
const AGG_LABEL={average:"average of judges",max:"max of judges",majority:"majority vote"};
function plotItem(divId,title,get,unit,d,q){
  if(q&&AGG_LABEL[judgeView]) return drawBar(divId,title+" - "+AGG_LABEL[judgeView],s=>s.judge_view[judgeView][q],unit,d);
  if(q&&judgeView==="each") return drawJudgeBars(divId,title+" - each judge",q,unit,d);
  return drawBar(divId,title,get,unit,d);
}

function render(){
  nav.querySelectorAll("button").forEach(b=>b.classList.toggle("active",b.dataset.id===page));
  const el=document.getElementById("page");
  if(!visible.length){el.innerHTML='<div class="desc">No runs match the pinned constants.</div>';return;}
  if(page==="home"){
    el.innerHTML='<div class="grid">'+HOME.map((_,i)=>`<div class="chartcard"><div class="plot" id="h${i}"></div></div>`).join("")+
      '</div><div class="hint">All metrics at a glance. Drag a chart to zoom, double-click to reset, click a bar for every parameter of that run. Use the tabs above for one metric at a time.</div>';
    HOME.forEach((h,i)=>plotItem("h"+i,...h));
  } else if(page==="runs"){
    renderRuns(el);
  } else {
    const t=TABS.find(x=>x.id===page),[name,get,unit,d,q]=t.subs[sub];
    el.innerHTML=`<div class="desc"><h3>${t.title}: ${name}</h3>${t.desc}</div>
      <div class="subtabs">${t.subs.map((s,i)=>`<button data-i="${i}" class="${i===sub?"active":""}">${s[0]}</button>`).join("")}</div>
      <div class="chartcard"><div class="plot big" id="one"></div></div>
      <div class="hint">Drag on the chart to zoom into a region, double-click to reset, use the toolbar to pan. Click a bar for every parameter of that run.</div>`;
    el.querySelector(".subtabs").onclick=e=>{if(e.target.dataset.i!=null){sub=+e.target.dataset.i;render();}};
    plotItem("one",t.title+": "+name,get,unit,d,q);
  }
}

function renderRuns(el){
  el.innerHTML=`<div class="scroll panel"><table id="summary"></table></div>
    <h3 class="sec">Per-question detail <select id="runpick"></select></h3>
    <div class="scroll panel"><table id="detail"></table></div>`;
  document.getElementById("summary").innerHTML=
   "<tr><th>Run</th><th class='l'>Name</th><th class='l'>Model</th><th class='l'>Judge</th><th>Turns</th><th>Pass %</th><th>Errors</th><th>Mean score</th>"+
   "<th>TTFT mean (s)</th><th>TTFT p95 (s)</th><th>Tok/s mean</th><th>Tok/s p50</th><th>Latency mean (s)</th><th>Out tokens</th></tr>"+
   visible.map(i=>{const s=S[i];return `<tr><td><a href="#" onclick="openRun(${i});return false">${esc(s.label)}</a></td>
   <td class="l">${esc(s.tags.name||"")}</td><td class="l">${esc(s.tags.model)}</td><td class="l">${esc(s.tags.judge)}</td><td>${s.turns}</td><td>${fmt(s.pass_rate,1)}</td>
   <td>${s.errors}</td><td>${fmt(s.score_mean)}</td>
   <td>${fmt(s.time_to_first_token.mean)}</td><td>${fmt(s.time_to_first_token.p95)}</td>
   <td>${fmt(s.tokens_per_second.mean,1)}</td><td>${fmt(s.tokens_per_second.p50,1)}</td>
   <td>${fmt(s.agent_latency.mean)}</td><td>${fmt(s.api_output_tokens.sum,0)}</td></tr>`}).join("");
  const pick=document.getElementById("runpick");
  pick.innerHTML=visible.map(i=>`<option value="${i}">${esc(nm(i))} - ${esc(S[i].label)}</option>`).join("");
  pick.onchange=()=>renderDetail(pick.value);
  renderDetail(pick.value);
}
function renderDetail(i){
  const run=R[i],box=document.getElementById("detail"); if(!run||!box) return;
  const perf={};run.perf.forEach(p=>perf[p.conversation+"|"+p.turn]=p);
  box.innerHTML=
   "<tr><th>Conv / turn</th><th class='l'>Metric</th><th>Result</th><th>Score</th><th>TTFT (s)</th>"+
   "<th>Tok/s</th><th>Latency (s)</th><th>Out tok</th><th class='l'>Query</th></tr>"+
   run.scored.map((r,n)=>{const p=perf[r.conversation+"|"+r.turn]||{};
    return `<tr><td>${esc(r.conversation)} / ${esc(r.turn)}</td><td class="l">${esc(r.metric)}</td>
    <td class="${r.result==="PASS"?"pass":"fail"}">${esc(r.result)}</td><td>${fmt(r.score)}</td>
    <td>${fmt(p.time_to_first_token)}</td><td>${fmt(p.tokens_per_second,1)}</td>
    <td>${fmt(p.agent_latency)}</td><td>${fmt(p.api_output_tokens,0)}</td>
    <td class="l"><a href="#" onclick="openTurn(${i},${n});return false">${esc((r.query||"").slice(0,90))}</a></td></tr>`}).join("");
}
function openTurn(i,n){
  const r=R[i].scored[n],p=R[i].perf.find(x=>x.conversation===r.conversation&&x.turn===r.turn)||{};
  document.getElementById("dlgbody").innerHTML=
   `<h2>${esc(r.conversation)} / ${esc(r.turn)}</h2>
    <div style="color:var(--muted)">${esc(nm(i))} - ${esc(R[i].label)}</div>
    <h3>Query</h3><div class="txt">${esc(r.query)}</div>
    <h3>Response</h3><div class="txt">${esc(r.response)||"(empty)"}</div>
    <h3>Expected response</h3><div class="txt">${esc(r.expected)||"(none)"}</div>
    <h3>Judge: ${esc(r.metric)} - <span class="${r.result==="PASS"?"pass":"fail"}">${esc(r.result)}</span> (score ${fmt(r.score)}, threshold ${fmt(r.threshold)})</h3>
    <div class="txt">${esc(r.reason)}</div>
    ${(r.judge_scores&&r.judge_scores.length>1)?`<h3>Each judge</h3><div class="scroll"><table><tr><th>Judge</th><th>Score</th><th class="l">Reason</th></tr>${r.judge_scores.map(j=>`<tr><td>${esc(j.judge_id)}</td><td>${fmt(j.score)}</td><td class="l">${esc(j.reason)}</td></tr>`).join("")}</table></div>`:""}
    <h3>Performance</h3><div class="kv"><span>TTFT (s)</span><span>${fmt(p.time_to_first_token)}</span><span>Tokens/s</span><span>${fmt(p.tokens_per_second,1)}</span><span>Latency (s)</span><span>${fmt(p.agent_latency)}</span><span>Input / output tokens</span><span>${fmt(p.api_input_tokens,0)} / ${fmt(p.api_output_tokens,0)}</span></div>`;
  document.getElementById("dlg").showModal();
}

function tree(v){
  if(v===null||typeof v!=="object") return `<span>${esc(JSON.stringify(v))}</span>`;
  const entries=Object.entries(v);
  if(!entries.length) return `<span>${Array.isArray(v)?"[]":"{}"}</span>`;
  const scalars=entries.filter(([,x])=>x===null||typeof x!=="object");
  const nested=entries.filter(([,x])=>x!==null&&typeof x==="object");
  return (scalars.length?`<div class="kv">${scalars.map(([k,x])=>`<span>${esc(k)}</span><span>${esc(JSON.stringify(x))}</span>`).join("")}</div>`:"")+
    nested.map(([k,x])=>`<details open><summary>${esc(k)}</summary>${tree(x)}</details>`).join("");
}
function openRun(i){
  const r=R[i],s=S[i];
  document.getElementById("dlgbody").innerHTML=
   `<h2>${esc(nm(i))}</h2><div style="color:var(--muted)">${esc(r.label)}</div>
    <h3>Config dimensions</h3>${tree(s.tags)}
    <h3>Run metadata</h3>${Object.keys(r.meta).length?tree(r.meta):'<div style="margin-left:14px;color:var(--muted)">None. Add a run_meta.json next to the run CSV (e.g. {"name": "...", "backend_model": "...", "aws_instance": "...", "cpu_cores": 28}).</div>'}
    <h3>All configuration parameters</h3>${Object.keys(r.configuration).length?tree(r.configuration):'<div style="color:var(--muted)">No summary JSON found for this run.</div>'}`;
  document.getElementById("dlg").showModal();
}
document.getElementById("close").onclick=()=>document.getElementById("dlg").close();

function refresh(){applyFilter();render();}
refresh();
</script>
</body>
</html>
"""


def main() -> None:
    """Parse arguments, load runs, and write the dashboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("dashboard.html"))
    args = parser.parse_args()

    csvs = find_csvs(args.results_dir)
    if not csvs:
        raise SystemExit(f"No evaluation_*_detailed.csv found under {args.results_dir}")

    runs = [load_run(label, path) for label, path in csvs]
    payload = {
        "source": str(args.results_dir.resolve()),
        "generated": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
        "runs": runs,
        "summaries": [summarize(run) for run in runs],
    }
    data = json.dumps(payload).replace("</", "<\\/")
    args.output.write_text(HTML.replace("__DATA__", data), encoding="utf-8")
    print(f"Wrote {args.output} ({len(runs)} run(s))")


if __name__ == "__main__":
    main()
