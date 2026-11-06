#!/usr/bin/env python3
"""Run every experiment in an experiments.yaml against the cpu-small InferenceService.

Without --run this only prints the plan and the manifest changes. With --run, for
each experiment it patches and applies the manifest, waits for vLLM to be ready,
checks the CPU binding, runs lightspeed-eval, and writes run_meta.json with the
experiment name.

Runs inside the eval-runner image as a Job (see eval/cluster/README.md). Needs PyYAML, a
logged-in `oc`, and the judge credentials in the environment (eval-credentials). API_KEY is
set from the Job's service account token, or from `oc whoami -t` when not in a pod.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BIND_ENV = "VLLM_CPU_OMP_THREADS_BIND"
CONTAINER = "kserve-container"
VLLM_ARG_PREFIXES = ("--model=", "--served-model-name=", "--max-model-len=")
PARSER_ARG_PREFIXES = ("--tool-call-parser=", "--reasoning-parser=")


def oc(*args: str, check: bool = True) -> str:
    """Run an `oc` command and return stdout."""
    result = subprocess.run(
        ["oc", *args], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise SystemExit(f"oc {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout

def oc_json(*args: str) -> dict[str, Any]:
    """Run an `oc ... -o json` command and parse the output."""
    return json.loads(oc(*args, "-o", "json"))

def parse_args(text: str, prefix: str) -> Optional[str]:
    """Return the value of a --flag=value style argument from a joined arg string."""
    match = re.search(rf"{re.escape(prefix)}=(\S+)", text)
    return match.group(1).strip("'\"") if match else None

def thinking_from_args(args_text: str) -> Optional[bool]:
    """Read enable_thinking from --default-chat-template-kwargs, if present."""
    match = re.search(r'"enable_thinking":\s*(true|false)', args_text, re.I)
    return match.group(1).lower() == "true" if match else None

def find_pod(isvc: str, namespace: str) -> dict[str, Any]:
    """Return the newest running pod of the InferenceService."""
    pods = oc_json(
        "get", "pods", "-n", namespace, "-l", f"serving.kserve.io/inferenceservice={isvc}"
    )["items"]
    running = [p for p in pods if p["status"].get("phase") == "Running"]
    if not running:
        raise SystemExit(f"No running pod found for InferenceService {isvc!r}")
    return max(running, key=lambda p: p["status"]["startTime"])

def parse_binding_log(log: str) -> dict[str, Any]:
    """Extract the last 'OpenMP thread binding info' block from the vLLM log."""
    info: dict[str, Any] = {}
    starts = [m.start() for m in re.finditer(r"OpenMP thread binding info", log)]
    if starts:
        block = log[starts[-1] :]
        bind = re.search(r"VLLM_CPU_OMP_THREADS_BIND='([^']*)'", block)
        cores = re.search(r"core ids=\[([^\]]*)\]", block)
        reserved = re.search(r"reserved_cpus=\[([^\]]*)\]", block)
        if bind:
            info["vllm_threads_bind"] = bind.group(1)
        if cores:
            ids = [int(x) for x in cores.group(1).split(",") if x.strip()]
            info["vllm_omp_core_ids"] = ids
            info["vllm_omp_threads"] = len(ids)
        if reserved:
            info["vllm_reserved_cpus"] = [
                int(x) for x in reserved.group(1).split(",") if x.strip()
            ]
    version = re.search(r"Initializing a V1 LLM engine \(v([^)]+)\)", log)
    if version:
        info["vllm_version"] = version.group(1)
    return info

def read_cpu_stat(pod: str, namespace: str) -> dict[str, int]:
    """Read cgroup CPU throttling counters from the serving container."""
    out = oc(
        "exec", "-n", namespace, pod, "-c", CONTAINER, "--",
        "cat", "/sys/fs/cgroup/cpu.stat",
        check=False,
    )
    stats: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in (
            "nr_periods", "nr_throttled", "throttled_usec", "usage_usec"
        ):
            stats[parts[0]] = int(parts[1])
    return stats

def collect(isvc: str, namespace: str) -> dict[str, Any]:
    """Gather metadata for the live InferenceService pod."""
    spec = oc_json("get", "inferenceservice", isvc, "-n", namespace)
    model = spec["spec"]["predictor"]["model"]
    args_text = " ".join(model.get("args", []))
    resources = model.get("resources", {})
    pod = find_pod(isvc, namespace)
    pod_name = pod["metadata"]["name"]
    node_name = pod["spec"]["nodeName"]
    node = oc_json("get", "node", node_name)
    labels = node["metadata"]["labels"]

    container = next(c for c in pod["spec"]["containers"] if c["name"] == CONTAINER)
    live = container.get("resources", {})
    limits = live.get("limits", resources.get("limits", {}))
    requests = live.get("requests", resources.get("requests", {}))

    meta: dict[str, Any] = {
        "inference_service": isvc,
        "backend_model": parse_args(args_text, "--model"),
        "served_model_name": parse_args(args_text, "--served-model-name"),
        "max_model_len": parse_args(args_text, "--max-model-len"),
        "aws_instance": labels.get("node.kubernetes.io/instance-type"),
        "node": node_name,
        "node_vcpus": int(node["status"]["capacity"]["cpu"]),
        "cpu_request": requests.get("cpu"),
        "cpu_limit": limits.get("cpu"),
        "memory_request": requests.get("memory"),
        "memory_limit": limits.get("memory"),
        "env_overrides": {e["name"]: e.get("value") for e in model.get("env", [])},
        "pod": pod_name,
        "pod_started_at": pod["status"]["startTime"],
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "thinking": thinking_from_args(args_text),
        "tool_call_parser": parse_args(args_text, "--tool-call-parser"),
        "reasoning_parser": parse_args(args_text, "--reasoning-parser"),
        "cpu_cores_note": "compute threads bound by vLLM; scheduler CPU not included",
    }
    log = oc("logs", pod_name, "-n", namespace, "-c", CONTAINER, check=False)
    binding = parse_binding_log(log)
    meta.update(binding)
    if "vllm_omp_threads" in binding:
        meta["cpu_cores"] = binding["vllm_omp_threads"]
    meta["cgroup_cpu_stat"] = read_cpu_stat(pod_name, namespace)
    return {k: v for k, v in meta.items() if v is not None and v != {} and v != []}


def expand(path: str) -> Path:
    """Expand ~ and resolve relative paths against the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


MEMORY_UNITS = {
    "Ki": 2**-20, "Mi": 2**-10, "Gi": 1.0, "Ti": 2**10,
    "K": 1e3 / 2**30, "M": 1e6 / 2**30, "G": 1e9 / 2**30, "T": 1e12 / 2**30,
}
MEMORY_KEYS = ("memory_request", "memory_limit")
VLLM_RESERVED_FRACTION = 0.92
STARTUP_BASELINE_GIB = 1.8
MIN_LIMIT_GIB = 32.0
HF_SIZE_CACHE: dict[str, Optional[float]] = {}


def memory_gib(value: Any) -> float:
    """Convert a Kubernetes-style memory quantity such as 96Gi or 512Mi to GiB."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]i?)?", str(value).strip())
    if not match:
        raise SystemExit(f"Cannot read memory value {value!r}; use a form like 96Gi")
    number, unit = match.groups()
    return float(number) * (MEMORY_UNITS[unit] if unit else 1 / 2**30)


def check_memory_pair(owner: str, entry: dict[str, Any]) -> None:
    """Fail if a memory value is unreadable or the request exceeds the limit."""
    for key in MEMORY_KEYS:
        if key in entry:
            memory_gib(entry[key])
    if all(key in entry for key in MEMORY_KEYS):
        if memory_gib(entry["memory_request"]) > memory_gib(entry["memory_limit"]):
            raise SystemExit(f"{owner}: memory_request is larger than memory_limit")


def validate_models(models: dict[str, Any]) -> None:
    """Every model needs its Hugging Face id and tool call parser; the rest is optional."""
    if not models:
        raise SystemExit("experiments file needs a `models:` section defining each model to test")
    for key, entry in models.items():
        unsupported = sorted(set(entry or {}) - set(ALLOWED_MODEL_KEYS))
        if unsupported:
            raise SystemExit(
                f"models.{key} has unsupported keys {unsupported}; supported: {list(ALLOWED_MODEL_KEYS)}. "
                "CPU and memory are set once under `defaults`, and can be overridden per experiment."
            )
        if "gated" in entry and not isinstance(entry["gated"], bool):
            raise SystemExit(f"models.{key}.gated must be true or false, got {entry['gated']!r}")
        missing = [k for k in REQUIRED_MODEL_KEYS if k not in (entry or {})]
        if missing:
            raise SystemExit(
                f"models.{key} is missing {missing}. Each model must set "
                f"{list(REQUIRED_MODEL_KEYS)} (use `none` for no tool parser). "
                "reasoning_parser is optional: leave it out if the model has none."
            )


def resolve_model(cfg: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    """Replace an experiment's model key with its definition (id and parsers)."""
    key = cfg.get("model")
    if key not in models:
        raise SystemExit(
            f"Experiment {cfg['name']!r} uses model {key!r}, which is not defined under "
            f"`models:`. Defined: {sorted(models)}"
        )
    entry = models[key]
    resolved = {
        **cfg,
        "model_key": key,
        "model": entry["model"],
        "tool_call_parser": NO_PARSER if entry["tool_call_parser"] is None else str(entry["tool_call_parser"]),
        "reasoning_parser": (
            NO_PARSER if entry.get("reasoning_parser") is None else str(entry["reasoning_parser"])
        ),
        "gated": bool(entry.get("gated", False)),
    }
    check_memory_pair(f"Experiment {cfg['name']!r}", resolved)
    return resolved


HF_INFO_CACHE: dict[str, dict[str, Any]] = {}


def hf_info(model_id: str) -> dict[str, Any]:
    """Checkpoint size in GiB and gating of a model on Hugging Face (None where unknown)."""
    if model_id not in HF_INFO_CACHE:
        info: dict[str, Any] = {"size": None, "gated": None}
        try:
            url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
            with urllib.request.urlopen(url, timeout=15) as response:  # nosec B310
                data = json.load(response)
            info["gated"] = data.get("gated", False)
            files = data.get("siblings") or []
            for suffix in (".safetensors", ".bin"):
                total = sum(f.get("size") or 0 for f in files if f["rfilename"].endswith(suffix))
                if total:
                    info["size"] = total / 2**30
                    break
        except Exception:  # noqa: BLE001
            pass
        HF_INFO_CACHE[model_id] = info
    return HF_INFO_CACHE[model_id]


def checkpoint_gib(model_id: str) -> Optional[float]:
    """Size of a model's weight files on Hugging Face in GiB, or None if it cannot be read."""
    return hf_info(model_id)["size"]


def supports_thinking(model_id: str) -> Optional[bool]:
    """True if the model's chat template has an enable_thinking switch, None if it cannot be read."""
    for name in ("chat_template.jinja", "tokenizer_config.json"):
        try:
            url = f"https://huggingface.co/{model_id}/raw/main/{name}"
            with urllib.request.urlopen(url, timeout=15) as response:  # nosec B310
                text = response.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        return "enable_thinking" in text
    return None


def thinking_report(experiments: list[dict[str, Any]]) -> list[str]:
    """Models run with thinking on whose chat template has no enable_thinking switch."""
    lines: list[str] = []
    seen: set[str] = set()
    for cfg in experiments:
        if not cfg["thinking"] or cfg["model"] in seen:
            continue
        seen.add(cfg["model"])
        if supports_thinking(cfg["model"]) is False:
            lines.append(
                f"  !  {cfg['model_key']}: its chat template has no enable_thinking switch, so the thinking flag "
                "changes nothing for it. If the model never reasons (Llama), set `thinking: false` so the run is "
                "labelled correctly. If it always reasons (gpt-oss), leave thinking on and run it once"
            )
    return lines


def secret_exists(namespace: str, name: str) -> Optional[bool]:
    """True or False if the secret can be looked up, None if oc cannot tell."""
    result = subprocess.run(
        ["oc", "get", "secret", name, "-n", namespace, "-o", "name"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        return True
    return False if "NotFound" in result.stderr else None


def gated_problems(experiments: list[dict[str, Any]]) -> list[str]:
    """Definite problems with gated models: they would fail to download and waste an hour."""
    problems: list[str] = []
    seen: set[str] = set()
    for cfg in experiments:
        if cfg["model"] in seen:
            continue
        seen.add(cfg["model"])
        hf_gated = hf_info(cfg["model"])["gated"]
        if hf_gated and not cfg["gated"]:
            problems.append(
                f"{cfg['model_key']} is gated on Hugging Face ({hf_gated}); set `gated: true` on it "
                "in `models:` so it uses the token"
            )
    if any(cfg["gated"] for cfg in experiments):
        namespace = experiments[0]["namespace"]
        if secret_exists(namespace, HF_TOKEN_SECRET) is False:
            problems.append(
                f"a gated model needs the `{HF_TOKEN_SECRET}` secret in {namespace}, which does not exist. "
                f"Create it: oc create secret generic {HF_TOKEN_SECRET} -n {namespace} "
                f"--from-literal={HF_TOKEN_KEY}=<your token>"
            )
    return problems


def gated_report(experiments: list[dict[str, Any]]) -> list[str]:
    """Lines for the plan: gated models that are set up, and any problems."""
    lines = [f"  !! {problem}" for problem in gated_problems(experiments)]
    gated = sorted({cfg["model_key"] for cfg in experiments if cfg["gated"]})
    if gated and not lines:
        lines.append(f"  ok {', '.join(gated)}: gated, the `{HF_TOKEN_SECRET}` secret is passed as {HF_TOKEN_ENV}")
    return lines


def node_memory_gib(isvc_node_label: str) -> Optional[float]:
    """Allocatable memory of the node the model pod is pinned to, if oc can read it."""
    out = oc(
        "get", "nodes", "-l", f"inference={isvc_node_label}",
        "-o", "jsonpath={.items[0].status.allocatable.memory}", check=False,
    ).strip()
    return memory_gib(out) if out else None


def memory_report(experiments: list[dict[str, Any]]) -> list[str]:
    """One line per distinct model and memory setting: whether the memory looks sufficient."""
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for cfg in experiments:
        marker = (cfg["model"], str(cfg["memory_request"]), str(cfg["memory_limit"]))
        if marker in seen:
            continue
        seen.add(marker)
        limit, request = memory_gib(cfg["memory_limit"]), memory_gib(cfg["memory_request"])
        label = f"{cfg['model_key']}: request {cfg['memory_request']} / limit {cfg['memory_limit']}"
        weights = checkpoint_gib(cfg["model"])
        if weights is None:
            lines.append(f"  ?  {label}: could not read the checkpoint size from Hugging Face")
            continue
        suggested = max(weights * 1.25 + 8, MIN_LIMIT_GIB)
        startup_floor = STARTUP_BASELINE_GIB / (1 - VLLM_RESERVED_FRACTION)
        if limit < startup_floor:
            lines.append(
                f"  !! {label}: vLLM reserves {VLLM_RESERVED_FRACTION:.0%} of the limit and needs the rest "
                f"(about {STARTUP_BASELINE_GIB} GiB) free at startup, so it will not start below "
                f"{startup_floor:.0f} GiB (suggested {suggested:.0f} GiB)"
            )
        elif limit < weights + 6:
            lines.append(f"  !! {label}: checkpoint {weights:.1f} GiB, the pod will probably be killed (suggested {suggested:.0f} GiB)")
        elif limit < suggested:
            lines.append(f"  !  {label}: checkpoint {weights:.1f} GiB, below the suggested {suggested:.0f} GiB")
        else:
            lines.append(f"  ok {label}: checkpoint {weights:.1f} GiB, suggested {suggested:.0f} GiB")
    node = node_memory_gib(experiments[0].get("node_label", "cpu-small"))
    if node is not None:
        biggest = max(memory_gib(c["memory_request"]) for c in experiments)
        status = "ok" if biggest <= node else "!!"
        lines.append(f"  {status} the largest memory_request is {biggest:.0f} GiB; the cpu-small node has {node:.0f} GiB allocatable")
    return lines


def validate_judges(judges: dict[str, Any]) -> None:
    """Every judge must state its provider and model."""
    if not judges:
        raise SystemExit("experiments file needs a `judges:` section defining each judge")
    for key, entry in judges.items():
        missing = [k for k in REQUIRED_JUDGE_KEYS if k not in (entry or {})]
        if missing:
            raise SystemExit(
                f"judges.{key} is missing {missing}. Each judge must set "
                f"{list(REQUIRED_JUDGE_KEYS)} (optional: temperature, max_completion_tokens)."
            )


def resolve_judges(cfg: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    """Turn an experiment's list of judge names into full judge definitions."""
    names = cfg.get("judges")
    if isinstance(names, str):
        names = [names]
    if not names:
        raise SystemExit(f"Experiment {cfg['name']!r} needs `judges:` (names from the `judges:` section)")
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(
            f"Experiment {cfg['name']!r} uses undefined judge(s) {unknown}. Defined: {sorted(registry)}"
        )
    if len(set(names)) != len(names):
        raise SystemExit(f"Experiment {cfg['name']!r} lists the same judge twice: {names}")
    if "judge_aggregation" in cfg:
        print(f"Warning: judge_aggregation in {cfg['name']!r} is ignored. Each judge's score is "
              "saved and the dashboard does the aggregating.")
    return {
        **cfg,
        "judge_names": list(names),
        "judge_entries": [{"alias": n, **registry[n]} for n in names],
    }


def judges_label(cfg: dict[str, Any]) -> str:
    """Short description of the judge panel for the plan output."""
    models = [e["model"] for e in cfg["judge_entries"]]
    return " + ".join(models)


def check_judge_credentials(experiments: list[dict[str, Any]]) -> None:
    """Fail before touching the cluster if a judge's credentials are not in the environment."""
    missing = sorted({
        (e["provider"], var)
        for cfg in experiments
        for e in cfg["judge_entries"]
        for var in PROVIDER_ENV.get(e["provider"], [])
        if not os.environ.get(var)
    })
    if missing:
        lines = "\n".join(f"  {provider}: {var}" for provider, var in missing)
        raise SystemExit(f"Missing judge credentials in the environment:\n{lines}")


def chat_template_kwargs(base_args: list[str]) -> dict[str, Any]:
    """Read the chat template kwargs JSON from a vLLM arg list (empty if absent)."""
    for arg in base_args:
        if arg.startswith(CHAT_KWARGS_PREFIX):
            return json.loads(arg[len(CHAT_KWARGS_PREFIX):].strip("'\""))
    return {}


def render_system_config(cfg: dict[str, Any], results: Path) -> Path:
    """Write the eval system config with this experiment's judge panel and return its path."""
    source = expand(cfg["system_config"])
    system = yaml.safe_load(source.read_text(encoding="utf-8"))
    models: dict[str, Any] = {}
    for judge in cfg["judge_entries"]:
        entry: dict[str, Any] = {
            "description": f"Judge {judge['alias']}",
            "provider": judge["provider"],
            "model": judge["model"],
        }
        params = {k: judge[k] for k in ("temperature", "max_completion_tokens") if k in judge}
        if params:
            entry["parameters"] = params
        models[judge["alias"]] = entry
    system.setdefault("llm_pool", {})["models"] = models
    panel = system.setdefault("judge_panel", {})
    panel["judges"] = list(cfg["judge_names"])
    panel["aggregation_strategy"] = EVAL_AGGREGATION
    out = results / "rendered_configs" / f"{slug(cfg['name'])}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(system, sort_keys=False), encoding="utf-8")
    return out


def render_manifest(base: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the base InferenceService patched with one experiment's settings."""
    manifest = copy.deepcopy(base)
    model = manifest["spec"]["predictor"]["model"]

    args = [
        a for a in model.get("args", [])
        if not a.startswith(VLLM_ARG_PREFIXES + PARSER_ARG_PREFIXES)
        and a != "--enable-auto-tool-choice"
    ]
    head = [
        f"--model={cfg['model']}",
        f"--served-model-name={cfg['served_model_name']}",
        f"--max-model-len={cfg['max_model_len']}",
    ]
    if cfg["tool_call_parser"] != NO_PARSER:
        head += ["--enable-auto-tool-choice", f"--tool-call-parser={cfg['tool_call_parser']}"]
    if cfg["reasoning_parser"] != NO_PARSER:
        head.append(f"--reasoning-parser={cfg['reasoning_parser']}")
    if cfg.get("thinking") is not None:
        args = [a for a in args if not a.startswith(CHAT_KWARGS_PREFIX)]
        kwargs = chat_template_kwargs(base_args=model.get("args", []))
        kwargs["enable_thinking"] = bool(cfg["thinking"])
        args.append(CHAT_KWARGS_PREFIX + json.dumps(kwargs))
    model["args"] = head + args + list(cfg.get("extra_args") or [])

    model["resources"] = {
        "requests": {"cpu": str(cfg["cpu_request"]), "memory": str(cfg["memory_request"])},
        "limits": {"cpu": str(cfg["cpu_limit"]), "memory": str(cfg["memory_limit"])},
    }

    env = [dict(entry) for entry in model.get("env", [])]

    def without(name: str) -> list[dict[str, Any]]:
        return [entry for entry in env if entry["name"] != name]

    bind = cfg.get("omp_threads_bind", "keep")
    if bind == "auto":
        env = without(BIND_ENV)
    elif bind != "keep":
        env = without(BIND_ENV) + [{"name": BIND_ENV, "value": str(bind)}]
    for name, value in (cfg.get("extra_env") or {}).items():
        env = without(name) + [{"name": name, "value": str(value)}]
    env = without(HF_TOKEN_ENV)
    if cfg.get("gated"):
        env.append({
            "name": HF_TOKEN_ENV,
            "valueFrom": {"secretKeyRef": {"name": HF_TOKEN_SECRET, "key": HF_TOKEN_KEY}},
        })
    if env:
        model["env"] = env
    else:
        model.pop("env", None)
    return manifest


def slug(name: str) -> str:
    """Make a filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def wait_until_ready(cfg: dict[str, Any], applied_at: datetime, restarted: bool) -> str:
    """Wait for a ready pod (newer than applied_at when the manifest changed)."""
    deadline = time.time() + float(cfg["ready_timeout_minutes"]) * 60
    cutoff = applied_at - timedelta(seconds=5)
    while time.time() < deadline:
        pods = oc_json(
            "get", "pods", "-n", cfg["namespace"],
            "-l", f"serving.kserve.io/inferenceservice={cfg['isvc']}",
        )["items"]
        for pod in pods:
            started = datetime.strptime(
                pod["status"]["startTime"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
            if restarted and started < cutoff:
                continue
            statuses = pod["status"].get("containerStatuses", [])
            main = next((s for s in statuses if s["name"] == CONTAINER), None)
            if not (main and main.get("ready")):
                continue
            name = pod["metadata"]["name"]
            log = oc("logs", name, "-n", cfg["namespace"], "-c", CONTAINER, check=False)
            if "Application startup complete" in log:
                return name
        time.sleep(20)
    raise SystemExit("Timed out waiting for the InferenceService pod to become ready")


def check_binding(cfg: dict[str, Any], pod: str) -> dict[str, Any]:
    """Read the vLLM binding from the log and fail if it is not what was requested."""
    log = oc("logs", pod, "-n", cfg["namespace"], "-c", CONTAINER, check=False)
    binding = parse_binding_log(log)
    wanted = cfg.get("omp_threads_bind", "keep")
    if wanted not in ("keep",) and binding.get("vllm_threads_bind") != str(wanted):
        raise SystemExit(
            f"Pod bound {binding.get('vllm_threads_bind')!r}, expected {wanted!r}"
        )
    return binding


def newest_eval_dir(results: Path, before: set[str]) -> Optional[Path]:
    """Return the eval_* folder created since `before`."""
    new = sorted(
        p for p in results.iterdir()
        if p.is_dir() and re.fullmatch(r"eval_\d{8}_\d{6}", p.name) and p.name not in before
    )
    return new[-1] if new else None


SA_TOKEN_FILE = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")


def cluster_token() -> str:
    """Return a bearer token for OLS: the pod's service account token, else `oc whoami -t`."""
    if SA_TOKEN_FILE.exists():
        return SA_TOKEN_FILE.read_text(encoding="utf-8").strip()
    return oc("whoami", "-t").strip()


FINGERPRINT_KEYS = (
    "model", "cores", "max_model_len", "dataset", "extra_args", "extra_env",
    "cpu_limit", "judges", "thinking", "tool_call_parser", "reasoning_parser",
)
DEFAULT_DEDUPE_ON = [
    "model", "cores", "max_model_len", "dataset", "judges", "thinking",
    "tool_call_parser", "reasoning_parser",
]
LEGACY_DEFAULTS = {
    "thinking": False,
    "tool_call_parser": "qwen3_coder",
    "reasoning_parser": "qwen3",
}
REQUIRED_JUDGE_KEYS = ("provider", "model")
EVAL_AGGREGATION = "average"
PROVIDER_ENV = {
    "hosted_vllm": ["HOSTED_VLLM_API_KEY", "HOSTED_VLLM_API_BASE"],
    "openai": ["OPENAI_API_KEY"],
}
REQUIRED_MODEL_KEYS = ("model", "tool_call_parser")
ALLOWED_MODEL_KEYS = ("model", "tool_call_parser", "reasoning_parser", "gated")
HF_TOKEN_ENV = "HF_TOKEN"
HF_TOKEN_SECRET = "hf-token"
HF_TOKEN_KEY = "HF_TOKEN"
NO_PARSER = "none"
CHAT_KWARGS_PREFIX = "--default-chat-template-kwargs="


def expand_cpu_list(text: str) -> Optional[list[int]]:
    """Expand a CPU list like '16-22,25' to sorted ids; None for auto/keep."""
    if text in ("auto", "keep"):
        return None
    ids: set[int] = set()
    for part in text.split(","):
        low, _, high = part.strip().partition("-")
        ids.update(range(int(low), int(high or low) + 1))
    return sorted(ids)


def dataset_conversations(cfg: dict[str, Any]) -> list[str]:
    """Conversation ids of the dataset, in file order."""
    data = yaml.safe_load(expand(cfg["eval_data"]).read_text(encoding="utf-8"))
    return [c["conversation_group_id"] for c in data]


def selected_conversations(cfg: dict[str, Any]) -> Optional[list[str]]:
    """The first max_conversations conversation ids, or None to run the whole dataset."""
    limit = cfg.get("max_conversations")
    if not limit:
        return None
    ids = dataset_conversations(cfg)
    if int(limit) > len(ids):
        raise SystemExit(
            f"max_conversations ({limit}) is larger than {expand(cfg['eval_data']).name} "
            f"({len(ids)} conversations)"
        )
    return ids[: int(limit)]


def dataset_label(cfg: dict[str, Any]) -> str:
    """Readable description of what is evaluated, for the plan and the dashboard."""
    name = expand(cfg["eval_data"]).name
    total = len(dataset_conversations(cfg))
    chosen = selected_conversations(cfg)
    return f"{name} (first {len(chosen)} of {total})" if chosen else f"{name} ({total} questions)"


def dataset_id(cfg: dict[str, Any]) -> str:
    """Identify the dataset by file name, content and subset, so any change counts as new."""
    path = expand(cfg["eval_data"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    chosen = selected_conversations(cfg)
    return f"{path.name}:{digest}" + (f":first{len(chosen)}" if chosen else "")


def eval_command(cfg: dict[str, Any], results: Path) -> list[str]:
    """The lightspeed-eval command line for one experiment."""
    command = [
        *shlex.split(cfg.get("eval_cmd", "uv run lightspeed-eval")),
        "--system-config", str(render_system_config(cfg, results)),
        "--eval-data", str(expand(cfg["eval_data"])),
        "--output-dir", str(results),
    ]
    chosen = selected_conversations(cfg)
    return command + (["--conv-ids", *chosen] if chosen else [])


def fingerprint(cfg: dict[str, Any], effective_cores: Optional[list[int]] = None) -> dict[str, Any]:
    """Describe what an experiment tests. effective_cores comes from the vLLM log."""
    bind = str(cfg.get("omp_threads_bind", "keep"))
    return {
        "model": cfg["model"],
        "cores": effective_cores if effective_cores is not None else expand_cpu_list(bind),
        "bind_requested": bind,
        "max_model_len": str(cfg["max_model_len"]),
        "dataset": dataset_id(cfg),
        "extra_args": list(cfg.get("extra_args") or []),
        "extra_env": {k: str(v) for k, v in (cfg.get("extra_env") or {}).items()},
        "cpu_limit": str(cfg["cpu_limit"]),
        "judges": sorted(e["model"] for e in cfg["judge_entries"]),
        "thinking": cfg.get("thinking"),
        "tool_call_parser": cfg.get("tool_call_parser"),
        "reasoning_parser": cfg.get("reasoning_parser"),
    }


def same_config(pending: dict[str, Any], stored: dict[str, Any], keys: list[str]) -> bool:
    """True if a finished run matches a pending experiment on every key in `keys`."""
    if "judges" not in stored and stored.get("judge_model"):
        stored = {**stored, "judges": [stored["judge_model"]]}
    for key in keys:
        if key == "cores":
            if pending["bind_requested"] == "auto":
                ok = stored.get("bind_requested") == "auto"
            elif pending["cores"] is None:
                ok = stored.get("bind_requested") == pending["bind_requested"]
            else:
                ok = stored.get("cores") == pending["cores"]
        elif key in stored:
            ok = stored[key] == pending[key]
        elif key in LEGACY_DEFAULTS:
            ok = LEGACY_DEFAULTS[key] == pending[key]
        else:
            ok = False
        if not ok:
            return False
    return True


def completed_runs(results: Path) -> list[dict[str, Any]]:
    """Finished runs (run_meta.json plus a results CSV): their name and fingerprint."""
    runs: list[dict[str, Any]] = []
    for meta in results.glob("eval_*/*/run_*/run_meta.json"):
        if not any(meta.parent.glob("evaluation_*_detailed.csv")):
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        runs.append({"name": data.get("name"), "fingerprint": data.get("fingerprint") or {}})
    return runs


def find_duplicate(
    cfg: dict[str, Any], finished: list[dict[str, Any]]
) -> tuple[Optional[str], bool]:
    """Return (name of a finished run with the same config, whether the name was reused)."""
    keys = [
        "judges" if k == "judge_model" else k
        for k in cfg.get("dedupe_on", DEFAULT_DEDUPE_ON)
        if k != "judge_aggregation"
    ]
    unknown = set(keys) - set(FINGERPRINT_KEYS)
    if unknown:
        raise SystemExit(f"Unknown dedupe_on keys: {sorted(unknown)}; use {list(FINGERPRINT_KEYS)}")
    pending = fingerprint(cfg)
    for run in finished:
        if run["fingerprint"] and same_config(pending, run["fingerprint"], keys):
            return run["name"] or "(unnamed)", False
    return None, any(run["name"] == cfg["name"] for run in finished)


def run_experiment(cfg: dict[str, Any], base: dict[str, Any]) -> None:
    """Apply one experiment, run the eval, and record its metadata."""
    results = expand(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    rendered_dir = results / "rendered_manifests"
    rendered_dir.mkdir(exist_ok=True)
    rendered = rendered_dir / f"{slug(cfg['name'])}.yaml"
    rendered.write_text(yaml.safe_dump(render_manifest(base, cfg), sort_keys=False))

    applied_at = datetime.now(timezone.utc)
    out = oc("apply", "-f", str(rendered))
    print(out.strip())
    restarted = "unchanged" not in out
    pod = wait_until_ready(cfg, applied_at, restarted)
    binding = check_binding(cfg, pod)
    print(f"Pod {pod} ready; vLLM threads: {binding.get('vllm_omp_threads')} "
          f"on {binding.get('vllm_omp_core_ids')}")

    env = dict(os.environ)
    env["API_KEY"] = cluster_token()
    before = {p.name for p in results.iterdir()}
    eval_repo = expand(cfg["eval_repo"])
    subprocess.run(eval_command(cfg, results), cwd=eval_repo, env=env, check=True)
    eval_dir = newest_eval_dir(results, before)
    if not eval_dir:
        raise SystemExit("Eval finished but no new eval_* folder was found")
    run_dirs = sorted(eval_dir.glob("*/run_*"))
    if not run_dirs:
        raise SystemExit(f"{eval_dir} has no run_* folders; cannot record run_meta.json")
    for run_dir in run_dirs:
        if not any(run_dir.glob("evaluation_*_detailed.csv")):
            raise SystemExit(
                f"{run_dir} has no results CSV: the eval produced nothing "
                "(check the judge credentials and OLS access in the log above)"
            )
        meta = collect(cfg["isvc"], cfg["namespace"])
        meta["name"] = cfg["name"]
        meta["judges"] = [
            {k: v for k, v in judge.items() if v is not None} for judge in cfg["judge_entries"]
        ]
        meta["dataset"] = dataset_label(cfg)
        meta["fingerprint"] = fingerprint(cfg, meta.get("vllm_omp_core_ids"))
        target = run_dir / "run_meta.json"
        target.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {target}")
    build_dashboard(cfg, results)


def build_dashboard(cfg: dict[str, Any], results: Path) -> None:
    """Rebuild the dashboard from every run in the results folder."""
    output = Path(cfg.get("dashboard_output", "~/Desktop/hybrid_dashboard.html")).expanduser()
    script = Path(__file__).resolve().parent / "build_dashboard.py"
    subprocess.run([sys.executable, str(script), str(results), "-o", str(output)], check=True)


def main() -> None:
    """Load the experiments file and print or execute the plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments_file", type=Path)
    parser.add_argument("--run", action="store_true", help="actually apply and run")
    parser.add_argument("--only", help="run just the experiment with this name")
    parser.add_argument("--rerun", action="store_true",
                        help="run experiments even if the same config already finished")
    for flag in ("results-dir", "system-config", "eval-repo", "eval-cmd", "dashboard-output"):
        parser.add_argument(f"--{flag}", help=f"override defaults.{flag.replace('-', '_')}")
    args = parser.parse_args()

    spec = yaml.safe_load(args.experiments_file.read_text(encoding="utf-8"))
    defaults = spec.get("defaults", {})
    overrides = {
        key: getattr(args, key)
        for key in ("results_dir", "system_config", "eval_repo", "eval_cmd", "dashboard_output")
        if getattr(args, key)
    }
    models = spec.get("models") or {}
    validate_models(models)
    judge_registry = spec.get("judges") or {}
    validate_judges(judge_registry)
    experiments = [
        resolve_judges(resolve_model({**defaults, **e, **overrides}, models), judge_registry)
        for e in spec.get("experiments", [])
    ]
    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]
    if not experiments:
        raise SystemExit("No experiments to run")
    names = [e["name"] for e in experiments]
    if len(set(names)) != len(names):
        raise SystemExit("Experiment names must be unique")

    base = yaml.safe_load(expand(experiments[0]["base_manifest"]).read_text(encoding="utf-8"))
    for cfg in experiments:
        cfg.setdefault("thinking", True)
    print(f"{len(experiments)} experiment(s):")
    for cfg in experiments:
        model = render_manifest(base, cfg)["spec"]["predictor"]["model"]
        env = {e["name"]: e.get("value") for e in model.get("env", [])}
        print(f"  - {cfg['name']}")
        print(f"      model={cfg['model_key']} ({cfg['model']})  "
              f"tool_parser={cfg['tool_call_parser']}  reasoning_parser={cfg['reasoning_parser']}")
        print(f"      bind={env.get(BIND_ENV, 'auto')}  "
              f"cpu={cfg['cpu_request']}/{cfg['cpu_limit']}  memory={cfg['memory_request']}/{cfg['memory_limit']}  "
              f"thinking={cfg['thinking']}  "
              f"judges={judges_label(cfg)}")
        print(f"      dataset={dataset_label(cfg)}")
    print("\nMemory check (checkpoint size from Hugging Face; suggested = max(checkpoint x 1.25 + 8, 32) GiB):")
    print("\n".join(memory_report(experiments)))
    thinking_lines = thinking_report(experiments)
    if thinking_lines:
        print("\nThinking:")
        print("\n".join(thinking_lines))
    gated_lines = gated_report(experiments)
    if gated_lines:
        print("\nGated models:")
        print("\n".join(gated_lines))
    finished = completed_runs(expand(experiments[0]["results_dir"]))
    if not args.run:
        print("\nAlready done (same config, would be skipped):")
        skipped = 0
        for cfg in experiments:
            twin, _ = find_duplicate(cfg, finished)
            if twin:
                skipped += 1
                print(f"  - {cfg['name']}  (matches finished run {twin!r})")
        print("  none" if not skipped else "")
        print("Plan only. Re-run with --run to apply and evaluate.")
        return

    pending = [
        cfg for cfg in experiments if args.rerun or not find_duplicate(cfg, finished)[0]
    ]
    check_judge_credentials(pending)
    problems = gated_problems(pending)
    if problems:
        raise SystemExit("Gated model problems:\n" + "\n".join(f"  {p}" for p in problems))
    for number, cfg in enumerate(experiments, 1):
        print(f"\n=== [{number}/{len(experiments)}] {cfg['name']} ===")
        twin, name_reused = find_duplicate(cfg, finished)
        if twin and not args.rerun:
            print(f"Skipping: same config already finished as {twin!r} (use --rerun to repeat)")
            continue
        if name_reused:
            print("Warning: a finished run has this name but a different config. "
                  "Consider renaming this experiment so the dashboard stays clear.")
        run_experiment(cfg, base)
        finished = completed_runs(expand(cfg["results_dir"]))


if __name__ == "__main__":
    main()
