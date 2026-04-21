"""Minimal Spark history server client for fetching per-node metrics.

Assumes a Spark history server is already running (default: localhost:18080)
and reading the same ``spark.eventLog.dir`` the benchmark writes to.

Usage:

    from history_server_client import fetch_op_time
    op = fetch_op_time("my_app_name_20260421_112233", node_name="GpuArrowEvalPython")
    # -> {"total": 6.2, "min": 6.2, "med": 6.2, "max": 6.2}  (seconds)

Inspired by cuaether-agent's utils/spark_history_server.py, trimmed to the
single call path we need: find app by name -> find SQL query -> find node ->
parse a named metric.
"""

import re
import time
from typing import Any, Dict, List, Optional
from urllib import request as _urlreq
from urllib.error import URLError


# Multi-task form: "total (min, med, max (stageId: taskId))", e.g.
#   "42.7 m (1.3 m, 1.3 m, 1.4 m (stage 1.0: task 8))"
_METRIC_FULL_RE = re.compile(
    r"(\d+\.?\d*)\s*(ms|s|m|h)\s*\("
    r"(\d+\.?\d*)\s*(ms|s|m|h),\s*"
    r"(\d+\.?\d*)\s*(ms|s|m|h),\s*"
    r"(\d+\.?\d*)\s*(ms|s|m|h)\s*\(.*\)\)"
)
# Single-task / compact form: "22.9 s"
_METRIC_SIMPLE_RE = re.compile(r"^\s*(\d+\.?\d*)\s*(ms|s|m|h)\s*$")

_UNIT_TO_SECONDS = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


def _get_json(url: str, timeout: float = 5.0) -> Any:
    import json
    with _urlreq.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _parse_metric(metric_str: str) -> Optional[Dict[str, float]]:
    """Parse op-time metric string -> dict of seconds, or None.

    Handles two shapes emitted by Spark:
      * multi-task: "total (min, med, max (stage X: task Y))", often prefixed
        by a label line (so we try the last line first).
      * single-task / compact: "22.9 s" (min==med==max==total).
    """
    # Try the full form against every line (values line is usually last).
    for line in reversed(metric_str.splitlines() or [metric_str]):
        m = _METRIC_FULL_RE.search(line)
        if m:
            g = m.groups()
            return {
                "total": float(g[0]) * _UNIT_TO_SECONDS[g[1]],
                "min":   float(g[2]) * _UNIT_TO_SECONDS[g[3]],
                "med":   float(g[4]) * _UNIT_TO_SECONDS[g[5]],
                "max":   float(g[6]) * _UNIT_TO_SECONDS[g[7]],
            }
    # Fall back to compact single-task form.
    m = _METRIC_SIMPLE_RE.match(metric_str.strip())
    if m:
        total = float(m.group(1)) * _UNIT_TO_SECONDS[m.group(2)]
        return {"total": total, "min": total, "med": total, "max": total}
    return None


def _find_app_id(api_url: str, app_name: str) -> Optional[str]:
    apps = _get_json(f"{api_url}/applications")
    for a in apps:
        if a.get("name") == app_name:
            return a.get("id")
    return None


def _find_node(nodes: List[Dict[str, Any]], node_name: str) -> Optional[Dict[str, Any]]:
    target = node_name.strip().lower()
    # Prefer exact match; fall back to substring (plugin sometimes adds suffixes
    # like " (Stage 0)" / " [codegen id X]").
    for n in nodes:
        if n.get("nodeName", "").strip().lower() == target:
            return n
    for n in nodes:
        if target in n.get("nodeName", "").strip().lower():
            return n
    return None


def fetch_op_time(
    app_name: str,
    node_name: str = "GpuArrowEvalPython",
    metric_name: str = "op time",
    port: int = 18080,
    num_retries: int = 10,
    retry_delay: float = 2.0,
) -> Optional[Dict[str, float]]:
    """Fetch the named timing metric for a given node in the given app's SQL plan.

    Retries while the history server is still ingesting the event log. Returns
    ``{"total", "min", "med", "max"}`` in seconds, or ``None`` if anything
    along the way isn't found (we keep failure quiet since metric collection is
    purely observational).
    """
    api_url = f"http://localhost:{port}/api/v1"
    last_err: Optional[BaseException] = None

    for _ in range(num_retries):
        try:
            app_id = _find_app_id(api_url, app_name)
            if app_id is None:
                raise LookupError(f"app '{app_name}' not found")

            queries = _get_json(f"{api_url}/applications/{app_id}/sql")
            # A simple read -> UDF -> write job produces a single SQL query.
            # If there are multiple, scan them all for the target node.
            for q in queries:
                node = _find_node(q.get("nodes") or [], node_name)
                if node is None:
                    continue
                for d in node.get("metrics") or []:
                    if d.get("name") == metric_name:
                        parsed = _parse_metric(d.get("value", ""))
                        if parsed is not None:
                            return parsed

            raise LookupError(
                f"node '{node_name}' with metric '{metric_name}' not found yet"
            )
        except (URLError, LookupError, ValueError, TimeoutError) as e:
            last_err = e
            time.sleep(retry_delay)

    if last_err is not None:
        # Surface the last error via return None + best-effort print at call site.
        pass
    return None
