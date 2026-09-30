#!/usr/bin/env python3
"""Run a TOML command plan with a cumulative wall ceiling and GPU telemetry.

No shell expansion, downloads, retries, or automatic budget extension. Outputs must
be below .data. GPU utilization/power are device-wide; process VRAM is separate.
"""
import argparse
from contextlib import ExitStack
import csv
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import tomllib


def atomic_json(path, value):
    stage = path.with_suffix(path.suffix + ".partial")
    stage.write_text(json.dumps(value, indent=2) + "\n")
    stage.replace(path)


def query(args):
    try:
        result = subprocess.run(
            ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return list(csv.reader(result.stdout.splitlines()))
    except (OSError, subprocess.SubprocessError):
        return []


def telemetry(pid, elapsed):
    row = {"elapsed_seconds": elapsed, "pid": pid}
    device = query(["-i", "0", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm"])
    if device:
        for key, value in zip(
            ["device_gpu_percent", "device_vram_mib", "device_power_w", "temperature_c", "sm_clock_mhz"], device[0]
        ):
            try:
                row[key] = float(value.strip())
            except ValueError:
                pass
    processes = query(["--query-compute-apps=pid,used_gpu_memory"])
    for entry in processes:
        if len(entry) == 2 and entry[0].strip() == str(pid):
            try:
                row["process_vram_mib"] = float(entry[1].strip())
            except ValueError:
                pass
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                row["process_rss_mib"] = int(line.split()[1]) / 1024
    except OSError:
        pass
    return row


def stop(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def run(resources):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="TOML command plan (Python 3.11+)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, help="shared 12-hour ledger across study legs")
    args = parser.parse_args()
    plan_text = args.plan.read_text()
    plan = tomllib.loads(plan_text)
    budget = plan["max_command_seconds"]
    if not 0 < budget <= 43200:
        parser.error("the bounded study ceiling is 43200 seconds (12 hours)")
    data_root = Path(".data").resolve()
    global_ledger = None
    budget_lock = None
    if args.budget_ledger:
        args.budget_ledger = args.budget_ledger.resolve()
        if not args.budget_ledger.is_relative_to(data_root):
            parser.error("budget ledger must be inside .data")
        args.budget_ledger.parent.mkdir(parents=True, exist_ok=True)
        budget_lock = resources.enter_context(args.budget_ledger.with_suffix(".lock").open("a"))
        try:
            fcntl.flock(budget_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another leg is already using this study budget")
        global_ledger = json.loads(args.budget_ledger.read_text()) if args.budget_ledger.exists() else {
            "ceiling_seconds": 43200, "command_seconds": 0, "legs": [],
            "scope": "cumulative capture, training, and GPU evaluation command wall time; CPU engineering/reporting excluded",
        }
        if not (0 < global_ledger['ceiling_seconds'] <= 43200
                and math.isfinite(global_ledger['command_seconds'])
                and 0 <= global_ledger['command_seconds'] <= global_ledger['ceiling_seconds']):
            parser.error("invalid shared study budget")
        if global_ledger.get("running"):
            parser.error("unfinished study leg: inspect and account for it before continuing")
        reservations = args.budget_ledger.with_suffix('.reservations.toml')
        if reservations.exists():
            required = tomllib.loads(reservations.read_text()).get('ledgers', [])
            if any(str(Path(p).resolve()) not in global_ledger['legs'] for p in required):
                parser.error('account all separately reserved legs before beginning a new shared-budget leg')
        budget = min(budget, global_ledger["ceiling_seconds"] - global_ledger["command_seconds"])
        if budget <= 0:
            parser.error("shared study budget exhausted")
    out = args.output.resolve()
    if not out.is_relative_to(data_root):
        parser.error("output must be inside .data")
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "ledger.json"
    if ledger_path.exists():
        parser.error("ledger already exists; preserve it and use a new output directory")
    (out / "plan.toml").write_text(plan_text)
    ledger = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "max_command_seconds": budget,
        "commands": [],
        "telemetry_interval_seconds": 1,
        "telemetry_scope": "device-wide utilization/power; separate process VRAM/RSS; other desktop GPU activity may contribute",
    }
    used = 0
    if global_ledger is not None:
        global_ledger["running"] = {"output": str(out), "started_utc": ledger["started_utc"]}
        atomic_json(args.budget_ledger, global_ledger)
    for item in plan["commands"]:
        name = item["name"]
        if Path(name).name != name or name in {".", ".."}:
            parser.error("command name must be a filename component")
        # Reserve teardown time so a process needing the five-second TERM grace
        # cannot overrun the cumulative ceiling.
        allowed = min(item["max_seconds"], budget - used - 10)
        if allowed <= 0:
            ledger["stop_reason"] = "cumulative_wall_limit"
            break
        print(f"start {name}: limit {allowed:.1f}s", flush=True)
        record = {"name": name, "argv": item["argv"], "limit_seconds": allowed, "status": "running"}
        ledger["commands"].append(record)
        atomic_json(ledger_path, ledger)
        start = time.monotonic()
        with (out / f"{name}.log").open("w") as log, (out / f"{name}-gpu.jsonl").open("w") as gpu:
            child = subprocess.Popen(item["argv"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            last_publish = -30
            try:
                while child.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed - last_publish >= 30:
                        record['running_elapsed_seconds'] = elapsed
                        atomic_json(ledger_path, ledger)
                        if global_ledger is not None:
                            global_ledger['running'].update(command=name, pid=child.pid, command_elapsed_seconds=elapsed)
                            atomic_json(args.budget_ledger, global_ledger)
                        last_publish = elapsed
                    if elapsed >= allowed:
                        record["status"] = "wall_time_limit"
                        stop(child)
                        break
                    gpu.write(json.dumps(telemetry(child.pid, elapsed)) + "\n")
                    gpu.flush()
                    time.sleep(min(1, max(0, allowed - (time.monotonic() - start))))
            finally:
                stop(child)
            record["exit_code"] = child.returncode
        record["elapsed_seconds"] = time.monotonic() - start
        used += record["elapsed_seconds"]
        if record["status"] == "running":
            record["status"] = "complete" if child.returncode == 0 else "failed"
        ledger["command_seconds"] = used
        atomic_json(ledger_path, ledger)
        if global_ledger is not None:
            global_ledger["command_seconds"] += record["elapsed_seconds"]
            global_ledger["running"]["completed_command_seconds"] = used
            atomic_json(args.budget_ledger, global_ledger)
        print(f"{name}: {record['status']}, {record['elapsed_seconds']:.1f}s; total {used:.1f}s", flush=True)
        if record["status"] != "complete":
            ledger["stop_reason"] = record["status"]
            break
    else:
        ledger["stop_reason"] = "plan_complete"
    ledger["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(ledger_path, ledger)
    if global_ledger is not None:
        global_ledger["legs"].append(str(ledger_path))
        global_ledger.pop("running", None)
        atomic_json(args.budget_ledger, global_ledger)
    if ledger["stop_reason"] != "plan_complete":
        raise SystemExit(1)


def main():
    with ExitStack() as resources:
        run(resources)


if __name__ == "__main__":
    main()
