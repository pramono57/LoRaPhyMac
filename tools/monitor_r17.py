import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime

ROOT = Path.home() / "Documents/LoRaPhyMAC"
OUTPUT = ROOT / "artifacts/experiments/apc_load_sweep/full_factorial_v1"
LOGS = Path.home() / "Library/Logs/LoRaPhyMAC/apc_load_sweep_r17"
LOCK = Path.home() / "Library/Caches/LoRaPhyMAC/r17_production.lock"

REFRESH = 5


def clear():
    print("\033[2J\033[H", end="")


def bar(done, total, width=32):
    ratio = min(done / total, 1) if total else 0
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def get_statuses():
    result = {"FF": {}, "AA": {}}

    if not OUTPUT.exists():
        return result

    # Counts only production run status files; ignores pilot namespace.
    for status_path in OUTPUT.rglob("status.json"):
        if "analysis" in status_path.parts:
            continue

        try:
            data = json.loads(status_path.read_text())
        except (OSError, ValueError):
            continue

        run_id = str(data.get("run_id") or status_path.parent.name)
        name = run_id.lower()

        if "apc_load_sweep_ff_full_factorial_v1" in name:
            model = "FF"
        elif "apc_load_sweep_aa_full_factorial_v1" in name:
            model = "AA"
        else:
            continue

        status = str(
            data.get("status", data.get("state", "UNKNOWN"))
        ).upper()

        result[model][run_id] = status

    return result


def last_lines(filename, count=4):
    path = LOGS / filename
    if not path.exists():
        return ["Belum ada log."]

    try:
        with path.open(errors="replace") as handle:
            from collections import deque
            return list(deque(handle, maxlen=count))
    except OSError:
        return ["Log belum dapat dibaca."]


def process_status():
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,pcpu=,rss=,command="],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode:
            return ["Pemeriksaan proses tidak diizinkan."]

        matches = []
        for line in result.stdout.splitlines():
            if (
                "loraphymac.experiments.apc_load_sweep" in line
                and (" -m " in line or "python" in line.lower())
            ):
                fields = line.strip().split(None, 3)
                if len(fields) >= 4:
                    pid, cpu, rss, command = fields
                    ram_mb = int(rss) / 1024
                    matches.append(
                        f"PID {pid} | CPU {cpu}% | RAM {ram_mb:.1f} MiB"
                    )
        return matches or ["Tidak terdeteksi (atau akses ps terbatas)."]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return ["Informasi proses tidak tersedia."]


def size_bytes(path):
    total = 0
    if path.exists():
        for directory, _, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(directory) / name).stat().st_size
                except OSError:
                    pass
    return total


while True:
    clear()
    statuses = get_statuses()

    ff = sum(s == "COMPLETED" for s in statuses["FF"].values())
    aa = sum(s == "COMPLETED" for s in statuses["AA"].values())
    total = ff + aa

    failures = sum(
        s in {"FAILED", "ERROR"}
        for model in statuses.values()
        for s in model.values()
    )

    disk = shutil.disk_usage(ROOT)
    output_mb = size_bytes(OUTPUT) / (1024 * 1024)

    print("=" * 68)
    print(" LoRaPhyMAC | R1.7 FULL FACTORIAL MONITOR")
    print(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 68)

    print(f"\n TOTAL   [{bar(total, 120)}] {total:3}/120")
    print(f" FF      [{bar(ff, 60)}] {ff:3}/60")
    print(f" AA      [{bar(aa, 60)}] {aa:3}/60")

    print("\n STATUS")
    print(f" Completed : {total}")
    print(f" Failed    : {failures}")
    print(f" Lock      : {'ADA' if LOCK.exists() else 'TIDAK ADA'}")
    print(f" Output    : {output_mb:,.2f} MiB")
    print(f" Disk free : {disk.free / (1024**3):.2f} GiB")

    print("\n PROSES SIMULASI")
    for line in process_status():
        print(" ", line)

    print("\n LOG FF (4 baris terakhir)")
    for line in last_lines("ff.log"):
        print(" ", line.rstrip()[:120])

    print("\n LOG AA (4 baris terakhir)")
    for line in last_lines("aa.log"):
        print(" ", line.rstrip()[:120])

    print("\n POSTPROCESSING")
    for line in last_lines("postprocessing.log", 2):
        print(" ", line.rstrip()[:120])

    print("\n" + "-" * 68)
    print(" Refresh: 5 detik | Ctrl+C: keluar dari monitor")
    print(" Read-only | Tidak memengaruhi simulator")

    try:
        time.sleep(REFRESH)
    except KeyboardInterrupt:
        print("\nMonitoring dihentikan. Simulasi tidak terpengaruh.")
        break
