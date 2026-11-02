#!/usr/bin/env python3
"""Monitor de solo lectura para la campaña; no requiere paquetes adicionales."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOTALS: dict[Path, int] = {}
LABELS = {"running": "en curso", "complete": "completo", "failed": "falló"}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def expected_records(folder, status):
    if "expected_records" in status:
        return status["expected_records"]
    if folder in TOTALS:
        return TOTALS[folder]
    manifest = read_json(folder / "manifest.json")
    config = manifest.get("config", {})
    vectors = read_json(folder / "vectors.json")
    if not config or not vectors:
        return None
    pairs = len(manifest["eval_ids"])
    baseline_count = pairs * (2 if config.get("text_interventions") or
                              config.get("both_baseline_variants") else 1)
    modes = sum(1 if mode == "steer" else config["control_replicates"]
                for mode in config["modes"])
    total = baseline_count + pairs * len(vectors) * len(config["alphas"]) * modes
    if config.get("text_interventions"):
        valid = {"entail": set(), "contra": set()}
        seen = set()
        try:
            # Las baselines son las primeras filas: no releer todas las generaciones.
            with (folder / "records.jsonl").open() as source:
                for line in source:
                    record = json.loads(line)
                    if record["condition"]["kind"] != "baseline":
                        break
                    seen.add(record["key"])
                    if record["parsed"]["status"] == "ok":
                        valid[record["variant"]].add(record["example_id"])
                    if len(seen) == baseline_count:
                        break
        except (OSError, ValueError):
            return None
        if len(seen) != baseline_count:
            return None
        total += len(valid["entail"]) * len(config["truncation_fractions"])
        if config.get("swap_cot"):
            total += len(valid["entail"]) + len(valid["entail"] & valid["contra"])
    TOTALS[folder] = total
    return total


def updated_text(value):
    try:
        stamp = datetime.fromisoformat(value)
        age = max(0, int((datetime.now(timezone.utc) - stamp).total_seconds()))
        return f"{stamp.astimezone():%H:%M:%S %Z} (hace {age // 60}m {age % 60}s)"
    except (TypeError, ValueError):
        return "sin datos"


def service_status(service):
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", service,
             "--property=ActiveState,SubState,MainPID"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode:
            return "no disponible (se muestran los archivos de progreso)"
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        return f"{fields.get('ActiveState', '?')}/{fields.get('SubState', '?')} · PID {fields.get('MainPID', '?')}"
    except (OSError, subprocess.TimeoutExpired):
        return "no disponible (se muestran los archivos de progreso)"


def log_tail(path):
    try:
        with path.open("rb") as source:
            source.seek(0, 2)
            size = source.tell()
            source.seek(max(0, size - 8192))
            lines = source.read().decode(errors="replace").splitlines()
        if size > 8192:
            lines = lines[1:]
        return lines[-4:]
    except OSError:
        return ["Todavía no hay log disponible."]


def render(folder, service):
    campaign = read_json(folder / "campaign.json")
    lines = [f"CAMPAÑA {folder.name} · {datetime.now().astimezone():%d/%m/%Y %H:%M:%S %Z}",
             f"Servicio: {service_status(service)}"]
    if not campaign:
        return "\n".join(lines + [f"Esperando un campaign.json legible en {folder}"])
    state = campaign.get("state", "desconocido")
    lines += [f"Estado registrado: {LABELS.get(state, state)}",
              f"Etapa: {campaign.get('current_step', 'pendiente')}", "",
              f"{'Modelo':<18} {'Generación':<12} {'Guardadas / total':<20} Avance"]
    for model in campaign.get("models", []):
        run = folder / model
        status = read_json(run / "status.json")
        count = status.get("records", 0)
        total = expected_records(run, status) if status else None
        state = status.get("state", "pendiente")
        progress = f"{100 * count / total:5.1f}%" if total else "—"
        counts = f"{count} / {total if total is not None else chr(63)}"
        lines.append(f"{model:<18} {LABELS.get(state, state):<12} {counts:<20} {progress}")
        if status:
            lines.append(f"  Última escritura: {updated_text(status.get('updated'))}")
            if status.get("last_key"):
                lines.append(f"  Condición: {status['last_key'].split('/')[0]}")
    steps = campaign.get("steps", [])
    lines += ["", f"Etapas terminadas: {sum(s.get('returncode') == 0 for s in steps)}",
              "El porcentaje cuenta generaciones; la evaluación y el análisis tienen etapas propias."]
    if campaign.get("error"):
        lines.append(f"ERROR: {campaign['error']}")
    if steps and steps[-1].get("log"):
        log = Path(steps[-1]["log"])
        if not log.is_absolute():
            log = ROOT / log
        lines += ["", f"Último log ({log.name}):", *log_tail(log)]
    lines += ["", "Ctrl+C para salir. El experimento continúa ejecutándose."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "runs/study-20260925",
                        help="directorio que contiene campaign.json")
    parser.add_argument("--interval", type=float, default=5, help="segundos entre actualizaciones (5)")
    parser.add_argument("--service", default="causal-fidelity-study.service")
    parser.add_argument("--once", action="store_true", help="mostrar una sola captura y salir")
    args = parser.parse_args()
    if not 0 < args.interval < float("inf"):
        parser.error("--interval debe ser un número positivo y finito")
    try:
        while True:
            display = render(args.campaign.resolve(), args.service)
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(display, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor cerrado; el experimento sigue su curso.")


if __name__ == "__main__":
    main()
