"""Serialize generation and the local vLLM judge on shared-memory machines."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request


def prepare_stage(name, container, *, url='http://127.0.0.1:8000',
                  model='nvidia/Qwen3.6-35B-A3B-NVFP4', timeout=600):
    """Called only for unfinished stages; abort before work if setup fails."""
    if not container:
        return
    if container.startswith('-'):
        raise ValueError('Container name cannot be a command option')
    if name.endswith('-generate'):
        print(f'Releasing vLLM memory before {name}: {container}', flush=True)
        subprocess.run(['docker', 'stop', '--timeout', '30', container],
                       check=True, timeout=90)
        running = subprocess.check_output(
            ['docker', 'inspect', '--format', '{{.State.Running}}', container],
            text=True, timeout=15).strip()
        if running != 'false':
            raise RuntimeError(f'{container} is still running; refusing concurrent generation')
    elif name == 'judge-validation' or name.endswith('-judge'):
        print(f'Starting vLLM for {name}: {container}', flush=True)
        subprocess.run(['docker', 'start', container], check=True, timeout=90)
        deadline = time.monotonic() + timeout
        last_error = 'endpoint not ready'
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url + '/v1/models', timeout=5) as response:
                    payload = json.load(response)
                models = {item['id'] for item in payload.get('data', [])}
                if model in models:
                    print(f'vLLM judge ready: {model}', flush=True)
                    return
                last_error = f'Expected {model}; advertised {sorted(models)}'
            except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError) as error:
                last_error = str(error)
            time.sleep(5)
        raise RuntimeError(f'vLLM judge did not become ready in {timeout}s: {last_error}')
