# Trabajo paralelo — 2 de noviembre de 2026

El usuario autorizó paralelizar todo el trabajo posible. Se mantuvo la generación
en marcha y se abrieron tres tareas independientes de agentes, sin usar el
evaluador local ni cargar otro modelo en la GPU:

| Tarea | Entrega |
|---|---|
| Auditoría del evaluador | `docs/judge-audit-20261102.md` y `artifacts/diagnostics/20261102/judge-review.json` |
| Verificación estadística y figuras | `scripts/build_interim_figures.py`, `docs/interim-statistics-20261102.md` y `artifacts/diagnostics/20261102/figures/` |
| Borrador independiente | `paper/interim-20261102.tex`, `paper/interim-20261102.pdf` y `docs/manuscript-progress-20261102.md` |

La auditoría es una revisión de asistente, no una anotación humana de referencia.
Los juicios originales se conservan; sus categorías de incoherencia no deben
presentarse como casos confirmados. Los análisis de respuestas finales no
dependen de esas categorías del evaluador.

## Descarga anticipada de QwQ

A las 09:44:38 CET arrancó `causal-fidelity-qwq-prefetch.service`. Descarga
`Qwen/QwQ-32B` en la revisión congelada
`976055f8c83f394f35dbd3ab09a285a984907bd0` mediante el script existente, hacia
la misma caché que usa la campaña. Así, cuando llegue su etapa de descarga,
podrá reutilizar los archivos ya descargados.

El servicio tiene prioridad reducida (`Nice=15`, `IOWeight=10`), cuota de CPU
equivalente a un núcleo, umbral de memoria de 2 GiB y máximo de 4 GiB. Se usa
descarga HTTP sin Xet ni HF Transfer. No carga pesos en la GPU ni modifica el
protocolo. Al arrancar había unos 35 GiB disponibles de RAM y 2,7 TiB libres de
disco. El servicio de generación seguía activo, con cero reinicios.

El lanzamiento registrado fue:

```bash
systemd-run --user --unit=causal-fidelity-qwq-prefetch \
  --working-directory=/home/ialab/jairo_experiments/causal-fidelity-cot/causal-fidelity-cot \
  --property=Nice=15 --property=CPUQuota=100% \
  --property=MemoryHigh=2G --property=MemoryMax=4G --property=IOWeight=10 \
  --setenv=HF_HUB_DISABLE_XET=1 --setenv=HF_HUB_ENABLE_HF_TRANSFER=0 \
  --setenv=PYTHONUNBUFFERED=1 \
  /home/ialab/jairo_experiments/causal-fidelity-cot/causal-fidelity-cot/env/bin/python \
  scripts/download_models.py qwq-32b
```

No repetir ese comando mientras exista una descarga activa. Dos intentos previos
fueron cancelados antes de ejecutarse por la herramienta; el arranque posterior
se verificó mediante systemd, PID y registro de la revisión descargada.

Monitorización de solo lectura:

```bash
systemctl --user status causal-fidelity-study.service causal-fidelity-qwq-prefetch.service
journalctl --user -u causal-fidelity-qwq-prefetch.service -n 20 --no-pager
env/bin/python scripts/monitor_campaign.py --once
```

El servicio de descarga es transitorio: continúa tras terminar esta conversación,
pero no se instala como tarea de arranque después de un reinicio del equipo.
La campaña conserva su propia etapa de descarga y puede aprovechar la caché
parcial si la descarga anticipada se interrumpe.

## Límites operativos

Solo hay una máquina de ejecución confirmada. Distill-32B y QwQ-32B siguen
generando secuencialmente; la descarga anticipada sí se solapa con Distill.
El evaluador local se mantiene bajo la política existente de uso exclusivo de
memoria. No se han iniciado recursos de pago ni trabajos en otros equipos.

El lanzador activo, los ocho archivos científicos congelados, sus semillas y
la configuración siguen intactos. Los agentes escriben en archivos separados
para evitar colisiones. La comparación principal Distill-32B/QwQ-32B requiere
las etapas restantes de la campaña; las entregas de hoy se etiquetan como
provisionales.
