# Reproducción y operación

Los comandos se ejecutan desde la raíz del repositorio. Esta guía distingue una
ejecución nueva de la campaña local que ya está en marcha.

**Campaña activa**

```bash
env/bin/python scripts/monitor_campaign.py --once
systemctl --user status causal-fidelity-study.service --no-pager
```

El primer comando lee los archivos de progreso y funciona aunque no haya servicio
systemd; el segundo corresponde a la instalación local. Las ejecuciones se guardan
en `runs/study-20260925/`. El servicio genera 7B, Distill-32B y QwQ-32B en secuencia.
La evaluación del razonamiento tiene etapas propias: terminar la generación no
significa terminar todo el estudio.

La campaña verifica `configs/study.json` y ocho archivos científicos antes de
cada etapa: `experiment.py`, `dataset_builder.py`, `evaluation.py`, `llm_judge.py`,
`rejudge.py`, `analyze.py`, `compare.py` y `scripts/audit_run.py`. También deben
mantenerse los datos y revisiones de modelos. No reformatear ni modificar estos
archivos mientras siga esta campaña: una diferencia de huella impide continuar.
Las notas, nuevas pruebas y preparación de Git no cambian estas huellas.

No ejecutar otro generador ni iniciar el evaluador en esta misma máquina durante
la generación actual. La memoria se comparte y ya hubo fallos por ejecución
concurrente. La [nota de recuperación](recovery-20261017.md) documenta la exclusión
del evaluador y la restauración que realiza el servicio. Para explorar cambios,
usar otro checkout y recursos disponibles, con un directorio de salida nuevo.

**Preparar otro entorno**

Usar Python 3.12 y un entorno aislado. Instalar primero una distribución de PyTorch
compatible con el hardware, después las dependencias del proyecto. El entorno
original está descrito en [environment.txt](../artifacts/environment.txt); las
versiones principales comprobadas figuran en [requirements.txt](../requirements.txt).
La lista completa del entorno es evidencia, no un instalador portable entre máquinas.

```bash
python3 -m venv env
# Instalar aquí el PyTorch apropiado para el hardware.
env/bin/python -m pip install -r requirements.txt
env/bin/python -m pytest -q
```

Los tests usan tensores y un modelo diminuto creado en CPU, sin descargar pesos ni
llamar a servicios. La generación de los modelos NF4 sí requiere CUDA. El notebook
tutorial adapta la precisión al hardware y guarda esa adaptación; no garantiza
resultados idénticos a los de la máquina del estudio.

**Datos y una ejecución pequeña**

El dataset seleccionado ya está incluido en `artifacts/dataset.json`, junto con
revisión, semilla, particiones y checksum. Usarlo directamente para reproducir el
estudio. `dataset_builder.py` consulta la revisión disponible al ejecutarse: volver
a construirlo no garantiza recuperar la misma revisión y sobrescribe su salida
por defecto. Para explorar otra selección, usar `--output` y una ejecución distinta.

En otra máquina preparada, o cuando los recursos estén libres:

```bash
env/bin/python scripts/download_models.py distill-1.5b
env/bin/python experiment.py --config configs/smoke.json \
  --model distill-1.5b --output runs/smoke-new
env/bin/python scripts/audit_run.py runs/smoke-new
env/bin/python analyze.py runs/smoke-new
env/bin/python robustness.py runs/smoke-new
```

`smoke.json` usa 16 pares de construcción y cuatro de calibración. Es una prueba
de funcionamiento. Los pesos se descargan con revisión fija; el generador lee
solo la caché local. La auditoría necesita también el `config.json` del checkpoint
en esa caché. Un cambio de código, datos o configuración exige una salida nueva.
La misma invocación puede reanudar una ejecución interrumpida con identidad intacta.

**Generación del estudio en un entorno nuevo**

`configs/study.json` contiene el protocolo fijado, con 500 pares de construcción,
50 de test, ocho capas, cuatro intensidades y controles. Por ejemplo:

```bash
env/bin/python scripts/download_models.py distill-32b
env/bin/python experiment.py --config configs/study.json \
  --model distill-32b --output runs/study-new/distill-32b
env/bin/python scripts/audit_run.py runs/study-new/distill-32b --require-test
env/bin/python analyze.py runs/study-new/distill-32b
env/bin/python robustness.py runs/study-new/distill-32b
```

El lanzador `scripts/run_campaign.py` automatiza las etapas y requiere un evaluador
disponible durante las fases de juicio. `scripts/start_study.py` es el arranque
histórico ligado a la calibración local y a systemd; no es el punto de entrada
general para un clon nuevo. No volver a usarlo para la campaña actual.

**Evaluación del razonamiento**

Una vez liberada la memoria de generación y con un servidor compatible preparado:

```bash
env/bin/python rejudge.py --validate
env/bin/python rejudge.py --run runs/smoke-new
```

El comando usa por defecto el evaluador local en el puerto 8000. `--url`, `--model`
y `--backend` seleccionan otro servidor para una ejecución nueva. El manifiesto
impide mezclar evaluadores distintos al reanudar. La validación sintética y el
retest comprueban aspectos técnicos y consistencia, no validez humana. Consultar
la [auditoría conocida](judge-audit-20261102.md) antes de interpretar sus etiquetas.

**Resultados, figuras y manuscritos**

| Archivo de una ejecución | Uso |
|---|---|
| `records.jsonl` | Generaciones completas y metadatos por condición |
| `manifest.json`, `source/` | Identidad y código usado para generar |
| `status.json`, `audit.json` | Progreso y comprobación de cobertura |
| `summary.json`, `summary.csv`, `plots/` | Tablas y figuras descriptivas |
| `robustness.json` | Contrastes pareados, ausencias y ajuste por multiplicidad |
| `judgments.jsonl`, `judge-summary.json` | Juicios automáticos y sus errores |

GitHub contiene fuentes, datos seleccionados e informes fechados. `runs/` no se
distribuye con el repositorio. Para recalcular las figuras provisionales se necesita
la ejecución completa de 7B; no basta con las tablas ya exportadas.

Con esa ejecución disponible, `scripts/build_interim_figures.py` verifica las
cifras y genera figuras sin GPU. El [registro del manuscrito](manuscript-progress-20261102.md)
documenta cómo compilar el informe independiente del 2 de noviembre.

El manuscrito principal se construye a partir de ejecuciones disponibles:

```bash
env/bin/python scripts/install_tectonic.py
env/bin/python scripts/build_paper.py --runs runs/smoke-new --allow-incomplete
```

Produce `paper/main.pdf` y un archivo de fuentes expresamente marcado como borrador.
Para empaquetar código, datos y una ejecución con generación y evaluación ya
detenidas, usar:

```bash
env/bin/python scripts/package_reproduction.py --runs runs/smoke-new \
  --output dist/reproduction-draft.tar.gz --allow-incomplete
```

El modo borrador permite que falten modelos del estudio, pero no admite una
generación aún en curso. El paquete se verifica contra un manifiesto de checksums.
El archivo para arXiv contiene el manuscrito principal; el paquete de reproducción
incluye también el notebook y las figuras del informe provisional. Una publicación
final requiere completar el estudio, revisar las etiquetas, aportar autoría y
superar la auditoría técnica y la revisión científica.
