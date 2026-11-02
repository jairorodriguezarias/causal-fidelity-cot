"""Build the Spanish, self-contained Colab tutorial from the working tree.

Run this again after changing the experiment or its selected dataset. The embedded
snapshot makes the notebook usable before those files have been published to GitHub.
Only the explicit allowlist below is included; no weights, runs or credentials.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/causal_fidelity_colab.ipynb"
BUNDLE_FILES = (
    "experiment.py", "dataset_builder.py", "evaluation.py", "analyze.py",
    "robustness.py", "configs/models.json", "artifacts/dataset.json",
    "artifacts/DATA_LICENSE.md", "research-plan.md", "scripts/download_models.py",
    "scripts/audit_run.py", "tests/test_core.py", "tests/test_analysis.py",
    "tests/test_robustness.py", "pytest.ini",
)


def build():
    files = {name: (ROOT / name).read_text() for name in BUNDLE_FILES}
    raw = gzip.compress(json.dumps(files, ensure_ascii=False, sort_keys=True).encode(), mtime=0)
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(raw).decode()
    cells = []

    def add(kind, content, *, hidden=False):
        source = textwrap.dedent(content).strip() + "\n"
        cell = {"cell_type": kind, "id": f"cell-{len(cells):02d}",
                "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            compile(source, f"cell-{len(cells):02d}", "exec")
            cell.update(execution_count=None, outputs=[])
            if hidden:
                cell["metadata"]["cellView"] = "form"
        cells.append(cell)

    def md(content):
        add("markdown", content)

    def code(content, **kwargs):
        add("code", content, **kwargs)

    md(r"""
        # ¿Cambia la respuesta si intervenimos en el razonamiento?

        **Tutorial en español del proyecto `causal-fidelity-cot`.** Ejecutaremos un
        modelo abierto sobre inferencia textual, modificaremos sus activaciones y
        su razonamiento escrito, y examinaremos las respuestas con controles.

        **Para empezar:** sube este `.ipynb` a [Google Colab](https://colab.research.google.com/),
        selecciona **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**
        y ejecuta las celdas en orden. La configuración inicial usa
        **DeepSeek-R1-Distill-Qwen-1.5B**, 16 pares para construir la dirección y
        4 pares de calibración. Los pesos ocupan varios GB y se descargan una vez por
        sesión; la descarga y la generación pueden tardar varios minutos o más,
        según la GPU y la conexión. Se necesita Internet para instalar y descargar.

        El notebook contiene una copia del código y del subconjunto de SNLI de este
        proyecto: basta con subir **este archivo**. No requiere un servidor local,
        una API de pago ni montar Google Drive. Las celdas de código son desplegables.

        Es una **demostración exploratoria**, no la reproducción completa de la
        comparación de modelos de 32B. El modelo de 1,5B puede tener poca precisión
        en esta tarea. Una respuesta que cambia no prueba por sí sola que el texto
        generado describa fielmente el mecanismo interno del modelo.
    """)
    md(r"""
        ## 1. La pregunta y los datos

        Una *premisa* describe una situación; una *hipótesis* propone algo sobre ella.
        Usaremos dos etiquetas de SNLI y excluiremos los ejemplos neutrales:

        | Premisa | Hipótesis | Etiqueta esperada |
        |---|---|---|
        | Una persona corre por el parque. | Una persona se mueve. | `ENTAILMENT` (se deduce) |
        | Una persona corre por el parque. | No hay ninguna persona en el parque. | `CONTRADICTION` (contradice) |

        Este ejemplo es ilustrativo. Los datos reales y los prompts permanecen en
        inglés para conservar la tarea del repositorio.

        Cada par real comparte una premisa y tiene una hipótesis de cada clase.
        **Construcción (`train`)** sirve para calcular una dirección de activaciones;
        no actualizamos los pesos del modelo. **Calibración** permite explorar los
        ajustes. **Test** queda reservado para la evaluación final y este tutorial
        no lo utiliza. Las premisas no se repiten entre particiones.

        Fuente y licencia: [SNLI, Stanford](https://nlp.stanford.edu/projects/snli/),
        de Bowman, Angeli, Potts y Manning; datos derivados bajo
        [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
        La atribución completa se incluye en `artifacts/DATA_LICENSE.md`.
    """)
    md("""
        ## 2. Recuperar la copia del proyecto

        Esta celda verifica y descomprime el código y los datos incluidos. Puedes
        inspeccionarlos en el panel **Archivos** de Colab. No descarga resultados
        históricos ni pesos. Los archivos modificados manualmente se protegen frente
        a sobrescrituras al volver a ejecutar la celda.
    """)
    bootstrap = '''
        #@title Preparar el proyecto incluido en el notebook
        import base64
        import gzip
        import hashlib
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys

        BUNDLE_SHA256 = "__DIGEST__"
        BUNDLE_B64 = "__ENCODED__"
        packed = base64.b64decode(BUNDLE_B64, validate=True)
        if hashlib.sha256(packed).hexdigest() != BUNDLE_SHA256:
            raise ValueError("El contenido del notebook está dañado.")
        embedded = json.loads(gzip.decompress(packed))
        ROOT = Path(os.environ.get("CF_COLAB_BASE", "/content/causal-fidelity-cot")) / BUNDLE_SHA256[:12]
        ROOT.mkdir(parents=True, exist_ok=True)
        for name, content in embedded.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Ruta inválida: {name}")
            path = ROOT / relative
            if path.exists() and path.read_text() != content:
                # La única modificación automática permitida es la adaptación de precisión.
                stamp = ROOT / "colab-adaptation.json"
                allowed = json.loads(stamp.read_text()) if stamp.exists() else {}
                if name != "experiment.py" or hashlib.sha256(path.read_bytes()).hexdigest() != allowed.get("adapted_sha256"):
                    raise RuntimeError(f"{path} tiene cambios manuales; usa otra carpeta CF_COLAB_BASE.")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        source_manifest = {
            "repository": "https://github.com/jairorodriguezarias/causal-fidelity-cot",
            "bundle_sha256": BUNDLE_SHA256,
            "source_sha256": {name: hashlib.sha256(content.encode()).hexdigest()
                              for name, content in embedded.items()},
        }
        (ROOT / "colab-bundle.json").write_text(json.dumps(source_manifest, indent=2))
        os.chdir(ROOT)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        print("Proyecto:", ROOT)
        print("Archivos verificados:", len(embedded))
    '''
    code(bootstrap.replace("__DIGEST__", digest).replace("__ENCODED__", encoded), hidden=True)
    md("""
        ## 3. Preparar las dependencias y comprobar la GPU

        Usamos un entorno Python separado que conserva el PyTorch/CUDA de Colab,
        para evitar reinstalar sus controladores o depender de reiniciar el kernel.
        Las versiones principales coinciden con las registradas por el proyecto;
        guardaremos además todas las versiones efectivamente instaladas.

        La copia para Colab selecciona **FP16** cuando la GPU no tiene soporte nativo
        para BF16, y **BF16** cuando sí lo tiene. Esta adaptación queda registrada y
        forma parte del código guardado con los resultados. No implica resultados
        numéricamente idénticos entre GPU diferentes.
    """)
    code(r'''
        #@title Instalar dependencias (primera ejecución: puede tardar unos minutos)
        import importlib.metadata

        ENV = ROOT / ".colab-env"
        PYTHON = str(ENV / "bin/python")
        if not Path(PYTHON).exists():
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(ENV)], check=True)
        torch_version = importlib.metadata.version("torch")
        constraint = ROOT / "colab-constraints.txt"
        constraint.write_text(f"torch=={torch_version}\n")
        dependencies = [
            "transformers==5.15.1", "accelerate==1.14.0", "datasets==5.0.1",
            "huggingface-hub==1.28.0", "bitsandbytes==0.50.1",
            "numpy>=1.26", "matplotlib>=3.7", "pytest>=8",
        ]
        subprocess.run([PYTHON, "-m", "pip", "install", "--disable-pip-version-check",
                        "-c", str(constraint), *dependencies], check=True)

        def run_python(*args):
            subprocess.run([PYTHON, "-u", *map(str, args)], cwd=ROOT, check=True)

        run_python("-c", "from transformers import Qwen2ForCausalLM; print('Importación del modelo: OK')")
        print("Dependencias listas; el kernel del notebook no necesita reiniciarse.")
    ''')
    code(r'''
        #@title Verificar hardware y adaptar la precisión
        hardware_code = """
        import json, torch
        if not torch.cuda.is_available():
            raise RuntimeError('Activa una GPU en Entorno de ejecución > Cambiar tipo de entorno de ejecución.')
        info = {
            'gpu': torch.cuda.get_device_name(0),
            'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'dtype': 'bfloat16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'float16',
        }
        print(json.dumps(info))
        """
        import textwrap
        detected = subprocess.check_output([PYTHON, "-c", textwrap.dedent(hardware_code)], text=True)
        HARDWARE = json.loads(detected)
        print(json.dumps(HARDWARE, indent=2))

        # Solo cambia la precisión de carga/cómputo; la lógica experimental se conserva.
        original = embedded["experiment.py"]
        replacements = {
            "torch.bfloat16 if torch.cuda.is_available() else torch.float32":
                f"torch.{HARDWARE['dtype']} if torch.cuda.is_available() else torch.float32",
            "bnb_4bit_compute_dtype=torch.bfloat16": f"bnb_4bit_compute_dtype=torch.{HARDWARE['dtype']}",
        }
        adapted = original
        for old, new in replacements.items():
            if adapted.count(old) != 1:
                raise RuntimeError("El código fuente cambió: revisa la adaptación de precisión.")
            adapted = adapted.replace(old, new)
        (ROOT / "experiment.py").write_text(adapted)
        ADAPTATION = {
            "reason": "Select a native GPU dtype for this Colab runtime",
            "hardware": HARDWARE,
            "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "adapted_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        }
        (ROOT / "colab-adaptation.json").write_text(json.dumps(ADAPTATION, indent=2))
    ''')
    md("""
        ## 4. Configurar una ejecución pequeña

        Empieza por **demo**. **barrido** amplía la muestra, prueba tres capas y varias
        intensidades, incluido `alpha=0` como comprobación sin perturbación. Su coste
        crece con ejemplos × capas × intensidades × condiciones. La celda calcula
        un máximo de generaciones; las intervenciones textuales pueden ser menos
        si no hay razonamientos completos.

        El modelo de **7B** es opcional y se carga en NF4 (4 bits). Necesita descargar
        más pesos y no garantiza que quepa en cualquier GPU. Los modelos de 32B del
        estudio completo quedan fuera de este tutorial. El lote inicial es de uno
        para reducir memoria. Si cambias la configuración, se usa otra carpeta de
        resultados; el código original comprueba la identidad antes de reanudar.
    """)
    code(r'''
        #@title Modelo y tamaño del experimento
        MODEL = "distill-1.5b" #@param ["distill-1.5b", "distill-7b"]
        PROFILE = "demo" #@param ["demo", "barrido"]
        MAX_NEW_TOKENS = 2048 #@param {type:"integer"}
        SEED = 42 #@param {type:"integer"}
        RUN_TAG = "intento-1" #@param {type:"string"}

        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_TAG):
            raise ValueError("RUN_TAG solo admite letras, números, guion y guion bajo.")
        if MAX_NEW_TOKENS < 128:
            raise ValueError("Usa al menos 128 tokens; 2048 es el valor inicial.")
        profiles = {
            "demo": {"train_pairs": 16, "eval_pairs": 4, "layer_fractions": [0.5], "alphas": [1.0]},
            "barrido": {"train_pairs": 64, "eval_pairs": 12, "layer_fractions": [0.0, 0.5, 1.0],
                        "alphas": [0.0, 1.0, 4.0]},
        }
        CONFIG = {
            **profiles[PROFILE], "evaluation_split": "calibration", "seed": SEED,
            "modes": ["steer", "orthogonal", "gaussian"], "control_replicates": 1,
            "batch_size": 1, "k_tokens": 15,
            "generation": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": True,
                           "temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "text_interventions": True, "truncation_fractions": [0.0, 0.5, 1.0],
            "swap_cot": True, "answer_max_tokens": 128,
        }
        MODELS = json.loads((ROOT / "configs/models.json").read_text())
        MODEL_CONFIG = MODELS[MODEL]
        freeze = subprocess.check_output([PYTHON, "-m", "pip", "freeze"], text=True)
        identity = {"config": CONFIG, "model": MODEL_CONFIG, "bundle": BUNDLE_SHA256,
                    "adaptation": ADAPTATION, "environment_sha256": hashlib.sha256(freeze.encode()).hexdigest()}
        run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        RUN_DIR = ROOT / "runs" / f"colab-{MODEL}-{PROFILE}-{RUN_TAG}-{run_id}"
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH = RUN_DIR / "config.json"
        CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2))
        (RUN_DIR / "environment.txt").write_text(freeze)
        (RUN_DIR / "colab-context.json").write_text(json.dumps(identity, indent=2))
        activation_cells = len(CONFIG["layer_fractions"]) * len(CONFIG["alphas"]) * len(CONFIG["modes"])
        base_and_activation = CONFIG["eval_pairs"] * (2 + activation_cells)
        max_text = CONFIG["eval_pairs"] * (len(CONFIG["truncation_fractions"]) + 2)
        print("Modelo:", MODEL_CONFIG["id"], "| Cuantización:", MODEL_CONFIG["quantization"])
        print("Generaciones baseline + activaciones:", base_and_activation)
        print("Generaciones textuales como máximo:", max_text)
        print("Límite superior de tokens nuevos:", base_and_activation * MAX_NEW_TOKENS + max_text * CONFIG["answer_max_tokens"])
        print("Resultados:", RUN_DIR)
        print(json.dumps(CONFIG, indent=2))
    ''')
    md("""
        ## 5. Mirar los datos y verificar la implementación

        La huella del dataset verifica su contenido; la validación comprueba también
        la separación de premisas. Mostramos únicamente ejemplos de calibración.
        Los tests pequeños verifican el parser, los hooks, las normas de los controles
        y el tratamiento estadístico de resultados ausentes. No descargan modelos.
    """)
    code(r'''
        from dataset_builder import load_contrast_dataset
        from IPython.display import display, HTML, Code, Image, FileLink
        from html import escape

        def show_table(rows):
            if not rows:
                print("No hay filas disponibles.")
                return
            columns = list(rows[0])
            def render(value):
                if value is None:
                    return "—"
                if isinstance(value, float):
                    return f"{value:.3f}"
                return escape(str(value))
            header = "".join(f"<th>{escape(str(c))}</th>" for c in columns)
            body = "".join("<tr>" + "".join(f"<td>{render(row.get(c))}</td>" for c in columns) + "</tr>" for row in rows)
            display(HTML(f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"))

        DATASET = load_contrast_dataset(ROOT / "artifacts/dataset.json")
        show_table([{"partición": split, "pares disponibles": len(pairs)}
                    for split, pairs in DATASET["splits"].items()])
        print("Revisión SNLI:", DATASET["revision"])
        print("SHA-256 de los pares:", DATASET["sha256"])
        show_table(DATASET["splits"]["calibration"][:2])
        run_python("-m", "pytest", "-q", "tests/test_core.py", "tests/test_analysis.py", "tests/test_robustness.py")
    ''')
    md(r"""
        ## 6. ¿Qué significa intervenir en las activaciones?

        En cada capa elegida, medimos la activación del último token del prompt,
        justo antes de generar razonamiento. Con los pares de construcción calculamos:

        $$v_\ell = \operatorname{media}_i(h_{\ell,i}^{contra})
                 - \operatorname{media}_i(h_{\ell,i}^{entail}).$$

        Durante las primeras $K=15$ predicciones de tokens, incluida la que produce
        el primer token, aplicamos $h'_{\ell,t}=h_{\ell,t}+\alpha v_\ell$.
        `alpha=1` significa añadir una vez la diferencia media; **no** una desviación
        estándar. Los pesos permanecen fijos. Una dirección positiva apunta hacia
        la media de contradicción, pero eso no garantiza una respuesta contradictoria.

        | Condición | Qué añadimos | Para qué sirve |
        |---|---|---|
        | `baseline` | Nada | Respuesta original de cada ejemplo |
        | `steer` | La dirección aprendida | Medir el efecto de esa dirección |
        | `orthogonal` | Vector aleatorio perpendicular, con la misma norma | Controlar perturbaciones ajenas a esa dirección |
        | `gaussian` | Vector gaussiano, normalizado a la misma norma | Controlar perturbaciones de igual magnitud |

        Cada intervención de activaciones se evalúa sobre la hipótesis de entailment.
        Las dos clases se usan para medir la competencia del baseline. Las semillas
        se comparten entre condiciones según el protocolo, pero una sola semilla y
        cuatro ejemplos no permiten conclusiones robustas.

        La siguiente celda muestra el hook que ejecutaremos: solo modifica la última
        posición, cuenta las primeras `K` llamadas y después deja de intervenir.
    """)
    code('''
        import ast
        engine_source = (ROOT / "experiment.py").read_text()
        tree = ast.parse(engine_source)
        hook_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SteeringHook")
        display(Code(ast.get_source_segment(engine_source, hook_node), language="python"))
    ''')
    md(r"""
        ## 7. ¿Qué significa intervenir en el razonamiento escrito?

        El modelo produce un texto de razonamiento (*chain of thought*, CoT) y una
        respuesta final. El parser requiere un cierre `</think>` y un único campo
        final reconocible como `FINAL: ENTAILMENT` o `FINAL: CONTRADICTION`.

        Partimos de baselines con razonamiento completo y respuesta válida; **no**
        exigimos que esa respuesta sea correcta para crear las variantes textuales:

        | Variante | Prefijo usado antes de regenerar la respuesta |
        |---|---|
        | `truncate_0.0` | Ningún token del razonamiento original |
        | `truncate_0.5` | La primera mitad de sus tokens |
        | `truncate_1.0` | Todo el razonamiento original |
        | `swap_contradiction_cot` | Razonamiento de la hipótesis contradictoria de la misma premisa |
        | `shuffle_cot` | Tokens originales en orden aleatorio |

        Forzamos el cierre de razonamiento y generamos una respuesta nueva, de hasta
        128 tokens. `truncate_1.0` es el control principal: incluso reconstruir todo
        el prefijo y muestrear otra respuesta puede cambiar el resultado. El análisis
        complementario compara las demás variantes con ese control.

        El intercambio exige además que el baseline contradictorio sea válido.
        Estos filtros introducen selección: los resultados textuales describen los
        ejemplos elegibles, no automáticamente toda la muestra. Barajar tokens también
        puede destruir la gramática; no es una intervención puramente semántica.
    """)
    md("""
        ## 8. Descargar solo el modelo elegido y ejecutar

        La revisión del checkpoint está fijada en `configs/models.json`. La descarga
        usa esa revisión y el experimento lee la caché local. No se descargan los otros
        modelos. Se conserva una generación por línea en `records.jsonl`, junto con
        tokens, prompt, etiqueta esperada, condición y semilla.

        La ejecución ocurre en otro proceso; cuando termina, libera la memoria del
        modelo. Si interrumpes la celda conservando la sesión, puedes repetirla para
        reanudar con los mismos ajustes. El proceso hijo se detiene al interrumpir.
    """)
    code('''
        #@title Descargar el checkpoint fijado
        run_python("scripts/download_models.py", MODEL)
    ''')
    code(r'''
        #@title Ejecutar baseline, activaciones y modificaciones del razonamiento
        import time
        command = [PYTHON, "-u", "experiment.py", "--config", str(CONFIG_PATH),
                   "--model", MODEL, "--output", str(RUN_DIR),
                   "--dataset", str(ROOT / "artifacts/dataset.json")]
        started = time.monotonic()
        log_path = RUN_DIR / "execution.log"
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if returncode:
            raise RuntimeError(f"El experimento terminó con código {returncode}; consulta {log_path}.")
        print(f"Duración de esta ejecución: {(time.monotonic() - started)/60:.1f} minutos")
        print((RUN_DIR / "status.json").read_text())
    ''')
    md("""
        ## 9. Auditar y resumir

        La auditoría comprueba cobertura de condiciones, hashes del código, IDs,
        parser y número de pasos del hook. Después generamos tablas, intervalos y
        figuras. Pasar la auditoría indica coherencia técnica; no demuestra una
        hipótesis científica. No usamos un juez LLM: el endpoint local del proyecto
        no existe en Colab. Este notebook estudia las respuestas y sus cambios, sin
        inventar puntuaciones de fidelidad del razonamiento.
    """)
    code('''
        run_python("scripts/audit_run.py", RUN_DIR)
        run_python("analyze.py", RUN_DIR)
        run_python("robustness.py", RUN_DIR)
        SUMMARY = json.loads((RUN_DIR / "summary.json").read_text())
        ROBUSTNESS = json.loads((RUN_DIR / "robustness.json").read_text())
        AUDIT = json.loads((RUN_DIR / "audit.json").read_text())
        print("Auditoría:", "OK" if AUDIT["passed"] else AUDIT["errors"])
    ''')
    md(r"""
        ## 10. Leer los resultados sin ocultar ausencias

        Primero miramos la competencia del baseline **por clase**. Un modelo que
        responde siempre contradicción puede parecer muy sensible si omitimos su
        punto de partida. La precisión cuenta respuestas ausentes como no acertadas.

        La **tasa de cambio** considera ejemplos cuyo baseline respondió entailment:

        $$\text{flip} = \frac{\#(baseline=E,\ intervención=C)}
                             {\#(baseline=E,\ intervención\ válida)}.$$

        Si no hay denominador, la tasa es **indefinida**, no cero. Los límites por
        ausencias asignan a las respuestas desconocidas todos los escenarios posibles;
        no son intervalos de confianza. Los intervalos Wilson describen incertidumbre
        muestral; con cuatro ejemplos serán muy amplios.

        `unfinished_thought`: no apareció el cierre de razonamiento.
        `invalid_answer`: no hay un único campo final reconocible.
        `empty_thought`: el razonamiento está vacío, aunque puede haber etiqueta válida.
        Estos estados se conservan por separado. Si abundan las salidas sin terminar,
        considera aumentar el límite de tokens y crear otra ejecución de calibración.
    """)
    code('''
        show_table([
            {"baseline": variant, "n": data["accuracy"]["n"],
             "precisión": data["accuracy"]["rate"], "estados": data["parse_status"],
             "finalización": data["finish_reason"]}
            for variant, data in ROBUSTNESS["baseline"]["by_class"].items()
        ])
        show_table([
            {"condición": row["name"], "n": row["n"], "respuestas válidas": row["valid_answers"],
             "precisión": row["accuracy_all"]["rate"],
             "baseline E": row["baseline_entailment_n"],
             "cambios / casos completos": row["flip_complete_cases"]["rate"],
             "ausentes entre baseline E": row["missing_among_baseline_entailment"],
             "límite inferior": row["flip_missingness_bounds"]["low"],
             "límite superior": row["flip_missingness_bounds"]["high"],
             "estados": row["status_counts"]}
            for row in SUMMARY["conditions"]
        ])
        baseline_summary = next(row for row in SUMMARY["conditions"] if row["name"] == "baseline")
        if baseline_summary["baseline_entailment_n"] == 0:
            print("No hay baselines entailment correctos: no se puede estimar la tasa de cambio condicionada.")
        if not ROBUSTNESS["text_contrasts_against_full_prefix"]:
            print("No hay comparaciones textuales elegibles. Revisa los estados del baseline.")
    ''')
    md("""
        ## 11. Comparar con los controles y visualizar

        Una diferencia positiva significa más respuestas contradictorias con la
        intervención que con el control, **entre pares completos**. El bootstrap
        mantiene unidos los resultados de cada ejemplo. No tratamos cada combinación
        de capa e intensidad como si fuera un ejemplo independiente.

        El análisis complementario añade límites por ausencias, prueba exacta pareada
        y ajuste Holm en la familia de contrastes de activaciones del modelo. No
        convierte esta muestra mínima en un estudio confirmatorio.

        En las figuras, una celda vacía puede significar una tasa no calculable.
        La curva de truncamiento usa precisión sobre sus ejemplos elegibles; consulta
        también los contrastes contra `truncate_1.0` para separar el efecto de reconstruir
        el prefijo. En modo demo solo habrá una capa y una intensidad.
    """)
    code('''
        show_table([
            {"intervención": row["treatment"], "control": row["control"],
             "pares completos": row["complete_pairs"], "pares ausentes": row["missing_pairs"],
             "diferencia": row["complete_case_difference"]["difference"],
             "IC bootstrap inferior": row["complete_case_difference"]["low"],
             "IC bootstrap superior": row["complete_case_difference"]["high"],
             "límites por ausencias": row["missingness_bounds"],
             "p ajustado Holm": row["holm_adjusted_p_within_model_activation_family"]}
            for row in ROBUSTNESS["activation_contrasts"]
        ])
        show_table([
            {"texto": row["treatment"], "control": row["control"],
             "solo baseline correcto": row["baseline_correct_only"],
             "pares completos": row["complete_pairs"], "pares ausentes": row["missing_pairs"],
             "diferencia": row["complete_case_difference"]["difference"],
             "límites por ausencias": row["missingness_bounds"]}
            for row in ROBUSTNESS["text_contrasts_against_full_prefix"]
        ])
        for plot_path in sorted((RUN_DIR / "plots").glob("*.png")):
            print(plot_path.name)
            display(Image(filename=str(plot_path)))
    ''')
    md("""
        ## 12. Inspeccionar un ejemplo completo

        Selecciona un índice de la muestra y una condición. Mostramos el baseline,
        el texto suministrado si es una intervención textual y toda la continuación
        generada. No recortamos silenciosamente el razonamiento para evaluarlo.

        Compara: ¿cambió la etiqueta?, ¿se completó el razonamiento?, ¿ya era erróneo el
        baseline? Leer un caso ayuda a interpretar los números, pero no sustituye la
        comparación de toda la muestra.
    """)
    code(r'''
        #@title Caso para inspección
        EXAMPLE_INDEX = 0 #@param {type:"integer"}
        CONDITION = "auto" #@param {type:"string"}

        records = [json.loads(line) for line in (RUN_DIR / "records.jsonl").read_text().splitlines()]
        example_ids = json.loads((RUN_DIR / "manifest.json").read_text())["eval_ids"]
        if not 0 <= EXAMPLE_INDEX < len(example_ids):
            raise ValueError(f"EXAMPLE_INDEX debe estar entre 0 y {len(example_ids)-1}.")
        selected = [r for r in records if r["example_id"] == example_ids[EXAMPLE_INDEX] and r["variant"] == "entail"]
        print("Condiciones disponibles:", [r["condition"]["name"] for r in selected])
        if CONDITION == "auto":
            chosen_name = next(r["condition"]["name"] for r in selected if r["condition"].get("mode") == "steer")
        else:
            chosen_name = CONDITION
        chosen = next((r for r in selected if r["condition"]["name"] == chosen_name), None)
        if chosen is None:
            raise ValueError("La condición no existe para este ejemplo; elige una de la lista.")
        base = next(r for r in selected if r["condition"]["kind"] == "baseline")
        print("Premisa:", base["context"]["premise"])
        print("Hipótesis:", base["context"]["hypothesis"])
        print("Etiqueta esperada:", base["gold_label"])
        for record in [base, chosen]:
            print("\nCONDICIÓN:", record["condition"]["name"])
            print("Etiqueta:", record["parsed"]["label"], "| Estado:", record["parsed"]["status"])
            if record["condition"]["kind"] == "text":
                print("PROMPT CON EL RAZONAMIENTO INTERVENIDO:\n", record["prompt"])
            print("CONTINUACIÓN COMPLETA:\n", record["generated_text"])
    ''')
    md("""
        ## 13. Guardar y descargar

        Los archivos de la máquina de Colab son temporales. Crea y descarga el ZIP
        antes de cerrar la sesión. Incluye la ejecución actual, código usado, datos,
        atribución, configuraciones, versiones y figuras PNG/PDF; excluye pesos y
        entornos instalados. No contiene resultados de otras ejecuciones.

        Puedes ejecutar esta celda también tras interrumpir el experimento, siempre
        que hayas llegado a configurar `RUN_DIR`: guardará lo que exista. Reanudar
        con la misma sesión es repetir la celda de ejecución. En una sesión nueva
        debes recuperar los archivos del ZIP en la misma estructura y conservar
        exactamente código, GPU, dependencias y ajustes; si cambian, usa una ejecución
        nueva. Una última línea JSONL incompleta requiere revisión antes de reanudar.
    """)
    code('''
        #@title Crear y descargar el ZIP de esta ejecución
        import zipfile
        from IPython.display import display, FileLink

        archive_path = ROOT / f"{RUN_DIR.name}.zip"
        export_files = set(path for path in RUN_DIR.rglob("*") if path.is_file())
        export_files.update(ROOT / name for name in embedded)
        export_files.update(ROOT / name for name in ["colab-bundle.json", "colab-adaptation.json", "colab-constraints.txt"])
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(export_files):
                if path.is_file():
                    archive.write(path, path.relative_to(ROOT))
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("La comprobación del ZIP ha fallado.")
        print("Archivo:", archive_path.name, f"({archive_path.stat().st_size / 2**20:.2f} MiB)")
        try:
            from google.colab import files as colab_files
        except ImportError:
            display(FileLink(str(archive_path)))
        else:
            colab_files.download(str(archive_path))
    ''')
    md("""
        ## 14. Cómo continuar y resolver problemas

        | Situación | Qué hacer |
        |---|---|
        | No se detecta CUDA | Activa GPU en el tipo de entorno y ejecuta desde el principio. |
        | Falla la instalación o importación | Usa una sesión nueva; conserva el error. No continúes con dependencias a medias. |
        | Se agota la memoria de GPU | Vuelve a 1,5B, conserva lote 1 y reduce el límite de tokens; ejecuta con otro identificador. |
        | La descarga se interrumpe | Repite la celda de descarga; se reutiliza la caché disponible. |
        | Hay muchos `unfinished_thought` | Prueba más tokens en una nueva ejecución de calibración y revisa ambos baselines. |
        | No hay tasa de cambio o gráficos textuales | Comprueba competencia y elegibilidad; un denominador vacío no permite estimar el efecto. |
        | El manifiesto no coincide al reanudar | Usa un `RUN_TAG` nuevo; no mezcles resultados de configuraciones distintas. |
        | La sesión se pierde | El disco temporal puede desaparecer; conserva el ZIP descargado. |

        **Extensiones sugeridas:** probar `alpha=0`, ampliar el barrido, comparar
        intensidades positivas y negativas, repetir con varias semillas y después
        usar 7B con controles equivalentes. Ajusta en calibración y fija las decisiones
        antes de abrir test. No selecciones solo los ejemplos que cambian de respuesta.

        Para la comparación completa, consulta `research-plan.md` y el repositorio.
        Comparar checkpoints destilados y modelos con SFT/RL no aísla por sí solo el
        método de entrenamiento: también varían datos y otras decisiones. Tampoco
        basta con que el razonamiento y la respuesta concuerden para demostrar
        mediación causal a través del texto.

        Referencias de uso:
        [repositorio](https://github.com/jairorodriguezarias/causal-fidelity-cot),
        [modelo 1,5B y licencia](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),
        [instalación de Transformers](https://huggingface.co/docs/transformers/installation).
        La disponibilidad de GPU y la duración de sesión dependen de los
        [límites de Colab](https://research.google.com/colaboratory/faq.html).
    """)
    notebook = {
        "nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"Created {OUTPUT.relative_to(ROOT)} ({len(cells)} cells, {OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
