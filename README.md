# Causal interventions on chain-of-thought reasoning

¿Cambia la respuesta de un modelo si modificamos sus activaciones internas o el
razonamiento que escribe? Este proyecto estudia esa sensibilidad con ejemplos de
inferencia textual, respuestas originales por ejemplo y controles de igual magnitud.

**Estudio en curso — estado revisado el 2 de noviembre de 2026.** Distill-7B está
completo; Distill-32B sigue generando y QwQ-32B está pendiente. La comparación
principal entre los dos modelos de 32B todavía no tiene resultados finales.

- [Guía del experimento en español](docs/guia-experimento.md).
- [Notebook práctico en español](notebooks/causal_fidelity_colab.ipynb): demostración con 1,5B.
- [Protocolo científico](research-plan.md) y [configuración congelada](configs/study.json).
- [Informe provisional PDF](paper/interim-20261102.pdf) y [lectura de resultados](docs/interim-findings-20261102.md).
- [Reproducción y ejecución local](docs/reproduction.md).
- [Índice de documentación y auditorías](docs/README.md).

**Qué puede concluirse.** Los cambios de respuesta miden sensibilidad a las
intervenciones. No demuestran por sí solos que el razonamiento escrito cause la
decisión. Comparar checkpoints tampoco aísla un efecto del método de entrenamiento:
QwQ incluye ajuste supervisado y aprendizaje por refuerzo, y los modelos difieren
en datos y otras decisiones de entrenamiento.

**Evaluador automático.** La [auditoría de sus etiquetas](docs/judge-audit-20261102.md)
detectó errores. Los desacuerdos entre razonamiento y respuesta que marca no son
casos confirmados; necesitan revisión humana independiente. Las medidas de exactitud
y cambios de respuesta no dependen de esas etiquetas.

**Empezar con una demostración**

Sube el [notebook](notebooks/causal_fidelity_colab.ipynb) a Google Colab, selecciona
un entorno con GPU y ejecuta las celdas en orden. Incluye código y datos; descarga
el modelo de 1,5B y comienza con cuatro pares de calibración. Explica cada
intervención, muestra resultados y permite exportarlos. No requiere el evaluador
local. Esta demostración no reproduce el estudio completo de 32B.

**Consultar la campaña que ya está ejecutándose**

Desde la raíz del repositorio, en la máquina del estudio:

```bash
env/bin/python scripts/monitor_campaign.py --once
```

Sin `--once`, el monitor actualiza la pantalla hasta pulsar Ctrl+C. Solo lee el
estado; cerrarlo no detiene el experimento. La configuración y el código científico
permanecen congelados durante la campaña. Los comandos para nuevas ejecuciones y
las reglas de reanudación están en la [guía de reproducción](docs/reproduction.md).

**Archivos principales**

| Ruta | Contenido |
|---|---|
| `experiment.py` | Generación, intervenciones y guardado reanudable |
| `dataset_builder.py`, `artifacts/dataset.json` | Construcción y selección fija de pares SNLI |
| `configs/` | Revisiones de modelos y configuraciones |
| `evaluation.py`, `llm_judge.py`, `rejudge.py` | Lectura de respuestas y evaluación del razonamiento |
| `analyze.py`, `robustness.py`, `compare.py` | Estadística, controles y comparación entre modelos |
| `scripts/`, `tests/` | Ejecución, auditorías, exportación y pruebas |
| `paper/`, `artifacts/diagnostics/20261102/` | Borradores, tablas y figuras provisionales |

Los registros completos de las ejecuciones (`runs/`), pesos descargados, entornos
Python y paquetes de distribución se conservan fuera de Git. Los informes publicados
son capturas fechadas; un clon de GitHub no contiene los resultados brutos necesarios
para recalcularlos. Los antiguos archivos locales `results/*_v3.json` son salidas
de depuración inválidas como evidencia científica.

Para ejecutar las pruebas en el entorno preparado:

```bash
env/bin/python -m pytest -q
```

No descargan modelos, no llaman al evaluador ni requieren GPU. Los datos derivados
de SNLI conservan su [atribución CC BY-SA 4.0](artifacts/DATA_LICENSE.md); los pesos
y dependencias mantienen sus propias licencias.
