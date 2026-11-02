# Revisión para GitHub — 2 de noviembre de 2026

Preparación del código y de la documentación mientras la campaña sigue ejecutándose.
La rama local es `prepare-github-20261102`. Esta revisión no convierte los resultados
parciales en una publicación científica terminada.

**Alcance**

Se revisaron el generador y sus hooks, construcción de datos, parser, evaluador,
análisis pareados, reanudación, gestión de recursos, pruebas, notebook, scripts de
documentos y paquetes, configuraciones y archivos candidatos a Git. Los ocho
archivos científicos congelados y la configuración siguen coincidiendo con
`runs/study-20260925/campaign.json`. También se comprobaron las huellas de datos,
revisiones de modelos, congelación del protocolo y scripts operativos antes y
después del trabajo. El detalle está en
[github-preparation.json](../artifacts/diagnostics/20261102/github-preparation.json).

No se reformateó ni refactorizó el código congelado: el lanzador comprueba sus
bytes antes de cada etapa. Las correcciones afectan a documentación, pruebas y
empaquetado, sin cambiar las generaciones ni los juicios del estudio.

**Cambios preparados**

- README más corto, en español, con estado fechado, enlaces y límites de interpretación.
- Guía del experimento, ejemplo de hooks, instrucciones de reproducción e índice
  que distingue documentación de uso y registros históricos de esta máquina.
- Dependencias principales fijadas a las versiones instaladas y probadas. PyTorch
  conserva selección por hardware; la versión del estudio está documentada.
- Exclusión de configuración local `.env*`, distribuciones y PDF principal generado.
  El archivo vacío `.envs` sale de Git y se conserva localmente. Pesos, entornos y
  registros de ejecución continúan excluidos. Se conserva el PDF provisional fechado.
- La prueba de rechazo de un smoke como estudio final usa ahora datos sintéticos
  y un `config.json` mínimo. Se ejecuta también en un clon sin resultados ni caché.
- Pruebas para rechazar generación activa en auditorías y paquetes, comprobar la
  copia de código incluida en el notebook y verificar los archivos de reproducción.
- Los paquetes incluyen el notebook y los diagnósticos y figuras anidados que
  necesita el informe provisional. El archivo de fuentes para arXiv contiene solo
  el manuscrito principal y sus dependencias, evitando incluir un informe separado
  sin sus figuras externas.
- Informes antiguos de pruebas y auditoría etiquetados como evidencia histórica;
  se conserva su contenido original.

**Verificación realizada**

| Comprobación | Resultado |
|---|---|
| Suite antes de los cambios | 44 pruebas correctas |
| Suite final en el repositorio | 48 pruebas correctas |
| Suite en copia limpia, sin `runs/` ni `hf_cache/`, GPU deshabilitada y modo offline | 48 pruebas correctas, ninguna omitida |
| Notebook | 28 celdas, código compilable, salidas vacías y snapshot coincidente con fuentes |
| Reconstrucción del notebook | Bytes idénticos |
| Revisión de resultados 7B en salida temporal | 5.229 registros, 64 contrastes de activaciones y 13 textuales verificados |
| Documentos | Compilación del informe provisional y del borrador principal en copia temporal |
| Archivo de fuentes principal | Extraído y recompilado en otra carpeta sin el resto del repositorio |
| Paquete de reproducción | Creado con una copia del smoke terminado; verificación de checksums correcta |
| Fuentes, configuración y datos protegidos | Sin cambios |
| Candidatos a Git | Sin pesos, ejecuciones ni coincidencias con los patrones de credenciales revisados |

La comprobación de copia limpia reutilizó las dependencias del entorno preparado.
No acredita una instalación nueva en todos los sistemas ni una ejecución completa
en Colab. Los builds y paquetes de prueba se hicieron en `/tmp`, sin reemplazar
informes originales ni escribir en las ejecuciones activas. La búsqueda de patrones
de credenciales es una comprobación acotada, no una garantía absoluta.

**Hallazgos que afectan a la interpretación o a una versión futura**

- El evaluador automático tiene errores semánticos documentados. La revisión de
  asistente no sustituye una anotación humana ciega. Los juicios originales permanecen
  disponibles para auditoría; no se corrigen silenciosamente durante la campaña.
- El campo `seed` del evaluador registra la semilla base; los reintentos usan
  `seed + attempt`. El historial conserva intentos, pero no debe interpretarse ese
  campo como prueba de que todas las solicitudes usaron exactamente la misma semilla.
  Registrar la semilla efectiva requiere una versión posterior del evaluador.
- El constructor de datos consulta la revisión disponible en el momento de ejecutarse.
  Para reproducir este estudio se utiliza el dataset ya seleccionado y versionado.
- La auditoría de ejecución necesita el `config.json` del checkpoint en la caché.
  No es una auditoría autónoma de un archivo de registros aislado.
- El guardado rechaza una última línea JSONL interrumpida y pide reparación explícita;
  no descarta resultados silenciosamente. La reanudación no hace limpieza automática.
- La comprobación de paquetes rechaza generación activa mediante `status.json`.
  Antes de empaquetar un run terminado hay que asegurarse también de que ya no haya
  un evaluador escribiendo en él. No se empaquetó la campaña activa.
- Los informes parciales distribuidos no contienen todos los registros brutos.
  El estudio necesita terminar ambos modelos de 32B, su evaluación, la comparación
  y una revisión científica antes de producir una entrega final.

**Referencias contrastadas**

Se volvieron a comprobar los metadatos y el alcance de los resúmenes de
[Lanham et al.](https://arxiv.org/abs/2307.13702),
[Arcuschin et al.](https://arxiv.org/abs/2503.08679),
[Cox et al.](https://arxiv.org/abs/2603.01437),
[Jia et al.](https://arxiv.org/abs/2605.24286) y
[DeepSeek-R1](https://arxiv.org/abs/2501.12948).
La [ficha oficial de QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) confirma que
incluye ajuste supervisado y aprendizaje por refuerzo; también se consultó el
[anuncio del equipo](https://qwenlm.github.io/blog/qwq-32b/).
La página de SNLI agotó el tiempo de consulta en esta revisión; se conserva la
atribución y la auditoría bibliográfica previas, sin presentarla como reverificada.
