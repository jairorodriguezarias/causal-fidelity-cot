# Guía del experimento

Queremos medir cómo cambian las respuestas de un modelo cuando intervenimos en sus
activaciones internas o en su razonamiento escrito (*chain of thought*, CoT).
Los pesos del modelo permanecen fijos: no lo entrenamos de nuevo.

Para ejecutarlo como tutorial, usa el
[notebook en español](../notebooks/causal_fidelity_colab.ipynb). Para reproducir
ejecuciones locales, consulta [reproduction.md](reproduction.md).

**1. La tarea**

El modelo recibe una premisa y una hipótesis. Debe escribir un razonamiento y
terminar con `FINAL: ENTAILMENT` o `FINAL: CONTRADICTION`.

| Premisa | Hipótesis | Etiqueta esperada |
|---|---|---|
| Una persona corre por el parque. | Una persona se mueve. | ENTAILMENT: se deduce |
| Una persona corre por el parque. | No hay ninguna persona en el parque. | CONTRADICTION: contradice |

Este ejemplo es ilustrativo. Los datos reales y los prompts están en inglés.
Usamos un subconjunto binario de SNLI; excluimos la clase neutral.

**2. Modelos y datos**

La comparación principal enfrenta DeepSeek-R1-Distill-Qwen-32B con QwQ-32B.
Ambos usan los mismos ejemplos, ajustes y cuantización NF4. Los modelos destilados
de 1,5B y 7B aportan comprobaciones de implementación y estudios de menor tamaño.

| Partición | Pares disponibles | Uso |
|---|---:|---|
| Construcción | 500 | Calcular direcciones de activación |
| Calibración | 50 | Comprobar competencia y elegir ajustes |
| Test | 50 | Evaluación final con el protocolo fijado |

Cada par comparte una premisa y contiene una hipótesis de cada clase. Las premisas
no se repiten entre particiones. La calibración registrada del modelo de 7B usó
20 pares; no debe confundirse el tamaño disponible con el efectivamente utilizado.

**3. Respuestas originales**

Primero generamos una respuesta sin intervención para ambas hipótesis de cada
par de test: 100 respuestas por modelo. Son los *baselines*. Permiten medir la
competencia inicial y distinguir los cambios inducidos de los errores previos.

**4. Intervenciones sobre activaciones**

En cada capa seleccionada calculamos una dirección:

```text
dirección = activación media con contradicción − activación media con entailment
activación intervenida = activación original + alpha × dirección
```

Las activaciones se extraen al final del prompt usando los pares de construcción.
Durante la generación añadimos la dirección en las primeras 15 predicciones de
tokens, incluida la del primer token. `alpha=1` representa una vez la diferencia
media, no una desviación estándar ni una magnitud universal entre modelos.

Probamos ocho capas y cuatro intensidades: 0,5; 1; 2 y 4. Cada ejecución modifica
una capa. Las intervenciones se evalúan sobre las 50 hipótesis de entailment.

| Condición | Qué se añade |
|---|---|
| Baseline | Nada |
| Steering | La dirección calculada |
| Ortogonal | Una dirección aleatoria perpendicular, de igual norma |
| Gaussiana | Una dirección aleatoria normalizada a igual norma |

Los controles permiten comparar el efecto de esa dirección con el de otras
perturbaciones de igual magnitud. El barrido produce
50 × 8 × 4 × 3 = **4.800 generaciones por modelo**, además de baselines y pruebas
textuales. Hay una generación por celda y ejemplo; no estimamos variabilidad entre
múltiples semillas de generación.

**Cómo se usan los hooks**

Un hook intercepta la salida de una capa. `SteeringHook` suma `delta` a la última
posición de cada ejemplo del lote durante las primeras `k_tokens` llamadas.
En la primera llamada esa posición es el final del prompt; predice el primer
token generado. Las llamadas posteriores corresponden a la decodificación con caché.

El generador instala y retira el hook automáticamente mediante `steering_hook`.
Este es el patrón que ya usa `generate_batch`, con el modelo, los prompts
tokenizados y el vector previamente preparados:

```python
from experiment import intervention_delta, steering_hook

layer = 28  # índice desde cero; debe existir en el modelo
delta = intervention_delta(
    direction=vectors[layer],  # diferencia media calculada con datos de construcción
    alpha=2.0,
    mode="steer",             # también "orthogonal" o "gaussian"
    example_ids=example_ids,   # un identificador por fila de inputs
    seed=42,
    layer=layer,
)

with steering_hook(model, layer, delta, k_tokens=15):
    output = model.generate(**inputs, **generation_settings)
```

`delta` tiene forma `[tamaño_del_lote, dimensión_oculta]`. El hook adapta su
dispositivo y precisión a la activación. Al salir del bloque `with`, se retira
incluso si la generación falla. No modifica los pesos.

Para usarlo en el experimento basta con configurar `layer_fractions`, `alphas`,
`modes` y `k_tokens` en una configuración nueva y ejecutar `experiment.py`.
`layer_fractions` selecciona posiciones relativas entre 0 y 1; el índice se
calcula como `round(fracción × (número_de_capas − 1))`. La campaña actual ya tiene
sus hooks activos y sus ajustes congelados. Probar variantes requiere otra
ejecución y recursos disponibles; no añadir un segundo hook al generador existente.

**5. Intervenciones sobre el texto**

Mantenemos la pregunta de entailment y cambiamos el razonamiento suministrado
antes de regenerar la respuesta:

| Intervención | Prefijo que recibe el modelo |
|---|---|
| Acortar | Primer 0 %, 25 %, 50 %, 75 % o 100 % de los tokens originales |
| Sustituir | Razonamiento de la hipótesis contradictoria del mismo par |
| Barajar | Tokens del razonamiento original mezclados aleatoriamente |

Se cierra el bloque de pensamiento y se generan hasta 128 tokens nuevos. El
100 % es el control: reconstruir el prefijo y volver a muestrear una respuesta
también puede cambiarla. El 0 % no elimina todo cómputo ni impide cualquier forma
de razonamiento en la continuación.

Solo entran baselines con razonamiento completo y respuesta válida; no es necesario
que sean correctos. La sustitución exige además un baseline contradictorio válido.
Por eso el número de pruebas textuales varía entre modelos. Se analizan tanto todos
los casos elegibles como el subconjunto inicialmente correcto.

**6. Medición e interpretación**

Medimos exactitud, cambios hacia contradicción, diferencias pareadas frente a los
controles y frecuencia de respuestas incompletas o inválidas. Conservamos las
ausencias y mostramos cómo pueden afectar las conclusiones. Los intervalos y
contrastes usan ejemplos emparejados; miles de generaciones no equivalen a miles
de ejemplos independientes. El análisis complementario ajusta por comparaciones
múltiples la familia de contrastes de activaciones de cada modelo.

Un evaluador separado lee el razonamiento sin la respuesta final. Sus etiquetas
describen acuerdo textual. La [auditoría](judge-audit-20261102.md) detectó errores
y requiere revisión humana independiente antes de usar tasas de incoherencia.

Cambiar la respuesta mediante texto insertado demuestra sensibilidad a esa
intervención. No identifica automáticamente mediación causal a través del
razonamiento natural: el texto sustituido puede incluir una conclusión y contexto
incompatible. Que acortar el razonamiento no cambie una respuesta tampoco demuestra
que razonar sea inútil. La comparación de checkpoints no aísla el entrenamiento
supervisado frente al refuerzo: QwQ incluye ambos y cambian los datos de entrenamiento.

**7. Qué queda registrado**

Cada ejecución guarda los textos completos, tokens, prompts, condiciones, semillas,
motivos de parada y estados de parseo. Su manifiesto registra revisiones, datos,
configuración, versiones y huellas del código. La reanudación comprueba la identidad
y conserva las generaciones anteriores.

El [protocolo](../research-plan.md) y su
[congelación local](../artifacts/protocol-freeze.json) definen el estudio. El
[informe provisional](../paper/interim-20261102.pdf) presenta resultados parciales;
la comparación principal sigue pendiente.
