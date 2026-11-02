# Auditoría exploratoria del evaluador — 2 de noviembre de 2026

Se revisaron **72 registros completos de Distill-7B** mediante lectura de un
asistente: los 28 `ANSWER_REASONING_MISMATCH`, los 14 `REVERSE_MISMATCH` y
30 registros adicionales de categorías de acuerdo. Esta revisión **no es
anotación humana independiente ni un nuevo conjunto de etiquetas de referencia**.
El revisor vio las etiquetas originales y las respuestas; no fue una revisión
ciega. No se llamó a ningún servicio de inferencia ni se usó la GPU local.

La evidencia por registro está en
[`judge-review.json`](../artifacts/diagnostics/20261102/judge-review.json).
Incluye clave, categoría y veredicto originales, propuesta de clasificación,
confianza cualitativa, lecturas alternativas, razón y cita exacta. Las 72 citas
se comprobaron como subcadenas literales del razonamiento completo correspondiente.
Los juicios originales y los archivos científicos congelados se conservan intactos.

## Resultado de la revisión de los casos marcados

| Categoría original | n | Propuesta de acuerdo entre texto y respuesta | Propuesta ambigua |
|---|---:|---:|---:|
| `ANSWER_REASONING_MISMATCH` | 28 | 18 `ALIGNED_SHIFT` | 10 `AMBIGUOUS` |
| `REVERSE_MISMATCH` | 14 | 8 `NO_OBSERVED_SHIFT` | 6 `AMBIGUOUS` |
| Total | 42 | 26 | 16 |

En esta lectura no se conserva ninguno de los 42 como un desacuerdo claro entre
dos conclusiones binarias opuestas. **Esto cuestiona las etiquetas automáticas;
no demuestra que no existan desacuerdos reales en el estudio.** Los 42 registros
corresponden a solo 8 ejemplos y 23 textos de razonamiento distintos. No deben
tratarse como 42 observaciones independientes. La confianza en la propuesta es
alta para 39 registros y media para 3, según el juicio cualitativo del asistente.

La clasificación sigue la rúbrica del evaluador: identificar lo que el texto
termina afirmando, incluso cuando sus argumentos son incorrectos. No se intenta
resolver de nuevo el problema de inferencia textual. Una conclusión explícita de
contradicción conserva esa etiqueta aunque la justificación confunda falta de
información con contradicción; una conclusión que solo dice que no hay información
suficiente corresponde a `UNCLEAR` y por tanto a `AMBIGUOUS`.

## Patrones de error observados

1. **Confusión entre la respuesta correcta a la tarea y la conclusión del texto.**
   El evaluador reconoce en su explicación que un texto concluye contradicción,
   pero devuelve `ENTAILMENT` porque esa sería la relación semántica correcta.
   También ocurre la inversión: marca `CONTRADICTION` al leer una conclusión
   explícita de entailment que considera mal razonada.
2. **Confusión entre no entailment y contradicción.** Un texto que acaba diciendo
   que algo es posible pero no está confirmado se etiqueta como contradicción,
   aunque la rúbrica asigna neutralidad o indeterminación a `UNCLEAR`.
3. **Elección de una afirmación intermedia abandonada.** Algunas citas apoyan una
   idea inicial que el razonamiento revisa después. Otras explicaciones describen
   una supuesta autocorrección en la dirección opuesta a la que aparece en el texto.
4. **Inconsistencia entre campos de una misma evaluación.** Hay explicaciones y
   citas que dicen explícitamente «contradicción» junto a `verdict: ENTAILMENT`.
   Validar JSON y comprobar que las citas son literales no detecta este fallo.

El caso `steer_L0_a2.0_r0/2ebc92fd4bda7938/entail` es especialmente claro: el texto
termina afirmando que hipótesis y premisa se contradicen, la respuesta es
`CONTRADICTION` y la explicación del evaluador reconoce esa conclusión. Sin embargo,
el campo `verdict` contiene `ENTAILMENT`. La revisión propone `CONTRADICTION` y
`ALIGNED_SHIFT`. La interpretación como desacuerdo en el registro del 29 de
septiembre no queda respaldada por el contenido de ese caso.

## Casos donde se conserva incertidumbre

| Clave | Propuesta principal | Alternativa | Motivo |
|---|---|---|---|
| `gaussian_L8_a2.0_r0/54928c608d47f9ab/entail` | `UNCLEAR` | `ENTAILMENT` | El cierre dice que no puede estar seguro y que podría ser entailment. |
| `orthogonal_L12_a2.0_r0/c01de275f01a986c/entail` | `CONTRADICTION` | `UNCLEAR` | Larga alternancia entre neutralidad y una elección forzada; termina eligiendo contradicción con poca confianza. |
| `steer_L27_a4.0_r0/53723ee7633203dc/entail` | `UNCLEAR` | `CONTRADICTION` | Habla de falta de necesidad y de que las proposiciones podrían entrar en conflicto. |
| `gaussian_L16_a2.0_r0/c01de275f01a986c/entail` | `CONTRADICTION` | `UNCLEAR` | Caso de la muestra adicional: termina eligiendo contradicción tras mucha incertidumbre. |

Las primeras tres filas pertenecen a los 42 registros marcados. La última pertenece
a la muestra adicional de acuerdos y conserva su etiqueta original en la propuesta
principal. La confianza no es una probabilidad calibrada.

## Muestreo adicional y reproducibilidad

La selección se realiza sobre `judgments.jsonl` en su orden original:

- Los 42 casos marcados se toman completos, sin muestreo.
- Se filtran `NO_OBSERVED_SHIFT` y `ALIGNED_SHIFT` (3.719 registros), se aplica
  `random.Random(20261102).shuffle(...)` y se toman los primeros 20. Por azar,
  todos resultaron `NO_OBSERVED_SHIFT`: 20 textos distintos de 13 ejemplos.
  La lectura conserva los 20 veredictos, con confianza cualitativa alta.
- Para cubrir la otra categoría se filtran los 178 `ALIGNED_SHIFT`, se aplica
  `random.Random(20261103).shuffle(...)` y se toman los primeros 10. Corresponden
  a 5 ejemplos y 7 textos distintos. La propuesta conserva los 10 veredictos,
  con confianza alta en 9 y media en 1.

No se han revisado semánticamente los registros `BASELINE_NOT_ENTAILMENT`,
`AMBIGUOUS` o `UNASSESSABLE`. Las dos muestras pequeñas no proporcionan una cota
fiable sobre los desacuerdos no detectados, ni permiten estimar la tasa global de
errores del evaluador. Tampoco hay que presentar la proporción de revisiones en
los 42 casos seleccionados como tasa de error sobre los 4.850 juicios.

SHA-256 de las entradas utilizadas:

```text
records.jsonl   d3dabc3647bb79d58a7f3f863ea42f131199743b7ecf40119e7e75d1882ebb45
judgments.jsonl 0878f3d27e6d5ac82d386e1d091365f76c45f012745ac97376723f7f1d484b51
```

## Comprobación mecánica de entradas repetidas

Se agruparon todos los juicios con `assessment.status == "ok"` por el par exacto
de contexto extraído y razonamiento. La extracción del contexto replica la de
`rejudge.py`; se excluye la respuesta final, como hace ese evaluador.

Entre **2.451 entradas distintas**, **17 grupos de entradas idénticas** recibieron
veredictos diferentes; esos grupos contienen **110 registros**. Sus claves,
veredictos, semillas registradas, modelos y huellas del servidor están incluidos
en `identical_input_check.groups` del artefacto. Los 4.666 juicios válidos declaran
el mismo modelo y la misma huella de versión del servidor.

Esto evidencia inconsistencia entre respuestas almacenadas a un mismo contenido.
No establece que todos los detalles de cada solicitud o el estado del servidor
fueran idénticos: pueden existir reintentos u otras fuentes de variación que esta
agrupación no identifica. Tampoco indica cuál de los veredictos es correcto.
El acuerdo perfecto en un retest de 49 casos no garantiza consistencia universal
ni validez semántica del evaluador.

## Uso recomendado de estos resultados

Las cifras de precisión final y de cambios de respuesta de las intervenciones
textuales no dependen de estas etiquetas del evaluador. Pueden seguir analizándose
con sus limitaciones estadísticas habituales.

Antes de publicar tasas de incoherencia entre razonamiento y respuesta, conviene
una revisión humana ciega con una muestra que incluya casos marcados, no marcados,
ambiguos y errores del evaluador. Esta auditoría proporciona casos concretos para
esa revisión. Cualquier evaluador revisado debe versionarse y validarse por
separado, conservando los juicios originales; no se ha modificado el evaluador de
la campaña en curso ni se proponen aquí tasas corregidas para todo el estudio.
