# Lectura provisional y trabajo en paralelo — 2 de noviembre de 2026

Esta revisión usa el estudio completo de Distill-7B y una captura parcial de
Distill-32B con 2.066 registros. No cambia el protocolo ni permite concluir todavía
la comparación Distill-32B/QwQ-32B. Los ocho archivos científicos y la configuración
congelada conservan sus hashes. La campaña sigue generando.

## Qué resulta interesante

### La sustitución del razonamiento tiene un efecto grande en 7B

Entre los 34 ejemplos con respuesta inicial correcta y ambos razonamientos
necesarios disponibles, sustituir el razonamiento por el generado para la hipótesis
contradictoria del mismo par cambia 33 respuestas a CONTRADICTION (97,1 %).
Con el razonamiento original completo, regenerar la respuesta no cambia ninguna
de esas 34 respuestas. El contraste pareado exacto tiene p = 2,33e-10, sin ajuste
por comparaciones múltiples; pertenece al análisis exploratorio de intervenciones
textuales. No necesita las etiquetas del evaluador LLM.

En los 41 ejemplos elegibles, incluidos aquellos que ya fallaban inicialmente,
la sustitución produce 40 respuestas CONTRADICTION frente a 7 con el prefijo
original. No son dos muestras independientes ni 41 aciertos iniciales.

Esto demuestra sensibilidad de la respuesta al texto insertado bajo esta
intervención. El texto insertado procede de otra hipótesis y puede incluir esa
hipótesis y su conclusión; el efecto no distingue entre usar pasos intermedios,
copiar una conclusión o seguir un contexto incompatible. No demuestra por sí
solo que el razonamiento generado naturalmente medie la decisión.

### Acortar el razonamiento no muestra una degradación clara en esta muestra

Sobre los mismos 48 ejemplos con razonamiento inicial completo:

| Prefijo conservado | Respuestas correctas | Sin etiqueta final válida |
|---|---:|---:|
| 100 % | 40/48 | 0 |
| 75 % | 40/48 | 0 |
| 50 % | 41/48 | 0 |
| 25 % | 43/48 | 0 |
| 0 % | 41/48 | 2 |

La condición 0 % cierra el bloque de pensamiento y permite regenerar hasta
128 tokens; no debe interpretarse como ausencia de todo cómputo o razonamiento.
Estas diferencias pequeñas no establecen equivalencia ni que razonar perjudique.
La exactitud cuenta las respuestas inválidas como no correctas. El contraste de
contradicciones entre 0 % y 100 % usa 46 pares con etiqueta válida y da p = 0,727.

Barajar los tokens cambia 8 de los 40 aciertos iniciales a contradicción, frente a
0 con el prefijo original (p exacta sin ajustar = 0,0078125). En toda la muestra
elegible también corrige errores iniciales: el contraste tiene 8 cambios en una
dirección y 4 en la contraria entre 47 pares completos (p = 0,388). Por tanto, su
interpretación depende del subconjunto; no presentar solo el resultado favorable.

### El steering de activaciones aún no destaca frente a los controles

En 7B hay 64 contrastes pareados entre steering y controles de igual norma.
La menor p exacta sin ajustar es 0,0625 y todas las p ajustadas por Holm son 1.
No hay evidencia estadística convincente de un efecto específico con este diseño
y tamaño muestral. Esto no demuestra ausencia de efecto. Las celdas reutilizan
los mismos 50 pares; miles de generaciones no equivalen a miles de casos
independientes.

Las baselines completas de 32B dan 48/50 aciertos en entailment y 43/50 en
contradiction, frente a 40/50 y 42/50 en 7B. La exactitud equilibrada es 91 %
frente a 82 %. Es una comparación descriptiva; no identifica un efecto del tamaño
o del entrenamiento. Los 26 contrastes de activación/control completos disponibles
en la captura parcial de 32B tienen p exacta 1, con diferencia absoluta máxima
de 2,22 puntos porcentuales entre casos completos. Faltan capas y condiciones;
esta inspección intermedia no debe cambiar el protocolo congelado.

### Hay que auditar las etiquetas del evaluador

La etiqueta `ANSWER_REASONING_MISMATCH` aparece 28 veces: 13 con steering,
10 con control ortogonal y 5 con gaussiano. Son etiquetas automáticas, no
28 incoherencias verificadas ni 28 ejemplos independientes.

En `steer_L0_a2.0_r0/2ebc92fd4bda7938/entail`, la respuesta es CONTRADICTION y
el razonamiento concluye explícitamente que las proposiciones se contradicen.
El evaluador cita esa conclusión y la describe como contradicción, pero devuelve
`verdict: ENTAILMENT`. La categoría de desacuerdo es incorrecta en ese caso.
Esto corrige la interpretación de ese ejemplo en el registro del 29 de septiembre.
Los resultados originales se conservan para auditoría. Antes de usar tasas de
incoherencia, hace falta revisión humana de los casos marcados y de una muestra
de casos no marcados. La consistencia del retest no garantiza validez.

## Cómo avanzar en paralelo

| Trabajo | Recurso | Dependencia |
|---|---|---|
| Generación Distill-32B | GPU local, proceso actual | Mantener su ejecución |
| Estadística y figuras de 7B | CPU | Datos completos; robustez calculada en esta revisión |
| Revisión de etiquetas y casos | Lectura humana o agentes con acceso a archivos | No requiere el evaluador local ni GPU |
| Métodos, limitaciones y tablas provisionales | CPU/editor | Separar resultados definitivos de los pendientes |
| Generación QwQ-32B simultánea | Otra máquina con GPU suficiente | Misma revisión, datos, configuración y fuentes; salida propia |
| Evaluador simultáneo a generación | Otro recurso de inferencia validado | Mantener modelo, prompt y procedencia del evaluador |

Con una sola máquina, podemos simultanear generación, análisis, revisión y
redacción. El generador ya usa lotes de 16. No hay capacidad
comprobada para otro modelo grande: en esta revisión había unos 35 GiB de memoria
disponible y la ejecución concurrente del evaluador ya provocó OOM anteriormente.

Con otra máquina, QwQ puede generar sin esperar a Distill. Debe usar un directorio
independiente; después se validan identidad, cobertura y resultados antes de
integrarlo. El lanzador actual es secuencial y no incorpora automáticamente una
ejecución externa. No se debe duplicar el lanzador sobre la misma campaña ni
permitir dos escritores en los mismos archivos. La configuración del segundo
equipo debe quedar registrada; una semilla igual no garantiza igualdad bit a bit
entre plataformas.

## Evidencia y reproducción

- Robustez de 7B: `runs/study-20260925/distill-7b/robustness.json`, generado con
  `env/bin/python robustness.py runs/study-20260925/distill-7b`.
- Tablas descriptivas: `runs/study-20260925/distill-7b/summary.json`.
- Captura de baselines de 32B, hash del prefijo observado y ejemplo del evaluador:
  `artifacts/diagnostics/20261102/interim-evidence.json`.
- Los registros, juicios y archivos científicos originales no se modificaron.
