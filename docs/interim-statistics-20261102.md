# Figuras y verificación estadística — 2 de noviembre de 2026

Se han preparado tres figuras de Distill-7B, en PNG y PDF, usando únicamente
los 5.229 registros del estudio completo. La generación de Distill-32B no se
interrumpe ni se utiliza como si fuera un estudio terminado. Este análisis no
requiere el evaluador LLM ni importa PyTorch o Transformers.

## Reproducción

Desde la raíz del repositorio:

```bash
env/bin/python scripts/build_interim_figures.py
```

Las salidas están en `artifacts/diagnostics/20261102/figures/`. Las leyendas de
las figuras están en inglés para su incorporación al manuscrito:

- `text_paired_contrasts.pdf` y `.png`: cambio en respuestas CONTRADICTION
  respecto al razonamiento original completo, sobre los mismos ejemplos dentro
  de cada contraste. Se muestran por separado todos los elegibles y los aciertos
  iniciales, con IC bootstrap pareados del 95 %, límites por etiquetas ausentes
  y denominadores de pares completos/elegibles. Los límites por datos ausentes
  no son intervalos de confianza.
- `truncation_accuracy.pdf` y `.png`: aciertos sobre los mismos 48 ejemplos en
  las cinco condiciones de truncación. Las etiquetas ausentes cuentan como no
  correctas; los intervalos Wilson son marginales y no prueban diferencias ni
  equivalencia entre condiciones emparejadas.
- `activation_control_contrasts.pdf` y `.png`: los 64 contrastes steering/control
  de igual norma, con diferencia entre casos completos y denominadores. No se
  agregan las celdas como si fueran observaciones independientes.
- `text_contrasts.csv`, `activation_contrasts.csv`, `truncation.csv` y
  `tables.json`: datos de las figuras, resultados completos, hashes de entradas
  y fuentes, versiones y parámetros del bootstrap.

Las tres figuras se han inspeccionado visualmente; se corrigió la posición de
los denominadores para que no oculten los intervalos del efecto de sustitución.

## Resultados comprobados desde los registros originales

| Comparación contra prefijo completo | Pares completos/elegibles | Contradicciones intervención/control en todos los elegibles | Diferencia pareada, casos completos | p exacta sin ajustar |
|---|---:|---:|---:|---:|
| Sustitución, todos los elegibles | 41/41 | 40/7 | +80,49 puntos | 2,33e-10 |
| Sustitución, aciertos iniciales | 34/34 | 33/0 | +97,06 puntos | 2,33e-10 |
| Barajado, todos los elegibles | 47/48 | 11/8 | +8,51 puntos | 0,3877 |
| Barajado, aciertos iniciales | 40/40 | 8/0 | +20,00 puntos | 0,0078125 |
| Prefijo 0 %, todos los elegibles | 46/48 | 5/8 | −4,35 puntos | 0,7266 |

Cuando hay etiquetas ausentes, la diferencia entre casos completos no se obtiene
restando las dos tasas sobre todos los elegibles. Por ejemplo, el control del
barajado tiene ocho contradicciones entre 48 elegibles y siete entre los 47 pares
completos; de ahí (11−7)/47 = 8,51 puntos. Las tablas conservan ambas cantidades
y los límites por datos ausentes para evitar mezclar denominadores.

Los cinco prefijos 0 %, 25 %, 50 %, 75 % y 100 % producen respectivamente
41, 43, 41, 40 y 40 aciertos sobre los mismos 48 ejemplos. Solo la condición
0 % tiene etiquetas ausentes: dos. La condición 0 % cierra el bloque de pensamiento
y permite hasta 128 tokens adicionales: no elimina todo cómputo o razonamiento.

En las 64 comparaciones de activaciones, la menor p exacta es 0,0625; todas las
p de Holm valen 1. La mayor diferencia positiva entre casos completos es
13,16 puntos (capa 0, intensidad 2, control gaussiano): son cinco pares
discordantes en una dirección y cero en la otra entre 38 pares completos de
40 elegibles. Con tan pocos discordantes, un IC bootstrap percentil puede excluir
cero mientras la prueba exacta bilateral da 0,0625. No debe elegirse el método
que produzca significación; se informa la prueba exacta y su ajuste múltiple.

La sustitución se mantiene como señal descriptiva fuerte, pero el texto sustituido
procede de una hipótesis distinta y puede contener la conclusión incompatible.
No identifica por sí sola mediación por el razonamiento natural. Las pequeñas
diferencias de truncación no demuestran equivalencia ni superioridad de razonar
menos. Las comparaciones textuales son exploratorias y no están ajustadas por
multiplicidad. Los resultados de subconjuntos deben acompañarse del conjunto
elegible completo.

## Qué se verificó

El constructor exige una ejecución completa, comprueba el número y unicidad de
registros, verifica el hash de registros de `robustness.json` y reconstruye los
recuentos de las 104 condiciones de entailment desde las generaciones. Recalcula
los 13 contrastes textuales y los 64 de activaciones, incluyendo denominadores,
discordancias, límites por ausencias, pruebas exactas e intervalos pareados,
y exige coincidencia con los informes existentes. Comprueba también el ajuste
Holm y que los cinco grupos de truncación usan exactamente los mismos ejemplos.

Se verifican los hashes de las cinco entradas antes y después de generar las
figuras. No se modifican generaciones, evaluaciones, resúmenes ni fuentes
científicas congeladas. El bootstrap usa semilla 42, 5.000 muestras y el ejemplo
emparejado como unidad; reutiliza las funciones de análisis del repositorio,
por lo que esta verificación de datos y coherencia no constituye una validación
estadística independiente de esas funciones.
