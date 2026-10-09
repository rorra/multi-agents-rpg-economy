# Economía RPG — Experimento de orquestación multiagente

Comparación entre **orquestación centralizada por Director y coordinación
descentralizada** de sistemas multiagente basados en LLMs, en una economía RPG
simulada.

## Resumen

La economía (aventureros y artesanos que comercian en un libro de órdenes
central) es la misma en todas las corridas; solo cambia la coordinación:

| Condición | Coordinación |
|---|---|
| A — Centralizada | Un Director LLM elige qué agente actúa y le asigna una tarea de una lista cerrada. |
| B — Descentralizada | Sin Director. Triggers mecánicos y un scheduler neutral deciden quién actúa; los agentes se comunican entre sí. |
| C — Descentralizada + boletín | Igual que B, más un resumen global pasivo (el mismo que ve el Director). |

Las corridas se comparan emparejadas por semilla, con igual cantidad de
activaciones por ciclo (Experimento 1) y con igual presupuesto de tokens
(Experimento 2).

## Cómo funciona la economía

- Los aventureros (guerrero, mago, explorador) juntan materias primas en
  expediciones y las venden. El equipo y las pociones suman a la probabilidad
  de éxito; el equipo se desgasta en cada viaje. Una expedición exitosa también
  trae algo de oro, más para los magos, que gastan una poción de maná en cada
  viaje. Después de cada viaje el aventurero descansa unos ticks: mientras tanto
  puede comerciar, y sus órdenes siguen en el libro aunque salga.
- Los artesanos (herrería, carpintería, alquimia) transforman materias primas
  en armas, armaduras, arcos, bastones y pociones.
- Se comercia solo en un libro central de órdenes limitadas: los precios salen
  de las transacciones. Las órdenes vencen a los pocos ticks. Una acción puede
  colocar varias órdenes a la vez (`place_orders`), hasta un tope configurado.
- Impuestos: quien vende un producto terminado paga un impuesto a la ganancia,
  una parte de lo que la venta deja por encima del costo de los insumos del
  producto a sus últimos precios transados; una venta a ese costo o por debajo
  no paga nada.
  Cada cierta cantidad de ticks, cada agente paga un impuesto a los bienes
  personales sobre su oro menos el oro congelado, menos el impuesto que ya
  adeuda, menos un mínimo no imponible (y, durante un congelamiento de liquidez,
  menos la parte congelada del oro retenido en ofertas de compra colocadas antes
  del congelamiento): por debajo del mínimo no paga nada, y quien tiene más paga
  más. Se paga del oro
  disponible; lo que no puede pagarse queda como deuda. La deuda es un
  gravamen: el oro adeudado no puede respaldar una oferta de compra ni
  gastarse, no vuelve a gravarse y se paga con oro libre al comienzo de cada
  ciclo. Los artesanos pagan además
  almacenamiento por unidad, que nunca se cobra por debajo de un piso de
  oro. La reproducción muestra lo recaudado por cada impuesto y lo que pagó
  cada agente; cada cobro de bienes personales se marca en la línea de tiempo y
  en el gráfico, y en ese tick cada condición muestra quién pagó cuánto y quién
  saldó deuda.
- En B y C los agentes pueden mandarse mensajes estructurados (REQUEST, OFFER,
  INFORM, ACK, REJECT). Un ACK acepta una propuesta y coloca la orden
  correspondiente en la misma acción; ACK y REJECT nunca se responden.
- Cada corrida tiene una fase base, una perturbación (escasez de recurso,
  aumento de demanda, caída de productividad o pérdida de liquidez) y una fase
  de recuperación.

Todos los valores están en `configs/base.yaml`.

## Documentos

| Documento | Rol |
|---|---|
| [`docs/design/DESIGN_v1.0.md`](docs/design/DESIGN_v1.0.md) | Diseño experimental — la fuente de verdad. |
| [`docs/design/IMPLEMENTATION.md`](docs/design/IMPLEMENTATION.md) | Cómo el diseño se traduce a código (en inglés). |
| [`docs/glossary.md`](docs/glossary.md) | Términos del diseño y sus nombres en el código. |

## Instalación

Requiere Python 3.14.

```bash
poetry install --extras analysis --extras llm
cp .env.example .env    # solo para corridas con el proveedor real
```

Por defecto todo corre sin conexión (`llm.mock: true`), incluidas las corridas A/B/C
completas y el análisis. Las excepciones son `smoke` y `derisk`, que siempre
llaman al proveedor real (ver Uso).

## Uso

Ejecutar un trío emparejado por semilla sin conexión y abrir su reproducción:

```bash
rpg-economy --locale es trio --seed 1004 --out runs/demo --good herbs
xdg-open runs/demo/replay/es/replay.html
```

La reproducción muestra las tres condiciones lado a lado, tick por tick: el precio
de cualquier bien en las tres condiciones sobre un mismo gráfico, una tabla
comparativa y, para cada condición, todas las activaciones del tick en orden: por
qué actuó el agente (tarea del Director o trigger), qué vio, la llamada al modelo,
qué decidió y por qué, qué hizo el motor y el RESULT devuelto al Director. Cada
llamada al modelo, incluidas las del Director, tiene un botón "Prompt y respuesta"
que abre el prompt de sistema, el prompt y la respuesta exactos. Haga
clic en un agente para seguir todas sus activaciones. Teclas: Espacio reproducir/pausar,
flechas avanzar, Inicio/Fin primer/último tick, S perturbación, R recuperación,
T próximo cobro del impuesto a los bienes personales.

Otros comandos:

```bash
rpg-economy config-show
rpg-economy campaign-plan --phase pilot --profile-logs runs/mock
rpg-economy campaign-run --phase pilot --out runs/mock
rpg-economy economy-check --shock-efficacy
rpg-economy economy-check --shock-efficacy --pilot-logs runs/pilot/calibration
rpg-economy report runs/final --layer configs/budget.yaml --out reports
rpg-economy dashboard runs/pilot/.../A.jsonl
rpg-economy replay runs/pilot/.../seed_1004 --out reports/replay --all-locales
```

Con `--locale es` antes del comando (`rpg-economy --locale es <comando>`), o
`SIM_LOCALE=es`, la salida sale en español; todos los
comandos, `replay`, `dashboard` y `trio` incluidos, respetan `SIM_LOCALE`. Todos
los comandos corren con el cliente simulado sin conexión salvo que se pase
`--real`, excepto `smoke` y `derisk`: esos dos existen para verificar el
proveedor real y siempre lo llaman, sin opción `--real`. Cualquier corrida con el
proveedor real, esas dos incluidas, se rechaza hasta tener una estimación de
costo, confirmar el total proyectado con `--confirm-cost` y reconocer con
`--spend-limit-set` que la cuenta del proveedor tiene un límite de gasto.

Hay ejemplos en [`reports/samples/`](reports/samples/): un informe bilingüe y una
reproducción de un trío que se abre en cualquier navegador.

## Piloto y experimento

Primero el piloto, después el experimento. El piloto verifica que todo funcione
con agentes reales y deriva el presupuesto de tokens del Experimento 2; el
experimento corre después con ese presupuesto sumado como una capa de
configuración.

```bash
# 1. Verificar sin conexión que cada perturbación mueva su objetivo
rpg-economy economy-check --shock-efficacy
# 2. Perfilar el piloto sin conexión, para proyectar su costo
rpg-economy campaign-run --phase pilot --out runs/mock
# 3. Correr las dos etapas del piloto con el proveedor real; cada una escribe en su
#    carpeta (runs/pilot/calibration, runs/pilot/budget-derivation). Sin
#    --confirm-cost el comando muestra el total proyectado y se detiene.
#    La calibración corre un horizonte comprimido (campaign.pilot_calibration_horizon,
#    con las fases escaladas en proporción); la derivación del presupuesto corre el
#    horizonte completo.
rpg-economy pilot --stage calibration --out runs/pilot \
    --real --profile-logs runs/mock --spend-limit-set --confirm-cost TOTAL
rpg-economy pilot --stage budget-derivation --out runs/pilot \
    --real --profile-logs runs/mock --spend-limit-set --confirm-cost TOTAL
# 4. Verificar que las perturbaciones muerdan con agentes LLM y derivar el presupuesto del Experimento 2
rpg-economy economy-check --shock-efficacy --pilot-logs runs/pilot/calibration
rpg-economy derive-budget --pilot-logs runs/pilot/budget-derivation \
    --out runs/pilot/budget.json --layer-out configs/budget.yaml
# 5. Correr el experimento e informar
rpg-economy campaign-run --layer configs/budget.yaml --phase final --out runs/final \
    --real --profile-logs runs/mock --spend-limit-set --confirm-cost TOTAL
rpg-economy report runs/final --layer configs/budget.yaml --out reports
# 6. Dibujar los gráficos del paper (ambos idiomas, PNG y PDF)
rpg-economy paper-figures runs/final --layer configs/budget.yaml \
    --results reports/resource_scarcity-experiment_1/en/results.json --out reports/paper
```

Notas para correrlo:

- `paper-figures` lee los registros de las corridas y el `results.json` del
  reporte; escribe cada gráfico en `reports/paper/en/` y `reports/paper/es/`.
- `economy-check --shock-efficacy` verifica el escenario primario y los
  secundarios de `campaign.secondary_scenarios`; `--scenario` verifica cualquiera.
- `trio --real`, `smoke` y `derisk` rechazan una semilla final salvo que se
  pase `--allow-final-seed`: correrla antes del experimento mira datos finales.
- Un trío cuenta como terminado cuando sus tres registros pasan la verificación
  de integridad (`integrity.json` en la carpeta del trío). Al reanudar se saltean
  los tríos terminados y se cotizan solo los que faltan. Una falla del proveedor
  vuelve a correr solo la condición que falló; los registros apartados quedan en
  `previous-N/` y su gasto se cuenta en `billing.json`.
- `pilot --verify-only` falla si falta o quedó sin terminar alguna corrida
  planeada.
- Al reanudar, además, los registros tienen que venir de los mismos prompts,
  el mismo formato de vista global, la misma versión de Python y el mismo código
  fuente (`code_digest` en los metadatos de la corrida); cualquier otro registro
  se aparta y su trío se vuelve a correr. `derive-budget` también rechaza
  registros escritos por otro código.
- Un presupuesto derivado de corridas con el cliente simulado sin conexión queda
  marcado como `budget.derived_from_mock`, y `campaign-run`, `pilot` y `trio`
  reales lo rechazan.
- Ctrl-C detiene un lote en dos pasos. El primer Ctrl-C no inicia nada nuevo:
  cada trío en curso termina la condición en la que está y no empieza la
  siguiente, lo ya pagado se conserva y al reanudar se corren solo las
  condiciones que faltan. Un segundo Ctrl-C aborta en el próximo macrociclo. El
  bloqueo de la carpeta se mantiene hasta que todo se detuvo, y el comando sale
  con código 130. Un tercer Ctrl-C sale de inmediato: el bloqueo se libera al
  terminar el proceso, y las corridas sin terminar conservan el gasto registrado
  y se vuelven a correr al reanudar.
- `campaign-run`, `pilot`, `trio`, `smoke` y `derisk` bloquean su carpeta de
  salida (`.campaign.lock`) mientras corren; una segunda corrida sobre la misma
  carpeta se rechaza con código de salida 10. Un bloqueo dejado en este equipo
  cuyo `flock` está libre se toma, aunque su pid registrado ahora sea de otro
  proceso (se avisa con una nota). `--break-lock` solo hace falta para un
  bloqueo dejado en otro equipo o uno que no se puede leer; uno que tiene un
  proceso en ejecución nunca se reemplaza.
- Un `campaign-run` o `pilot` real sin nada pendiente saltea la confirmación de
  costo, pero igual verifica y escribe `billing.json`. Si lo pendiente cambia
  entre la confirmación y el arranque, el comando se niega con código de salida
  5 y no corre nada.
- `billing.json` se escribe aunque el lote falle. También cuenta el gasto de las
  corridas sin terminar (`unfinished_spend`), y `billed_total_all` suma los
  registros terminados, sin terminar y apartados.
- `report --allow-mock` y `economy-check --shock-efficacy --pilot-logs ...
  --allow-mock` leen también corridas con el cliente simulado sin conexión, solo
  para probar el flujo.
- `derive-budget --layer-out` se niega a sobrescribir un archivo existente salvo
  que se pase `--force`.
- El código de salida 5 indica que el comando se negó antes de gastar nada:
  falta la confirmación de costo o el límite de gasto, se pidió una corrida real
  sin `--real`, una semilla final, un presupuesto derivado de corridas simuladas,
  un presupuesto que no escribió `derive-budget` o que se derivó con otro código,
  o lo pendiente cambió.
- El código de salida 6 indica un registro de corrida que no se puede usar:
  falta, no se puede leer o está mal formado.
- El código de salida 7 indica que una corrida, etapa o verificación no terminó
  limpiamente: una falla de integridad, un trío fallido o una verificación de
  facturación fallida.
- Agregar semillas a una campaña conserva los tríos ya terminados: el hash de
  la configuración no cubre la lista de semillas ni la sección de campaña, ni el
  presupuesto en las corridas del Experimento 1.
- Volver a correr `smoke` aparta los registros anteriores en `previous-N/`;
  volver a correr `derisk` conserva el trío de cada modelo que terminó y pasó la
  verificación, y cotiza y confirma solo el resto. `smoke` pide confirmar su
  costo típico y además muestra el peor caso, para fijar el límite de gasto.
- Un `--locale` o `SIM_LOCALE` no soportado sale con código 2.

## Tests

```bash
pytest
ruff check .
ruff format --check .
```

La integración continua corre las mismas verificaciones en Python 3.14, con los
extras `analysis` y `llm` instalados.

## Notas

- Todos los parámetros están en `configs/base.yaml`. Cada corrida registra el hash
  de la configuración con la que corrió, los hashes de los prompts, el modelo, la
  versión de Python y un resumen de los módulos del código fuente que determinan
  el comportamiento de una corrida.
- El motor es determinista dada una semilla, y cada corrida se guarda como un
  registro JSONL de solo agregado que alcanza para reconstruirla.
- Los registros y resultados guardan valores neutros; el idioma se aplica al
  mostrar.
