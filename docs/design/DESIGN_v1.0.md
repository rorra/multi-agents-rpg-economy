<!--
EXPERIMENTAL DESIGN v1.0 — SOURCE OF TRUTH
This document defines the experiment. A gap or ambiguity found while
implementing is resolved here first and only then coded: the design changes
before the code does. Code-level instructions live in IMPLEMENTATION.md (same
directory).
The document is in Spanish; code uses English identifiers —
see docs/glossary.md for the canonical es↔en mapping of all domain terms.
-->

**REGLAS Y DISEÑO EXPERIMENTAL**

**Simulación de una economía RPG con agentes LLM**

Comparación entre orquestación centralizada por Director y coordinación descentralizada entre agentes autónomos, con ablación de observabilidad global

Documento de diseño experimental — versión 1.0

# Propósito

Definir un entorno reproducible en el que la variable principal sea la estrategia de orquestación del sistema multiagente, manteniendo constantes las reglas económicas, los roles, los recursos y las perturbaciones externas. La versión 1.0 consolida la versión 0.9 y cierra los dos últimos aspectos metodológicos antes de fijar el diseño: el tratamiento explícito del tiempo de recuperación como observación censurada cuando la recuperación no ocurre dentro del horizonte analizado, y la fijación del presupuesto del Experimento 2 exclusivamente a partir de las corridas piloto del escenario primario de escasez de recurso. Esta versión constituye el diseño experimental fijado previo a la implementación y a las corridas finales.

# Abstract

Los sistemas multiagente basados en modelos de lenguaje de gran escala (LLMs) permiten construir entornos en los que múltiples agentes autónomos razonan, toman decisiones, se comunican y colaboran para alcanzar objetivos individuales o colectivos. Una decisión relevante en el diseño de estos sistemas es la estrategia utilizada para coordinar la interacción entre los agentes. Este trabajo compara experimentalmente arquitecturas de orquestación en un entorno económico simulado inspirado en los videojuegos de rol (RPG). La simulación está compuesta por agentes basados en LLMs que desempeñan distintos roles económicos —aventureros y artesanos— y participan en un mercado compartido mediante la obtención, producción, compra y venta de recursos.

Se evalúan tres condiciones bajo reglas económicas idénticas. En la condición centralizada, un agente Director con visión global del sistema coordina el flujo de ejecución: selecciona qué agentes intervienen y les asigna tareas de un vocabulario cerrado según el estado de la economía. En la condición descentralizada, no existe coordinador: los agentes actúan de manera autónoma a partir de información local y se comunican directamente entre sí. Una tercera condición de ablación —descentralizada con boletín global— publica a todos los agentes el mismo resumen agregado que recibe el Director, sin coordinador central, para separar el efecto de la topología de coordinación del efecto de la observabilidad global.

Las condiciones se evalúan en corridas emparejadas considerando métricas económicas —estabilidad de precios, disponibilidad de recursos, volumen de transacciones y resiliencia frente a perturbaciones— y métricas del sistema multiagente, incluyendo llamadas a los modelos, consumo de tokens, tráfico de mensajes y eficiencia de coordinación. El objetivo es analizar cómo la estrategia de orquestación afecta el comportamiento emergente, la eficiencia y la estabilidad de un sistema multiagente basado en LLMs, identificando ventajas y limitaciones de los enfoques centralizados y descentralizados.

Idea central: el RPG es el banco de pruebas; la variable de investigación es la arquitectura de coordinación. El Director no es un personaje del juego.

# 1. Objetivo, alcance y pregunta de investigación

La simulación no pretende construir un videojuego completo. Se utiliza una economía RPG como banco de pruebas porque obliga a múltiples agentes con intereses y capacidades diferentes a coordinarse para mantener cadenas de suministro, producción y comercio.

| **Elemento**              | **Definición**                                                                                                                                                                          |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Variable independiente    | Arquitectura de orquestación: (A) centralizada mediante Director; (B) descentralizada mediante comunicación directa; (C) descentralizada con boletín global.                            |
| Variables dependientes    | Desempeño económico, estabilidad, resiliencia, eficiencia de coordinación y coste computacional.                                                                                        |
| Constantes experimentales | Roles, perfiles, capital inicial, recetas, recursos, reglas económicas, shocks, topología inicial de contactos, modelo LLM, versión, parámetros de inferencia y semillas del simulador. |
| Controles específicos B/C | Parámetros de triggers y scheduler idénticos entre B y C.                                                                                                                               |
| Unidad de análisis        | Una corrida completa. Las corridas se emparejan por semilla entre condiciones.                                                                                                          |

Pregunta principal: ¿Cómo afecta una arquitectura de orquestación centralizada, basada en un agente Director, frente a una arquitectura descentralizada basada en agentes autónomos que se comunican directamente, al desempeño, estabilidad y eficiencia de una economía multiagente simulada con LLMs?

Pregunta secundaria: ¿qué parte de la diferencia observada se explica por la topología de coordinación y qué parte por la observabilidad global de información?

# 2. Principios de diseño

- Mismo mundo, distinta coordinación: las tres condiciones comparten la misma economía; no se crean reglas más favorables para una arquitectura.

- Separación entre decisión y mecánica: el LLM elige qué intenta hacer; el motor valida inventario, dinero, recetas, aleatoriedad, matching y efectos del entorno.

- El Director no es un Gobernador: coordina el flujo del MAS; no crea recursos, fija precios, otorga dinero ni modifica probabilidades.

- Autonomía local preservada: incluso bajo Director, el agente decide cómo ejecutar la tarea dentro de las capacidades de su rol.

- Decisiones semánticas solo en LLMs identificados: scheduler, triggers y matching aplican reglas fijas sin interpretación semántica.

- Comunicación observable: toda coordinación queda registrada como mensajes estructurados.

- Complejidad mínima suficiente: se excluyen mecánicas que no ayudan a responder la pregunta de investigación.

# 3. Arquitecturas experimentales

## 3.1 Condición A — Orquestación centralizada por Director

Existe un agente Director fuera de la economía RPG. Interpreta el estado global, selecciona agentes, asigna TASKs de un vocabulario cerrado y procesa RESULTs antes de decidir el siguiente paso.

| **Paso** | **Flujo centralizado**                                                                                                                   |
|----------|------------------------------------------------------------------------------------------------------------------------------------------|
| 1        | El motor actualiza el mundo y genera una vista global resumida.                                                                          |
| 2        | Se realiza una llamada al Director al inicio del macrociclo.                                                                            |
| 3        | El Director selecciona el siguiente agente y una TASK válida.                                                                            |
| 4        | El agente recibe estado local + TASK y decide una acción concreta.                                                                       |
| 5        | El motor valida y ejecuta la acción.                                                                                                     |
| 6        | El agente devuelve un RESULT estructurado.                                                                                               |
| 7        | El Director puede realizar una nueva inferencia tras cada RESULT para decidir la siguiente activación, hasta agotar K o cerrar el ciclo. |

Regla de coste: el Director es adaptativo y puede ser llamado después de cada RESULT. Sus llamadas no consumen K activaciones económicas en el Experimento 1, pero se registran íntegramente como overhead y sí consumen el presupuesto total de tokens del Experimento 2.

Tope de inferencias del Director: el Director dispone de como máximo K + 1 llamadas al LLM por macrociclo (una inicial y, como máximo, una adicional después de cada RESULT). Cada llamada al LLM Director, incluidos los reintentos por una salida inválida, consume una unidad de este límite. Cada inferencia válida debe producir exactamente una de dos salidas: asignar una TASK a un agente o cerrar el macrociclo. No se permiten inferencias deliberativas que no asignen ni cierren.

Política ante TASK inválida del Director: una salida que no valide contra el vocabulario cerrado admite un único reintento, siempre que queden inferencias disponibles dentro del límite K + 1. Si el reintento también es inválido, la TASK se descarta y se registra como error del Director. Si el límite K + 1 se agota durante el reintento, no se realiza una llamada adicional y el macrociclo se cierra. Los errores alimentan la métrica de errores y dependencia del coordinador.

Una llamada por activación: cada activación económica en A se resuelve con una única llamada al LLM del agente. Su salida estructurada contiene la acción y, en la misma respuesta, los campos del RESULT que aporta el agente (needs, offers, observations). Los campos status, action_taken y outcome los completa el motor a partir de lo que efectivamente validó y ejecutó: un agente no puede declarar su propio éxito. El reintento del RESULT (3.1.2) y el de la salida del agente (7) son, por lo tanto, el mismo reintento, y una activación nunca se cuenta dos veces como desperdiciada.

Sin mensajes P2P en A: los agentes económicos de A no emiten mensajes peer-to-peer. El campo recipient de su salida debe ser nulo; una acción que intenta enviar un mensaje es una salida inválida y sigue la política de reintento y WAIT. En A no existe, por construcción, consumo de tokens de comunicación P2P.

### 3.1.1 Vocabulario cerrado de tareas (TASK)

| **TASK**                 | **Parámetros**         | **Rol destino**     | **Propósito**                                             |
|--------------------------|------------------------|---------------------|-----------------------------------------------------------|
| ANALIZAR_MERCADO         | bien                   | Todos               | Evaluar oferta, demanda y precios recientes, y actuar en consecuencia. |
| EVALUAR_EXPEDICION       | recurso, zona opcional | Aventurero          | Decidir si realizar expedición, zona y riesgo.            |
| EVALUAR_PRODUCCION       | producto               | Artesano            | Decidir si producir, cantidad y coste.                    |
| GESTIONAR_ABASTECIMIENTO | insumo                 | Artesano            | Conseguir insumo mediante mercado o coordinación mediada. |
| GESTIONAR_INVENTARIO     | bien                   | Todos               | Vender, conservar o re-preciar inventario.                |
| ATENDER_DESEQUILIBRIO    | bien/recurso           | Todos               | Responder desde el rol a escasez o exceso.                |

Cada TASK admite prioridad (alta/media/baja) y contexto. El contexto solo puede referenciar RESULTs recientes mediante identificador y resumen estructurado; no permite instrucciones libres que microgestionen al agente. Una referencia a un RESULT fuera de la ventana reciente se descarta y se registra, y la TASK se asigna igual. El Director ve los RESULTs completos; el agente que recibe la TASK ve de cada RESULT referenciado solo sus datos públicos (identificador, agente, acción y estado), nunca códigos de rechazo ni otro estado privado. Un parámetro opcional de la TASK que llega nulo o vacío se ignora; no invalida la TASK. El tamaño de la ventana reciente es configuración (director_recent_results); con 0 el Director no ve ningún RESULT anterior.

La coordinación mediada de GESTIONAR_ABASTECIMIENTO es mediada por el Director: el agente compra el insumo en el mercado o reporta la necesidad en RESULT.needs, y es el Director quien decide si asigna a otro agente su provisión.

### 3.1.2 Formato estructurado de RESULT

Todo agente activado mediante una TASK devuelve un RESULT validable. El RESULT comunica al Director qué ocurrió y qué necesidades u oportunidades surgieron sin exponer la cadena de razonamiento privada ni permitir texto libre que funcione como una instrucción encubierta.

| **Campo**    | **Valores / estructura**                          | **Propósito**                                                      |
|--------------|---------------------------------------------------|--------------------------------------------------------------------|
| status       | SUCCESS / PARTIAL / BLOCKED / FAILED              | Estado final de la tarea.                                          |
| action_taken | acción estructurada o WAIT                        | Acción concreta intentada o ejecutada por el agente.               |
| outcome      | resultado estructurado                            | Efecto observable de la acción validada por el motor.              |
| needs        | lista opcional: bien/recurso, cantidad, prioridad | Necesidades detectadas que pueden motivar nuevas TASKs.            |
| offers       | lista opcional: bien, cantidad, precio            | Bienes o capacidades que el agente puede ofrecer.                  |
| observations | lista breve de señales estructuradas              | Observaciones relevantes del estado local o del mercado permitido. |

Si el motor rechaza la acción, outcome lleva un código de rechazo de un vocabulario cerrado (oro insuficiente, inventario insuficiente, sin capacidad, etc.), nunca el texto del motor: ese texto puede mencionar cantidades de otros agentes, y en A llegaría al Director.

Regla de trazabilidad: el Director solo puede utilizar los campos estructurados del RESULT como contexto de decisiones posteriores. El short_reason de la salida del agente se conserva en el registro para análisis cualitativo, pero se elimina de action_taken y de todo contenido que llegue al prompt del Director. Los campos vacíos son válidos; cualquier RESULT que no valide admite un único reintento del agente y, si vuelve a fallar, se registra la activación como desperdiciada y se devuelve status=FAILED.

### 3.1.3 Qué puede y qué no puede decidir el Director

Puede decidir:

- qué agente activar y en qué orden

- qué TASK asignar y con qué parámetros

- qué RESULTs referenciar como contexto

- cuándo una necesidad está suficientemente resuelta

No puede decidir:

- alterar oro, inventarios, recursos, precios o probabilidades

- forzar transacciones o producción

- declarar éxito de expediciones

- saltarse restricciones del motor o del rol

- acceder a cadenas de razonamiento privadas

## 3.2 Condición B — Coordinación descentralizada

No existe Director. Cada agente mantiene su ciclo de observación y decisión, detecta necesidades a partir de información local y puede comunicarse directamente con otros agentes. La coordinación emerge de mensajes peer-to-peer, señales del mercado y respuestas de otros participantes.

| **Paso** | **Flujo descentralizado**                                                   |
|----------|-----------------------------------------------------------------------------|
| 1        | El motor actualiza el mundo y publica observaciones locales y de mercado.   |
| 2        | Triggers mecánicos determinan qué agentes solicitan activación.             |
| 3        | El scheduler neutral procesa hasta K activaciones.                          |
| 4        | El agente activado decide actuar, esperar o enviar un mensaje estructurado. |
| 5        | El receptor incorpora el mensaje; su recepción puede disparar T1.           |
| 6        | El motor valida y ejecuta acciones económicas.                              |
| 7        | El ciclo termina al agotar K o cuando no quedan solicitudes pendientes.     |

## 3.3 Condición C — Descentralizada con boletín global

Es idéntica a B, salvo que al inicio de la fase de decisión de cada macrociclo el motor publica a todos los agentes el mismo resumen global agregado que recibe el Director en A. Contiene precios recientes, volúmenes, señales agregadas de escasez/exceso y shocks ya observables.

Regla de incorporación pasiva: el boletín se incorpora al contexto observable y NO dispara el trigger T1 ni ningún otro trigger por sí mismo. Solo será procesado cuando el agente sea activado por un trigger independiente. La generación del boletín por el motor es determinista y no cuenta como llamada LLM. Sus tokens se contabilizan únicamente cuando el contenido del boletín es incluido efectivamente en el prompt de un agente activado. No consume una activación K ni cuenta como mensaje peer-to-peer.

Motivación: separar la ventaja de coordinación de la ventaja de observabilidad global. Si C se acerca a A, la información global explica buena parte de la diferencia; si C se acerca a B, la coordinación central aporta valor adicional.

## 3.4 Diferencia experimental exacta

| **Aspecto**                 | **A — Centralizada**  | **B — Descentralizada**      | **C — Desc. + boletín**      |
|-----------------------------|-----------------------|------------------------------|------------------------------|
| Coordinador semántico       | Director LLM          | No existe                    | No existe                    |
| Selección de quién actúa    | Director              | Triggers + scheduler neutral | Triggers + scheduler neutral |
| Asignación de tareas        | Director              | Objetivo propio              | Objetivo propio              |
| Comunicación estratégica    | Agente \<-\> Director | Peer-to-peer                 | Peer-to-peer                 |
| Información global agregada | Director              | No                           | Todos reciben boletín pasivo |
| Economía y capacidades      | Iguales               | Iguales                      | Iguales                      |

# 4. Agentes económicos y roles

Los únicos roles del mundo RPG son Aventurero y Artesano. No hay intermediarios: el libro central de órdenes ya junta a compradores y vendedores, y un rol que solo compra y revende no tiene en esta economía una función que el libro no cumpla. Los mismos agentes, rasgos y dotaciones se reutilizan en las corridas emparejadas.

## 4.1 Aventurero

| **Subclase**       | **Función económica**                           | **Ventaja**                          | **Demanda característica**         |
|--------------------|-------------------------------------------------|--------------------------------------|------------------------------------|
| Guerrero           | Minerales, cuero pesado y zonas peligrosas      | Supervivencia y carga                | Armas, armaduras, salud            |
| Mago               | Cristales y recursos arcanos                    | Eficiencia arcana; consumibles caros | Focos, maná, componentes           |
| Explorador/Arquero | Madera, hierbas, cuero liviano y recursos raros | Exploración/retirada; bajo coste     | Arcos, equipo liviano, consumibles |

- Decisión del LLM: salir o permanecer en ciudad

- Decisión del LLM: elegir zona, recurso y riesgo

- Decisión del LLM: comprar/reemplazar equipamiento

- Decisión del LLM: vender, conservar o responder solicitudes

El aventurero no fabrica: la materia prima solo le sirve para venderla, y el oro que obtiene es lo que le compra equipo y pociones. Puede retener un recurso si su precio está muy bajo, esperando que suba.

- Decisión del LLM: adaptar conducta según precios, pérdidas y mensajes

## 4.2 Artesano

Transforma materias primas en productos. Especializaciones internas: herrería, carpintería/arquería y alquimia/arcano. Los artesanos pagan por macrociclo un costo de almacenamiento por unidad de bienes que tienen, publicada o no en el libro, de modo que acumular stock que nadie compra tiene un costo. El almacenamiento no deja a un artesano por debajo de un piso de oro, suficiente para su próxima compra: al llegar al piso se deja de cobrar. El piso se compara con el oro sin contar el congelado, el impuesto adeudado ni el oro que una oferta de compra anterior al congelamiento dejaría congelado al liberarse: ese oro no puede gastarse, así que no cuenta para sostener al artesano por encima del piso. Los sumideros equilibran el botín, pero no pueden sacar a un productor de la economía.

- Decisión del LLM: qué fabricar y cantidad

- Decisión del LLM: qué insumos comprar y precio máximo

- Decisión del LLM: qué solicitudes aceptar

- Decisión del LLM: qué inventario reservar

- Decisión del LLM: qué productos vender y precio

# 5. Reglas compartidas de la economía RPG

## 5.1 Recursos y productos

| **Tipo**        | **Elementos iniciales**                                                                                          |
|-----------------|------------------------------------------------------------------------------------------------------------------|
| Materias primas | Hierro, madera, cuero, hierbas, cristal arcano.                                                                  |
| Productos       | Arma cuerpo a cuerpo, armadura, arco/equipo de exploración, bastón/foco mágico, poción de salud, poción de maná. |

- Arma cuerpo a cuerpo = hierro + cuero

- Armadura = hierro + cuero

- Arco/equipo de exploración = madera + cuero

- Bastón/foco mágico = madera + cristal arcano

- Poción de salud = hierbas

- Poción de maná = hierbas + cristal arcano

Especializaciones: herrería fabrica arma y armadura; carpintería/arquería, arco y bastón; alquimia, las dos pociones.

Cada especialización tiene una capacidad de producción por macrociclo, que cuenta ejecuciones de receta (tandas), no unidades producidas: una receta que rinde dos unidades consume una sola ejecución. Puede ser fraccionaria: por debajo de una unidad se acumula de ciclo en ciclo (una unidad cada pocos ciclos). Así la capacidad puede fijarse cerca de lo que el artesano realmente fabrica, y la caída de productividad la vuelve una restricción efectiva. El crédito de capacidad conserva la fracción que sobra tras producir, de modo que una tasa de 0,3 da 3 tandas cada 10 ciclos.

Las cantidades exactas se calibran en el piloto y se fijan en la configuración antes de las corridas finales. La configuración se valida al cargarla: cada receta tiene al menos un insumo y cada insumo es al menos 1, la durabilidad del equipo es al menos 1, el blanco del aumento de demanda tiene que ser un consumible que alguna subclase lleve en su equipo, el recurso de la escasez tiene que aparecer en alguna zona, la especialización de la caída de productividad tiene que tener artesanos, cada tabla de expediciones por nivel de riesgo tiene una entrada para cada nivel que usa alguna zona, y las listas de interés tienen que poder interpretarse.

Los precios iniciales de referencia de cada producto salen del costo de sus insumos (a los precios iniciales de las materias primas) más un margen parejo de alrededor del 35%, repartido entre las unidades que rinde cada receta. Así ningún oficio arranca con un margen desproporcionado; el impuesto a la venta por producto corrige las diferencias que queden.

## 5.2 Mercado y observabilidad

Se utiliza un mercado central con libro de órdenes. El mercado ejecuta operaciones válidas, pero no coordina intenciones.

- órdenes limitadas con bien, cantidad y precio

- prioridad precio-tiempo

- sin ventas sin inventario ni compras sin capital

- órdenes con vencimiento corto (calibrado en el piloto): una orden que no se ejecuta en pocos ticks vence, el agente vuelve a activarse (T2) y puede re-publicarla a otro precio

- precios emergentes de transacciones

- cada transacción se ejecuta al precio límite de la orden que ya estaba en el libro; si el comprador reservó más oro del que gasta, la diferencia se libera

- sin self-trading: las órdenes de un mismo agente nunca se cruzan entre sí; una orden que cruzaría una orden propia vigente del otro lado se rechaza, para que el libro nunca quede cruzado y nada se cancele sin que el agente lo decida: el agente cancela primero su orden, si quiere en el mismo place_orders

- registro completo de cada orden y transacción

Definición corregida de mercado observable para agentes económicos: cada agente puede consultar sus propias órdenes y transacciones, best bid/best ask y las últimas N transacciones de los bienes incluidos en su lista de interés fija por rol/subclase. No observa oro, inventarios, planes de producción ni estados internos de otros agentes; tampoco recibe métricas agregadas de escasez, producción total o demanda insatisfecha en B.

La vista global de A/C puede incluir agregados derivados del mismo estado canónico: precio medio/variación, volumen por bien, oferta disponible, demanda insatisfecha y shocks públicamente observables. El formato y cálculo de estos agregados se fija antes de las corridas finales.

Momento de la vista global: se construye una vez por macrociclo, al inicio de la fase de decisión, con el mismo generador y en el mismo momento que el boletín de C. El Director reutiliza esa misma vista en todas sus llamadas del ciclo; no se recalcula entre una llamada y la siguiente.

Línea del shock en la vista global (Director y boletín): informa solo el tipo de shock y el bien o la especialización afectada. Nunca incluye agregados del oro privado de los agentes ni los multiplicadores exactos del shock.

## 5.3 Expediciones y desgaste

- riesgo bajo, medio y alto

- zonas con dificultad y distribución de recursos

- éxito/recompensa según zona, subclase, equipo y aleatoriedad reproducible

- mayor riesgo implica mayor recompensa esperada y varianza

- fracaso produce recuperación/pérdida parcial, no muerte permanente

- tras cada expedición el aventurero descansa un número fijo de ticks antes de poder salir de nuevo; tras un fracaso se suma la recuperación que corresponde al riesgo de la zona. Mientras está afuera puede comerciar: poner órdenes y mantener las que ya tiene en el libro, que siguen vigentes mientras está afuera (salir no exige que se completen); lo que no puede es equiparse hasta volver al pueblo. De vuelta en el pueblo, mientras descansa, puede comerciar y equiparse, pero no puede volver a salir hasta terminar el descanso

- las expediciones que vuelven en el mismo tick sacan de los recursos de la zona en un orden derivado de la semilla, el tick y el agente, igual en A, B y C

- equipamiento con durabilidad para sostener demanda recurrente

- una expedición exitosa trae, además de materiales, un botín de oro según el riesgo de la zona, multiplicado por un factor de la subclase: el mago, que gasta una poción de maná en cada viaje, trae más oro por viaje. El botín, los impuestos y el almacenamiento son los únicos puntos donde se crea o destruye oro; se calibran juntos para que el oro por rol se mantenga estable en la fase base
- impuestos: (a) impuesto a la venta: quien vende un producto terminado paga un porcentaje del valor de la venta, con una tasa por producto (más alta donde el oficio deja más margen, como las pociones; más baja donde deja poco, como arcos y bastones); las materias primas no pagan. (b) impuesto a los bienes personales: cada cierta cantidad de ticks, todo agente paga un porcentaje de una base igual a su oro menos el oro congelado, menos lo que ya adeuda, menos un mínimo no imponible; quien no supera el mínimo no paga, y quien tiene más paga más. Mientras rige un congelamiento de liquidez, la parte congelada de las reservas de ofertas de compra anteriores al shock también queda fuera de la base. Fuera de eso, el oro reservado en órdenes de compra forma parte de la base, así que nadie lo evita dejando oro en ofertas. El impuesto a la venta se paga del producto de la venta. El de bienes personales se paga del oro disponible; lo que no puede pagarse queda como deuda, que funciona como un gravamen: el oro adeudado no está disponible (no puede reservarse en ofertas de compra ni gastarse), no vuelve a gravarse, se paga con oro libre al comienzo de cada ciclo, antes de cualquier nueva liquidación, y nunca se condona. El agente ve en su estado cuánto adeuda. Al final de la corrida, la deuda pendiente se resta de la riqueza del agente (Gini, quiebra). Tasas, período y mínimo son configuración

- cada zona acumula sus recursos en un stock con tope; las expediciones extraen de ese stock, de modo que una menor tasa de aparición reduce efectivamente la oferta

- las pociones no se consumen con el paso del tiempo: se gastan al salir de expedición (una de cada tipo que la subclase lleva), y su demanda viene de las expediciones

- cada subclase tiene su equipo de expedición, y cada objeto suma su propio bonus a la probabilidad de éxito (valores de configuración, calibrados en el piloto): Guerrero: arma cuerpo a cuerpo, armadura y poción de salud; Explorador/Arquero: arco/equipo de exploración, armadura y poción de salud; Mago: bastón/foco mágico, armadura, poción de salud y poción de maná, siendo la poción de maná la que más suma al Mago. La armadura sirve a todas las subclases. Armas y armadura se equipan y suman mientras les quede durabilidad; se desgastan solo en las expediciones (cada viaje consume una cantidad fija de durabilidad), nunca en el pueblo. Las pociones suman si el aventurero las lleva al salir y se gastan en el viaje. El bonus de cada objeto se fija al salir, con lo que el aventurero lleva en ese momento. Un objeto que no es del equipo de su subclase se desgasta igual pero no suma

# 6. Comunicación entre agentes

No se utiliza chat libre ilimitado. Los mensajes son estructurados y medibles.

| **Tipo**     | **Uso**                                                                                        |
|--------------|------------------------------------------------------------------------------------------------|
| REQUEST      | Solicitar recurso, producto o colaboración.                                                    |
| OFFER        | Ofrecer bien/cantidad/precio.                                                                  |
| INFORM       | Compartir señal de mercado o información.                                                      |
| TASK         | Solo A: instrucción estructurada del Director.                                                 |
| RESULT       | Respuesta estructurada tras TASK: status, action_taken, outcome, needs, offers y observations. |
| REJECT / ACK | Rechazo o confirmación.                                                                        |

Un mensaje no mueve oro ni bienes: la compraventa solo ocurre cuando dos órdenes se cruzan en el libro. ACK y REJECT responden a un REQUEST u OFFER recibido. ACK lo acepta y, en la misma acción, pone la orden de quien acepta: una compra si acepta un OFFER, una venta si acepta un REQUEST, con la cantidad y el precio que indica; si esa orden no puede ponerse, el ACK no se envía. Quien propuso pone entonces su propia orden, que se cruza con la anterior. REJECT declina, y quien propuso busca otra contraparte u otro precio. Es la única acción que combina un mensaje con una orden: aceptar sin comprometer la orden dejaba el trato a una activación posterior que a menudo hacía otra cosa. Entre un mismo emisor y un mismo destinatario hay a lo sumo una propuesta abierta por bien: un REQUEST u OFFER nuevo reemplaza al anterior, y un ACK o REJECT responde al último. Cada uno responde una sola vez a un mensaje de ese agente sobre ese bien, y nunca se responden: el motor rechaza un ACK o REJECT que no responda a una propuesta abierta. Una propuesta (REQUEST u OFFER) vence a la misma cantidad de ticks que una orden de mercado; el motor rechaza un ACK o REJECT a una propuesta vencida porque no responde a ninguna propuesta abierta: es una activación desperdiciada, sin reintento. Así un intercambio no puede convertirse en un bucle de confirmaciones que consuma activaciones.

Propuestas abiertas: en B y C, el agente ve en su observación las propuestas REQUEST y OFFER que le fueron dirigidas y siguen abiertas (emisor, tipo, bien, cantidad, precio y ticks que le quedan), hasta que vencen o las responde. Sus tokens se cuentan como comunicación P2P (p2p_content). En A no se muestran, porque no hay P2P.

Todo mensaje P2P lleva un contenido estructurado: el bien al que se refiere, y en REQUEST y OFFER también cantidad y precio. Un mensaje al que le falte alguno de esos datos es una salida inválida y sigue la política de reintento. TASK y RESULT no son mensajes entre agentes.

En A, la coordinación estratégica se canaliza por el Director; no hay P2P estratégico. En B/C, la coordinación es P2P. El boletín de C es información del motor, no un mensaje de agente y no dispara activación.

## 6.1 Topología de contactos

- directorio público completo de IDs, roles y especializaciones

- sin acceso a oro, inventario ni planes de otros

- sin descubrimiento dinámico en la primera versión

- cada mensaje P2P tiene destinatario único y coste de interacción/tokens

# 7. Estado, memoria y formato de decisión

| **Actor**        | **Información disponible**                                                                                                             |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Agente económico | Rol, rasgos, oro, inventario, equipo, historial propio, mercado observable limitado, directorio y mensajes. En C añade boletín pasivo. |
| Director         | Resumen global, agentes disponibles, TASK/RESULT recientes y estado agregado del mercado.                                              |
| Motor            | Estado verdadero completo para validación, sin coordinación semántica.                                                                 |

Rasgos: tolerancia al riesgo, orientación temporal, agresividad comercial y capacidad de adaptación; permanecen fijos en corridas emparejadas.

Salida estructurada: action, target, quantity, price, recipient, priority, short_reason, message_type y orders. message_type solo se usa con send_message (REQUEST, OFFER, INFORM, REJECT o ACK), y en ese caso target es el bien del mensaje. Al interpretar la salida, los valores de action y target se normalizan (los espacios y guiones pasan a `_`), igual en el Director y en los agentes, y un message_type vacío, "none" o "null" se lee como nulo.

Varias órdenes por activación: la acción place_orders coloca en una sola activación varias órdenes de mercado (compra, venta o cancelación), hasta un tope fijo por configuración, idéntico en todas las condiciones. Así un aventurero que vuelve con varios bienes los pone todos a la venta y un artesano ofrece lo que fabricó y pide sus insumos a la vez. Solo se agrupan órdenes de mercado: producir, salir de expedición, equipar y enviar mensajes siguen siendo una acción por activación. El motor valida y ejecuta cada orden por separado: las que acepta quedan en el libro y las rechazadas se informan con su código, sin anular las demás. La activación cuenta como desperdiciada solo si no se aceptó ninguna. En A, el RESULT informa el resultado de cada orden, y una tanda aceptada solo en parte tiene estado PARTIAL. Una respuesta inválida admite un único reintento; si persiste, se registra como activación desperdiciada y ejecuta WAIT. El reintento consume tokens, pero no una activación K adicional: K cuenta activaciones económicas, no llamadas al LLM.

Contenido del prompt: el prompt describe las reglas del mundo que afectan al agente, incluida la regla de observabilidad de 5.2 (los demás agentes solo conocen sus necesidades si él se las comunica), y documenta los parámetros de cada acción válida. No incluye consejos tácticos que orienten al agente hacia una forma de coordinación: esa conducta es lo que el experimento mide. Además, el prompt incluye una guía de precios (el precio de referencia de cada bien, aclarando que no es un precio oficial), el impuesto que el agente adeuda, sus propias transacciones recientes y, en B y C, las propuestas abiertas dirigidas a él. Antes del estado, unas líneas destacadas resumen hechos que el agente ya puede ver: si un aventurero puede salir de expedición y qué equipo le conviene, qué productos del artesano tienen comprador o quedaron sin vender, qué insumos le faltan y a qué precio se consiguen. Las señales de mercado indican, para cada bien que el agente sigue, si escasea o sobra y cuáles de sus órdenes vencieron sin llenarse, y sugieren pasos de precio de alrededor de un 10%. Todo se arma solo con información que el agente ya ve, y es igual en A, B y C.

# 8. Tiempo, activación y presupuesto de cómputo

## 8.1 Presupuesto de activaciones

| **Concepto** | **Regla**                                                                                                                                                                  |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Macrociclo   | Unidad temporal comparable.                                                                                                                                                |
| K            | Máximo de activaciones de agentes económicos por macrociclo, idéntico en A/B/C. Se calibra en el piloto para que sea una restricción efectiva también en B/C.          |
| A            | El Director asigna activaciones; sus llamadas son overhead adicional en el Experimento 1, con un límite total de K + 1 llamadas LLM por macrociclo, incluidos reintentos. |
| B/C          | Triggers mecánicos generan solicitudes; scheduler neutral atiende hasta K.                                                                                                 |
| Fin temprano | Puede cerrar si no quedan solicitudes/tareas; se reportan activaciones efectivas.                                                                                          |
| Coste        | Llamadas, tokens, latencia y activaciones desperdiciadas por arquitectura.                                                                                                 |

Estructura del macrociclo: un macrociclo abarca un número fijo de ticks mecánicos (ticks_per_cycle, por defecto 1, con lo que tick y macrociclo coinciden y todos los umbrales expresados en ticks conservan su sentido directo). Cada macrociclo tiene tres fases:

1.  Fase mecánica: en cada tick aparecen recursos, avanzan las expediciones (que desgastan el equipo que llevan), vencen órdenes y se aplican los shocks. Los triggers se evalúan después de cada tick y las solicitudes se acumulan.

2.  Fase de decisión: en A, el loop del Director; en B/C, el scheduler atiende hasta K solicitudes acumuladas. Las activaciones se ejecutan en secuencia y cada una ve el efecto de la anterior; esta regla es idéntica en las tres condiciones.

3.  Cierre: se reportan las activaciones efectivas; las solicitudes no atendidas persisten mientras su trigger siga vigente.

ticks_per_cycle es idéntico en todas las condiciones y en ambos experimentos.

## 8.2 Triggers mecánicos de activación — solo B/C

- T1 — Mensaje P2P recibido desde la última activación. El boletín global de C está explícitamente excluido.

- T2 — Orden propia ejecutada, vencida o cancelada.

- T3 — Movimiento relevante de precio en un bien de interés, por encima de umbral fijo.

- T4 — Cruce de umbral de estado propio: insumos, capacidad, liquidez, durabilidad, regreso de una expedición o fin del descanso.

- T5 — Inactividad durante N ticks.

Los umbrales T3/T4 y N se calibran en piloto y se fijan en la configuración; son idénticos en B y C. Los umbrales de oro de T4 leen el oro disponible: sin el oro reservado en órdenes de compra, sin el congelado y sin el adeudado como impuesto. El prompt aclara que el oro disponible excluye el reservado, el congelado y el impuesto adeudado.

## 8.3 Scheduler neutral

- procesa solicitudes hasta agotar K

- si hay exceso, prioriza menor número de activaciones acumuladas

- desempata con permutación fija de IDs derivada de la semilla

- no inspecciona contenido de estados ni mensajes

- solicitudes no atendidas persisten mientras el trigger siga vigente

## 8.4 Experimentos de presupuesto

Experimento 1 — actividad económica igualada: mismo K de activaciones económicas por macrociclo. Las llamadas del Director se registran como overhead, sin restar K, respetando el límite K + 1 de la sección 3.1.

Experimento 2 — coste total igualado: todas las condiciones reciben el mismo presupuesto máximo de tokens LLM por corrida. En A, los tokens consumidos por el Director y por los agentes económicos se descuentan del mismo presupuesto. En B, se descuentan los tokens de los agentes y de su comunicación P2P incluida en los prompts. En C, además se descuentan los tokens del boletín cada vez que se incorpora al prompt de una activación. El número de llamadas LLM se registra como métrica secundaria, pero no se utiliza como unidad del presupuesto.

Emparejamiento entre experimentos: el Experimento 2 reutiliza exactamente las mismas semillas finales que el Experimento 1, formando tríos A/B/C emparejados por semilla en ambos regímenes de presupuesto. El presupuesto total de tokens del Experimento 2 se determina durante el piloto mediante una regla neutral aplicada exclusivamente al escenario primario de escasez de recurso: se calcula el consumo total de tokens de las corridas piloto A, B y C de ese escenario bajo el régimen del Experimento 1 y se fija como presupuesto común la mediana agrupada de esos consumos, sin privilegiar a ninguna condición. La mediana agrupada es la mediana del conjunto de totales por corrida de todas esas corridas A, B y C juntas, sin ponderar por condición (no una mediana de medianas). Se calcula por script y no se ajusta a mano. El valor se escribe en la configuración antes de iniciar las corridas finales.

Unidad contable: el presupuesto se descuenta del total de tokens que informa el proveedor en cada llamada. La separación por categorías (agentes, Director, comunicación P2P y boletín) es una atribución analítica que explica en qué se gastó; nunca reemplaza al total informado.

Vigencia de K en ambos experimentos: la estructura de macrociclos y el tope K de activaciones económicas por macrociclo rigen de forma idéntica en los Experimentos 1 y 2. En el Experimento 2, el presupuesto de tokens es una restricción adicional a nivel de corrida, no un reemplazo de K. Así, la dinámica por ciclo permanece comparable.

Agotamiento del presupuesto: si el presupuesto de tokens se agota antes de la duración planificada, cesan todas las llamadas LLM —activaciones de agentes y, en A, inferencias del Director—, pero el motor continúa ejecutando ticks mecánicos hasta completar el horizonte planificado. Las órdenes vigentes pueden ejecutarse o vencer, los shocks programados ocurren y las reglas mecánicas continúan. El tick de agotamiento se registra: agotar antes el presupuesto es en sí mismo un resultado del Experimento 2. Como el coste de una llamada solo se conoce después de realizarla, la activación en curso cuando se cruza el límite se completa: su acción se ejecuta y se registra normalmente, y todas las activaciones posteriores se rechazan. Al ser secuenciales, hay como máximo un exceso de este tipo por corrida.

Horizonte activo común — análisis secundario: además de las métricas primarias calculadas sobre la duración completa, para cada semilla se define un horizonte activo común del trío A/B/C que termina en el primer tick de agotamiento de presupuesto observado entre las tres condiciones; si ninguna se agota, coincide con el final planificado. Todas las métricas económicas se recalculan sobre este intervalo común como análisis secundario, también las secundarias (producción útil, Gini, quiebres de inventario, agregados sobre todos los bienes y propagación), de modo que A, B y C se comparen mientras las tres arquitecturas todavía disponen de capacidad de decisión LLM. La regla de censura del tiempo de recuperación definida en 11.1 se aplica igualmente sobre este horizonte: una recuperación no observada dentro del horizonte activo común es una observación censurada, no un dato faltante, aun cuando el horizonte termine antes de la ventana de referencia.

# 9. Ejemplos de coordinación

| **A — Centralizado: escasez de hierro**                          | **B/C — Descentralizado**                                               |
|------------------------------------------------------------------|-------------------------------------------------------------------------|
| Director detecta señal global y asigna ANALIZAR_MERCADO(hierro). | Herrero se activa por trigger de precio/orden.                          |
| Con RESULT, asigna EVALUAR_EXPEDICION(hierro).                   | Herrero envía REQUEST a aventureros.                                    |
| Puede encadenar GESTIONAR_ABASTECIMIENTO o EVALUAR_PRODUCCION.   | Aventureros/artesanos responden según objetivos propios.                |
| Secuencia mediada por Director.                                  | Secuencia emerge de señales y P2P; C dispone además del boletín pasivo. |

# 10. Perturbaciones controladas

Escenarios independientes: cada tipo de perturbación constituye un escenario experimental independiente. Una corrida final contiene una fase base, una única perturbación principal y una fase de recuperación. Para una misma semilla y escenario, A, B y C reciben exactamente el mismo tipo de shock, magnitud y tick de aplicación; el Director no puede prevenirlos ni modificarlos. No se encadenan distintos shocks principales dentro de una misma corrida final, evitando efectos de arrastre entre perturbaciones. Los escenarios se analizan por separado (11.4).

Jerarquía pre-registrada de escenarios: para contener el coste total (semillas × condiciones × experimentos × escenarios), se fija de antemano una jerarquía. El escenario primario es la escasez de recurso: es el de cadena causal más limpia y blanco primario mejor definido, y se ejecuta con el set completo de semillas finales y ambos experimentos (1 y 2). Los escenarios secundarios son aumento de demanda, caída de productividad y pérdida de liquidez; los tres están implementados, pero por presupuesto la campaña final ejecuta solo uno, aumento de demanda, con el Experimento 1 y 8 semillas finales. Los otros dos, y la ejecución de cualquier secundario bajo el Experimento 2, quedan como extensión opcional. La condición de información ruidosa solo se incorpora como escenario adicional si se define antes de las corridas finales. Esta jerarquía se fija junto con el resto del diseño: recortar o ampliar escenarios después de observar resultados no está permitido.

Efectividad de las perturbaciones: antes de las corridas finales, cada perturbación se verifica sin costo con agentes de reglas, bajo el mismo régimen de activación del experimento (triggers, scheduler neutral y K), corriendo cada semilla piloto con y sin la perturbación. Una perturbación es efectiva si, en la mediana de las semillas y sobre la fase de shock, mueve el precio de su blanco primario al menos un 10% o le agrega al menos 10 ticks de desabastecimiento sobre la fase de shock completa, en proporción si la fase está comprimida (en la pérdida de liquidez, el valor transado al menos un 10%). Una menor oferta por sí sola no alcanza: la escasez la reduce por construcción, aunque el mercado nunca la note. Una perturbación que no cumple se recalibra. En el piloto se repite la verificación sobre los logs de las corridas LLM, comparando la fase de perturbación con los últimos W ticks de la fase base de cada corrida, escalados a la duración de la fase de perturbación. Si el escenario secundario no muestra efecto en el piloto, se reemplaza por otro de los secundarios antes de las corridas finales. Esta decisión se toma solo con la efectividad de la perturbación, nunca con las diferencias entre condiciones.

| **Shock**                      | **Implementación**                         | **Qué observa**                          |
|--------------------------------|--------------------------------------------|------------------------------------------|
| Escasez de recurso             | Reducir aparición de hierbas               | Reorganización de suministro.            |
| Aumento de demanda             | Multiplicar las pociones por expedición    | Velocidad de respuesta.                  |
| Caída de productividad         | Reducir capacidad artesanal                | Priorización.                            |
| Pérdida de liquidez            | Congelar oro disponible de todos los agentes | Propagación y recuperación.              |
| Información ruidosa — opcional | Ocultar/retrasar señales                   | Robustez ante observaciones imperfectas. |

Duración de los shocks: todo shock tiene una duración finita fijada en configuración. El fin del shock es el tick en que el parámetro perturbado vuelve a su valor base, y desde ese tick se mide el tiempo de recuperación.

Subconjunto de la pérdida de liquidez: los agentes afectados se especifican explícitamente en la configuración, como lista de IDs o como cantidad por rol, y nunca se infieren del orden de los IDs (los IDs codifican rol y subclase, por lo que tomar un prefijo ordenado elegiría roles por accidente alfabético). El subconjunto es el mismo en todas las semillas y no consume aleatoriedad. La configuración piloto afecta a todos los agentes, como una crisis de liquidez general: con un subconjunto, los afectados siguen vendiendo y gastan lo que cobran, y el comercio agregado apenas lo nota. La pérdida congela, en el tick del shock, la fracción configurada del oro disponible de cada afectado: su oro libre menos lo que adeuda del impuesto. El oro que en ese momento ya respalda ofertas de compra sigue comprometido (la oferta puede ejecutarse), pero cuando ese oro se libera porque la oferta vence o se cancela, la misma fracción de lo liberado queda congelada, también neta de la deuda que ese oro va a pagar. Si esa oferta se libera (vence o se cancela) durante el congelamiento, el agente termina con el mismo oro congelado que uno que tenía ese oro libre en el shock. Si en cambio se ejecuta, su reserva se gasta sin congelarse, porque la compra se comprometió antes del congelamiento. El vuelto de una oferta que se ejecuta por debajo de su límite no se congela. El oro reservado en ofertas puestas después del shock no se congela. Así el tamaño de la perturbación no depende de si el agente tenía una oferta abierta en el tick del shock, que difiere entre condiciones de un mismo trío. Todo el oro congelado se libera en la reversión.

# 11. Métricas de evaluación

## 11.1 Métricas económicas

| **Métrica**            | **Definición operativa**                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Volatilidad            | Desvío estándar del precio de transacción por bien en ventanas W, promediado por fase. Una ventana con menos de dos transacciones no entra en el promedio; si ninguna ventana de la fase califica, la métrica es faltante, nunca cero. Compañera pre-registrada: volatilidad de fase, el desvío estándar de todos los precios de la fase, medible con dos o más transacciones en total.                                                                                                        |
| Stockouts              | Cantidad/duración de intervalos sin oferta de un bien esencial. Definición primaria: ticks sin órdenes de venta vigentes del bien en el libro, lo que los compradores efectivamente pueden observar. Variante secundaria: inventario total del bien en el mundo igual a cero.                                                                                                                                                                                                             |
| Volumen                | Unidades y valor transados por bien y fase.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Tiempo de recuperación | Ticks desde el fin del shock hasta que la métrica objetivo vuelve a la banda ±X% de referencia durante M ticks consecutivos. La referencia es la media de la métrica objetivo en los últimos W ticks de la fase base, calculada por corrida: el precio medio de las transacciones de esos W ticks (o el valor transado en ellos), es decir, el valor de la serie suavizada en el último tick base. Si en esa ventana no hubo transacciones, la recuperación no es medible y la corrida se trata como dato faltante, no como censurada; como esa mensurabilidad depende del tratamiento, recibe la regla de los faltantes causados por el tratamiento (11.3). Si la recuperación no ocurre dentro del horizonte analizado (completo o activo común), la observación se marca como censurada y se registra únicamente el tiempo máximo observado hasta el final del horizonte. La tasa de censura se reporta por condición y escenario; una observación censurada no se interpreta como un tiempo efectivo de recuperación. |
| Producción útil        | Unidades vendidas o consumidas dentro de T ticks desde su producción. Su complemento, las unidades no colocadas dentro de T ticks, se reporta como proxy de sobreproducción.                                                                                                                                                                                                                                                                                                              |
| Gini                   | Oro + inventario valorizado al precio de referencia de cada bien: precio medio de transacción en la fase base de la corrida o, si el bien no se transó, su precio inicial configurado. Nunca precios concurrentes, para que la volatilidad no contamine la medida de distribución. Secundaria.                                                                                                                                                                                         |

Fase de medición: la volatilidad, los stockouts y el volumen primarios se miden sobre la fase de shock, que es donde la arquitectura tiene que responder; promediarlos sobre toda la corrida diluiría el efecto con fases base y de recuperación idénticas entre condiciones. El tiempo de recuperación se mide desde la reversión del shock, como se define arriba.

