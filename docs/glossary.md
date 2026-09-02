# Glossary (es / en)

The design document is written in Spanish and the code uses English
identifiers. This is the mapping between the two. Logs and metrics store only
the English values; Spanish is applied when rendering, through
`locales/es.json`. `tests/test_domain.py` checks that this table covers every
enum value in `src/rpg_economy/domain.py`.

## Raw materials — *materias primas*

| Español | English (code) |
|---|---|
| hierro | `iron` |
| madera | `wood` |
| cuero | `leather` |
| hierbas | `herbs` |
| cristal arcano | `arcane_crystal` |

## Products — *productos*

| Español | English (code) |
|---|---|
| arma cuerpo a cuerpo | `melee_weapon` |
| armadura | `armor` |
| arco / equipo de exploración | `bow_gear` |
| bastón / foco mágico | `staff_focus` |
| poción de salud | `health_potion` |
| poción de maná | `mana_potion` |

## Roles — *roles*

| Español | English (code) |
|---|---|
| Aventurero | `adventurer` |
| Artesano | `artisan` |
| Director | `Director` (not an economic role; outside the economy) |

## Adventurer subclasses — *subclases de aventurero*

| Español | English (code) |
|---|---|
| Guerrero | `warrior` |
| Mago | `mage` |
| Explorador / Arquero | `ranger` |

## Artisan specializations — *especializaciones de artesano*

| Español | English (code) |
|---|---|
| herrería | `smithing` |
| carpintería / arquería | `woodworking` |
| alquimia / arcano | `alchemy` |

## TASK vocabulary — *vocabulario cerrado de tareas* (Condition A only)

| Español | English (code) |
|---|---|
| ANALIZAR_MERCADO | `ANALYZE_MARKET` |
| EVALUAR_EXPEDICION | `EVALUATE_EXPEDITION` |
| EVALUAR_PRODUCCION | `EVALUATE_PRODUCTION` |
| GESTIONAR_ABASTECIMIENTO | `MANAGE_PROCUREMENT` |
| GESTIONAR_INVENTARIO | `MANAGE_INVENTORY` |
| ATENDER_DESEQUILIBRIO | `ADDRESS_IMBALANCE` |

## Shock scenarios — *escenarios de perturbación*

| Español | English (code) | Hierarchy |
|---|---|---|
| escasez de recurso | `resource_scarcity` | primary |
| aumento de demanda | `demand_surge` | secondary |
| caída de productividad | `productivity_drop` | secondary |
| pérdida de liquidez | `liquidity_loss` | secondary |
| información ruidosa | `noisy_info` | optional; only if defined before final runs |

## Conditions and experiments — *condiciones y experimentos*

| Español | English (code) |
|---|---|
| A — centralizada por Director | `Condition.A` |
| B — descentralizada | `Condition.B` |
| C — descentralizada + boletín global | `Condition.C` |
| Experimento 1 — actividad económica igualada | `experiment_1` |
| Experimento 2 — coste total igualado | `experiment_2` |

## Message types — *tipos de mensaje*

Logs and prompts use the English codes; the Spanish replay and reports show them
translated, except TASK and RESULT. Those two are protocol names and stay in
English in both languages, as the design document writes them.

| Español | English (code) |
|---|---|
| SOLICITUD | `REQUEST` |
| OFERTA | `OFFER` |
| INFORME | `INFORM` |
| TASK | `TASK` |
| RESULT | `RESULT` |
| RECHAZO | `REJECT` |
| ACUSE | `ACK` |

## Triggers — *triggers mecánicos* (Conditions B and C only)

The Spanish term is *trigger* (plural *triggers*), as in the design document,
never *disparador* or *disparo*.

| Español | English (code) |
|---|---|
| T1 — mensaje P2P recibido | `T1_message_received` |
| T2 — orden propia ejecutada/vencida/cancelada | `T2_order_event` |
| T3 — movimiento relevante de precio | `T3_price_move` |
| T4 — cruce de umbral de estado propio | `T4_own_state_threshold` |
| T5 — inactividad durante N ticks | `T5_inactivity` |

## Agent actions — *acciones del agente*

| Español | English (code) |
|---|---|
| esperar | `wait` |
| comprar | `buy` |
| vender | `sell` |
| cancelar orden | `cancel_order` |
| producir | `produce` |
| salir de expedición | `start_expedition` |
| equipar | `equip` |
| enviar mensaje | `send_message` |
| colocar varias órdenes (compra, venta, cancelación) en una acción | `place_orders` |

## RESULT and observations — *RESULT y observaciones* (Condition A only)

| Español | English (code) |
|---|---|
| éxito | `SUCCESS` |
| parcial | `PARTIAL` |
| bloqueado | `BLOCKED` |
| fallido | `FAILED` |
| precio en alza | `price_rising` |
| precio en baja | `price_falling` |
| escaso | `scarce` |
| sobrante | `surplus` |
| sin compradores | `no_buyers` |
| sin vendedores | `no_sellers` |

## Rejection codes — *códigos de rechazo* (RESULT, Condition A)

| Español | English (code) |
|---|---|
| oro insuficiente | `insufficient_gold` |
| inventario insuficiente | `insufficient_inventory` |
| faltan insumos | `missing_inputs` |
| sin capacidad | `no_capacity` |
| no permitido para el rol | `not_allowed` |
| agente fuera (en expedición) | `agent_away` |
| descansando (en el pueblo, puede comerciar y equiparse, no puede salir aún) | `resting` |
| orden inválida | `invalid_order` |
| objetivo inválido | `invalid_target` |
| cantidad inválida | `invalid_quantity` |
| faltan campos | `missing_fields` |
| mensaje rechazado | `message_rejected` |
| no responde a ningún mensaje | `not_a_reply` |
| otro | `other` |

## Market and expeditions — *mercado y expediciones*

| Español | English (code) |
|---|---|
| compra / venta (lado de la orden) | `buy` / `sell` |
| abierta | `open` |
| parcialmente ejecutada | `partially_filled` |
| ejecutada | `filled` |
| vencida | `expired` |
| cancelada | `cancelled` |
| riesgo bajo / medio / alto | `low` / `medium` / `high` |
| éxito / fracaso de expedición | `success` / `failure` |
| prioridad alta / media / baja | `high` / `medium` / `low` |
| botín de oro de una expedición | `gold_loot` |
| impuesto a la ganancia (sobre el precio menos el costo de los insumos) | `taxes.profit_rate` |
| impuesto a los bienes personales | `taxes.wealth_rate`, `taxes.wealth_every_ticks` |
| mínimo no imponible | `taxes.wealth_exemption` |
| deuda del impuesto (impuesto a los bienes personales no pagado) | `tax_arrears` |
| impuesto pagado | `tax_paid` |
| almacenamiento pagado | `storage_paid` |
| costo de almacenamiento por unidad | `storage_cost_per_unit` |
| piso de cobros | `charge_floor` |
| descanso tras una expedición (*descansa*, no *en recuperación*: la recuperación es la fase posterior a la perturbación) | `rest_ticks` |
| descanso adicional tras un fracaso, por riesgo | `recovery_ticks_on_failure` |
| multiplicador de botín por subclase | `loot_multiplier` |
| capacidad de producción (ejecuciones de receta por macrociclo) | `production_capacity_per_cycle` |
| fracción de oro congelado (pérdida de liquidez) | `frozen_gold_fraction` |

## Token buckets — *categorías de tokens*

| Español | English (code) |
|---|---|
| agente | `agent` |
| Director | `director` |
| contenido P2P | `p2p_content` |
| boletín | `bulletin` |

## Other recurring terms

| Español | English |
|---|---|
| macrociclo | macro-cycle |
| activación | activation |
| activación desperdiciada | wasted activation |
| solicitud de activación | activation request |
| scheduler neutral | neutral scheduler |
| boletín global | global bulletin |
| vista global | global view / global summary |
| libro de órdenes | order book |
| tiempo de recuperación | recovery time |
| observación censurada | censored observation |
| horizonte activo común | common active horizon |
| tick de agotamiento | exhaustion tick |
| eficiencia de coordinación | coordination efficiency |
| blanco primario | primary target |
| corrida | run |
| semilla | seed |
| trío emparejado | seed-paired trio |
| rasgos | traits |
| campaña | campaign |
| matriz de escenarios | scenario matrix |
| escenario primario / secundario | primary / secondary scenario |
| estimación de costo | cost estimate |
| panel de observación | dashboard |
| cliente simulado sin conexión | mock client |
| registro (de eventos) | log (event log) |
| informe | report |
| reproducción | replay |
| cota superior (estimación de costo) | upper bound (cost estimate) |
| ejecución de receta, tanda | recipe run, batch |
| propuesta (REQUEST u OFFER) vencida | expired proposal |
| mensurabilidad | measurability |
| volatilidad a nivel de fase | phase-level volatility |
| no medible por tratamiento | unmeasurable by treatment |
