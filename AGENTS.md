# Guía de trabajo para agentes y colaboradores

## Objetivo y referencias

- El proyecto clasifica alojamientos de Barcelona en los segmentos de precio
  anunciado `Económico`, `Medio` y `Premium`. No clasifica lujo o calidad ni
  recomienda un precio óptimo, ingresos u ocupación futura.
- Leer el `README.md`, la issue asignada y `docs/decisiones.md` antes de
  implementar. Contrastar el comportamiento real con `src/cleaning.py` y
  `src/preprocessing.py`; comunicar cualquier contradicción, no resolverla
  inventando requisitos o cambiando silenciosamente los acuerdos.
- Al retomar trabajo documental, consultar
  `docs/seguimiento_documentacion.md` y verificar el estado actual de las
  PRs mencionadas. Un recordatorio escrito no es una notificación programada.

## Alcance, dependencias y coordinación

- Una issue por entrega y una rama/PR por issue, salvo alcance distinto
  aprobado expresamente. Si una tarea no tiene issue, confirmar su alcance;
  no crear issues ni implementar entregables de otras personas sin permiso.
- Revisar dependencias, asignaciones, PRs abiertas y archivos compartidos
  antes de empezar. Acordar contratos de integración y quién modifica qué.
- Usar `dev` como base de integración actual: comprobar su estado remoto,
  actualizarla sin descartar trabajo y crear una rama dedicada desde ella.
  Dirigir las PRs a `dev`, salvo cambio acordado por el equipo.
- Preservar cambios ajenos. No revertir, sobrescribir ni borrar trabajo o
  datos para facilitar una tarea sin autorización explícita.

## Permisos y entrega en GitHub

- Solicitar autorización explícita para commit, push, abrir una PR y hacer
  merge. Una autorización puede cubrir varias operaciones nombradas; no
  repetir la solicitud mientras siga vigente para el mismo alcance.
- Implementar no autoriza por sí solo a publicar. Aprobar una PR no equivale
  a fusionarla. No hacer merges automáticos ni saltarse revisión humana.
- Solicitar revisión a una persona real del equipo distinta de la autora.
  La configuración de protección de ramas se verifica por separado; estas
  instrucciones no la activan ni garantizan que GitHub la imponga.
- Usar una línea propia `Closes #N` solo para una issue completamente
  entregada; usar `Refs #N` para trabajo parcial. No cerrar por un simple push.
- Comprobar el estado real de la issue, su pertenencia al Project y su
  tarjeta. No afirmar que una acción movió estados sin observarlo. Si la
  automatización no está instalada o falla, informar; no cambiar secretos,
  reglas, permisos ni lanzar mutaciones remotas para arreglarla sin permiso.

## Preparación de datos y evaluación

- Los notebooks de modelos reutilizan `load_train_test()` y
  `build_preprocessor()`. No duplicar lectura, limpieza, selección, split o
  creación del target. Los notebooks de preparación pueden demostrar sus
  etapas sin convertirse en rutas alternativas de entrenamiento.
- No incluir `price`, derivados prohibidos del precio ni datos personales
  excluidos entre las variables del modelo. Revisar el esquema al incorporar
  fuentes o columnas nuevas; no asumir que la selección actual las protege.
- Ajustar imputación, escalado y encoding dentro del Pipeline y de cada fold,
  nunca sobre test ni sobre todo train antes de la validación cruzada.
- Los umbrales finales del target se calculan con train y se reutilizan en
  test. Documentar las salvedades de estratificación y definición de clases
  del protocolo vigente en `docs/decisiones.md`.
- Test queda reservado para la evaluación final: no usarlo para guiar EDA,
  selección de variables, Optuna o elección de modelos. Si ya influyó en
  cambios, declarar esa limitación en lugar de presentar independencia falsa.
- Respetar las reglas comunes de evaluación acordadas. Registrar semillas,
  configuración, versión de datos/código y dependencias; diferenciar siempre
  entrenamiento, validación cruzada y test. No presentar resultados ajenos
  como reproducidos sin ejecutar y verificar el experimento correspondiente.

## Tests y documentación como parte de cada entrega

- Añadir o actualizar tests cuando cambie un comportamiento. Revisar los
  existentes para no duplicarlos ni debilitar expectativas para ocultar fallos.
- Ejecutar los tests relevantes y, cuando sea viable, la suite de Python
  desde la raíz con `python -m pytest -q`, usando el entorno virtual del
  proyecto. Explicar qué no se pudo probar y por qué.
- Documentar los acuerdos significativos de cada issue en
  `docs/decisiones.md`, no únicamente en código, chat o descripciones de PR.
  Incluir qué se decidió, motivo, validación disponible, limitaciones y
  referencias a issue/PR. Actualizar en la misma PR que implementa el cambio.
- Distinguir decisiones vigentes, propuestas pendientes y resultados
  comprobados. No convertir opiniones de una persona en consenso del equipo.
  Para acuerdos anteriores, contrastar las fuentes y marcar lo no verificado.
- Actualizar el README o una guía específica si cambian el uso, instalación,
  comandos o interpretación de resultados. Enlazar el detalle para evitar
  duplicaciones; no copiar toda la conversación ni crear registros por cada
  edición menor. Conservar brevemente el motivo si se sustituye una decisión.

## Privacidad, seguridad y recursos

- No subir datos raw, credenciales, archivos `.env` con valores reales ni
  información personal. Usar ejemplos sintéticos y no registrar clientes en
  código, chat, tests, logs o documentación.
- No incluir secretos ni conversaciones expresamente excluidas en acuerdos,
  resúmenes o handoffs. Las preferencias personales ajenas al proyecto no
  forman parte de su registro de actividad.
- No publicar modelos, datasets o archivos generados por defecto: comprobar
  antes su política de almacenamiento y autorización.
- Pedir permiso antes de enviar información, escribir en servicios remotos,
  cambiar accesos o crear recursos con coste. Comprobar dependencias y no
  interferir con credenciales o entornos compartidos de otra persona.

## Comunicación y cierre de tarea

- Explicar con lenguaje claro y contexto suficiente, sin asumir conocimientos
  técnicos ni recordar al usuario pasos ya verificados innecesariamente.
- En ejemplos y tablas de clasificación, identificar expresamente la
  categoría real según precio/umbrales y la categoría predicha por el modelo.
  No usar flechas o abreviaturas sin explicar su dirección y significado.
- Separar hechos, hipótesis, propuestas y acuerdos pendientes. Evaluar el
  valor para el anfitrión, coste, riesgos y mantenimiento sin ampliar alcance.
- Al entregar, indicar rama, archivos, comprobaciones y resultados, límites,
  acciones remotas realmente realizadas y el siguiente permiso necesario.
  Distinguir tests simulados, integración local y ejecución remota real.
