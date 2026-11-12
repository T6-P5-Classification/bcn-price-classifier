# Automatización del ciclo de issues y PRs

## Estado de esta entrega

Esta PR prepara la automatización para revisión de Esra. No configura
credenciales, permisos, protección de ramas ni reglas del Project, no hace
merge y no ejecuta pruebas que cambien issues o tarjetas reales.
La instalación no se considera verificada hasta completar la activación y
observar las transiciones en GitHub.

## Destino y comportamiento

- Repositorio: `T6-P5-Classification/bcn-price-classifier`.
- Rama de integración y rama por defecto comprobadas: `dev`.
- Project de la organización: [bcn-price-classifier, número 3](https://github.com/orgs/T6-P5-Classification/projects/3).
- Campo: `Status`; opciones exactas: `In progress`, `In Review`, `Done`.
- Configuración versionada: `.github/personal-github.json`.

La automatización mueve la tarjeta de la **issue**, no una tarjeta separada
de la PR. Solo procesa referencias de finalización locales en líneas propias
del cuerpo de la PR, fuera de comentarios HTML y bloques de código:

```text
Closes #14
```

El ejemplo anterior está dentro de un bloque de código y no se ejecuta.
Para vincular una entrega real, poner la referencia sin las comillas del
bloque en la descripción de su PR. Se admiten también `Fixes` y `Resolves`.
`Refs #14` indica trabajo parcial y no activa este ciclo.

| Evento o estado actual | Resultado sobre la issue vinculada |
|---|---|
| PR abierta o reabierta, no borrador, dirigida a `dev` y sin revisión bloqueante efectiva | Pasa a `In Review`; sigue abierta. |
| PR pasa a borrador o se cierra sin merge | Vuelve a `In progress`, salvo que otra PR lista para revisión cubra la misma issue. |
| Revisión con cambios solicitados | Vuelve a `In progress`, con la misma excepción de otra PR lista. |
| Se vuelve a solicitar explícitamente revisión a cada persona que bloqueaba, aprueba o se descarta su revisión bloqueante | Vuelve a `In Review` si no quedan bloqueos y la PR no es borrador. |
| Push o comentario sin resolver el bloqueo de revisión | No borra los cambios solicitados. |
| PR fusionada en `dev` | Cierra la issue como completada si seguía abierta y mueve su tarjeta a `Done`. |
| PR dirigida a otra rama | No cambia issues ni tarjetas. |
| Inicio manual autorizado de una issue | Pasa de pendiente a `In progress`, sin reabrir ni retroceder tarjetas en revisión o terminadas. |

GitHub ya marca la PR como fusionada/cerrada al hacer merge; no necesita
otra acción para eso. Este workflow completa el cierre de la issue y su
estado en el Project. **Nunca aprueba ni fusiona PRs automáticamente**.
La exigencia de aprobación humana debe configurarse por separado mediante
las reglas de protección del repositorio; este workflow no la impone.

## Seguridad y convivencia

- El workflow privilegiado usa `pull_request_target`, `workflow_run` y
  ejecución manual. Solo carga script y configuración de la rama por
  defecto revisada; nunca descarga ni ejecuta el código de la rama de la PR.
- La señal de revisión usa `pull_request_review`, sin permisos concedidos
  al token, sin checkout, secretos ni artefactos. El listener vuelve a leer
  el estado real de las PRs y sus revisiones; no consume código ni artefactos
  producidos por una PR.
- `GITHUB_TOKEN` tiene `contents: read`, `pull-requests: read` e
  `issues: write`. La credencial del Project se mantiene separada.
- No añade tarjetas ausentes, modifica asignaciones ni crea issues.
  Las tarjetas archivadas y los estados terminados no retroceden. No se
  tocan las dailies ni el campo `Estado daily`; el workflow de dailies de
  la PR #50 tiene otro cometido.
- Revisar con Esra los workflows nativos del Project: no activar otra regla
  que compita por los mismos estados. Una regla autorizada de auto-add puede
  incorporar issues nuevas, pero no la configura esta entrega.
- Si se elimina una referencia de la descripción de una PR, la antigua
  tarjeta no se reconcilia automáticamente: revisarla expresamente.
- Si falta una tarjeta, la ejecución avisa y falla. Después de un merge,
  la issue podría haberse cerrado aunque falle el movimiento de la tarjeta;
  corregir la pertenencia y repetir la sincronización.

## Activación por Esra o una persona con permisos suficientes

### 1. Revisar y aceptar esta PR

Revisar configuración, script, workflows y tests. Fusionar mediante el
proceso habitual, sin saltarse revisión. Los cuatro archivos de instalación
deben llegar a la rama por defecto; actualmente esa rama es `dev`.
No se añade ningún secreto a los archivos versionados.

### 2. Configurar la credencial del Project

Esta acción **concede a la automatización capacidad para cambiar el Project**.
No es necesario entregar a David la credencial personal de Esra.

En el repositorio de GitHub:

1. Abrir **Settings → Secrets and variables → Actions**.
2. Pulsar **New repository secret**.
3. En **Name**, introducir exactamente `PROJECT_AUTOMATION_TOKEN`.
4. En **Secret**, pegar directamente una credencial autorizada para leer
   las issues del repositorio y leer/escribir el Project de la organización.
   No pegarla en chat, código, descripciones de PR, comandos o logs.
5. Guardar el secreto. Mantener sin cambios los nombres de estados y demás
   opciones del Project, salvo decisión explícita del equipo.

Un PAT clásico con scopes `repo` y `project` es compatible si la política
de la organización lo permite; son permisos amplios, por lo que se debe
elegir caducidad, limitar exposición y prever su rotación. Autorizar SSO si
la organización lo exige. Si esa política no lo permite, acordar una
alternativa antes de activar; una GitHub App requiere integración adicional
para generar y renovar tokens, no guardar un token temporal como permanente.

La autenticación local de `gh` no proporciona este secreto a Actions.
Ver el nombre del secreto tampoco demuestra que el valor tenga permisos.

### 3. Comprobar permisos y política de Actions

Esra debe verificar en **Settings → Actions → General** que la política
permite ejecutar los workflows y las actions utilizadas (`actions/checkout`
y `actions/github-script`), y que permite la escritura de issues solicitada
por el workflow. No hace falta autorizar a Actions para crear o aprobar PRs.
No ampliar permisos globales ni cambiar protecciones sin revisión.

Confirmar que la issue de prueba ya pertenece al Project número 3. Una PR
vinculada no incorpora por sí sola la tarjeta de la issue.

### 4. Verificar una sincronización explícita

Esta ejecución **puede mover tarjetas y, si se indica una PR ya fusionada,
cerrar sus issues**. Elegir y autorizar el caso antes de ejecutarlo.
No hay reproducción automática de eventos anteriores a la instalación.

En **Actions → Personal GitHub lifecycle → Run workflow**:

1. Seleccionar la rama `dev`.
2. Rellenar **pull_request_number** con la PR acordada para la prueba.
3. Dejar **issue_number** vacío.
4. Ejecutar y comprobar el resultado del job, la issue y su tarjeta.

Para sincronizar la PR #51, si sigue abierta y lista, introducir `51`.
Si su issue ya estaba en `In Review`, el resultado puede ser un no-op:
eso no demuestra una transición real. Si la PR ya está fusionada, ejecutar
solo si se quiere reconciliar su cierre y estado `Done`.

Alternativa por CLI, únicamente después de autorizar la ejecución real:

```bash
gh workflow run personal-github.yml \
  --repo T6-P5-Classification/bcn-price-classifier \
  --ref dev -f pull_request_number=51
```

Para marcar el inicio de una issue, usar el mismo formulario rellenando solo
**issue_number**. Crear una rama local no dispara un cambio en el Project.

### 5. Verificar el ciclo completo

Con un caso de prueba aprobado por Esra, comprobar y registrar los hechos:

- PR lista: issue abierta y tarjeta en `In Review`.
- Cambios solicitados: tarjeta en `In progress`; un push no despeja el bloqueo.
- Nueva solicitud al revisor bloqueante: tarjeta de nuevo en `In Review`.
- Merge revisado: PR fusionada, issue cerrada como completada y tarjeta en `Done`.
- Cierre sin merge: issue abierta, sin marcar `Done`.

La prueba puede crear/cambiar registros reales; no se ejecuta como parte de
la validación local de esta PR. Si la señal de revisión carece de metadatos
de PR, el listener falla de forma explícita: usar sincronización manual
autorizada y revisar los logs. No descargar artefactos ni ejecutar código
de la PR para intentar resolverlo.

## Validación local sin mutaciones remotas

Requiere Node.js 20 o superior y Python para comprobar el JSON:

```bash
node --check .github/scripts/personal-github.cjs
node --test .github/scripts/personal-github.test.cjs
python3 -m json.tool .github/personal-github.json
git diff --check
```

Los tests usan clientes simulados: no usan credenciales ni llaman a GitHub.
Su éxito no certifica los permisos de Actions ni cambios reales del Project.
La procedencia del script y los dos workflows es el kit portátil de ciclo
GitHub, versión `1.1.0`; la configuración corresponde a este repositorio.
