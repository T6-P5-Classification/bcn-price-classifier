# Daily Scrum — rutina del equipo

Rutina diaria de **David** (`davidlafuentem`), **Esra** (`esrakesken`), **Beste** (`BesteOzaydin`) y **Miguel** (`mmatesanzg23`).

No es un flujo de n8n ni una feature del proyecto: es la **ceremonia** con la que se mueve el Kanban, se desbloquean
tareas y se decide el siguiente paso. Vive en el mismo Project que las tareas, pero en **su propia vista**.

- **Cuándo:** de lunes a viernes, 15 min (referencia: 09:00 `Europe/Madrid`).
- **Tarjeta:** una por **equipo y día**, no una por persona.
- **Automatización:** [`../../.github/workflows/daily-scrum.yml`](../../.github/workflows/daily-scrum.yml).
- **Regla de oro:** la daily no construye, no aprueba specs y no sustituye la revisión humana.

## Rotación del facilitador/a

Una semana cada uno, en este orden:

| Semana | Facilitador/a |
|---|---|
| 1 (ancla: 2026-11-16) | David |
| 2 | Esra |
| 3 | Beste |
| 4 | Miguel |
| 5 | David (vuelve a empezar) |

La rotación se calcula sola con el número de semanas desde el ancla. El facilitador/a aparece mencionado en el
cuerpo de la tarjeta; **la tarjeta se asigna siempre a las cuatro personas**, para que nadie se pierda la notificación.

Roles por issue (dueño/a de la spec, constructor/a, revisor/a): ver [`../../.agents/team.md`](../../.agents/team.md).
El revisor/a nunca es quien construyó.

## Qué hace la automatización

| Paso | Detalle |
|---|---|
| Cuándo se ejecuta | Cron `30 6 * * 1-5` (UTC) → **08:30 en Madrid en verano**, **07:30 en invierno** |
| Fecha y semana | Se calculan en `Europe/Madrid`, no en UTC |
| Etiquetas | Crea `daily` y `scrum` si no existen (no rompe nada si ya están) |
| Duplicados | Si ya hay un issue titulado `[Daily] AAAA-MM-DD`, no crea otro |
| Issue | Título `[Daily] AAAA-MM-DD`, etiquetas `daily` + `scrum`, asignado a las cuatro personas |
| Manual | *Actions → Daily Scrum → Run workflow* con `forzar` para crear una tarjeta extra |

**No usa secretos ni PAT.** Solo el `GITHUB_TOKEN` que da Actions, con permiso `issues: write`. El Project se toca
solo desde la interfaz (auto-add y workflows integrados), nunca desde el workflow.

## El Project: un solo tablero, varias vistas

Hay **un único Project**. Se mantienen las vistas que ya existan y se añaden las de la daily.
Ninguna vista existente cambia de columnas, de filtro ni de nombre.

### Campos

| Campo | Para | Opciones | Valor por defecto |
|---|---|---|---|
| `Status` | Tareas, bugs y PRs | los que ya tenga el Project (**sin cambios**) | — |
| `Estado daily` | **Solo dailys** (campo nuevo) | `Inicio`, `¿Ha pasado algo?`, `Cierre` | `Inicio` |

Al ser un campo con valor por defecto, las tareas también reciben `Inicio` en `Estado daily`. Es inofensivo:
las vistas de tareas no forman columnas con ese campo. Lo mismo al contrario: las dailys reciben un `Status`,
que queda oculto en la vista `Dailies`.

### Vistas

| Vista | Estado | Layout | Columnas (campo) | Filtro |
|---|---|---|---|---|
| vistas existentes | **Sin cambios** | — | — | — |
| `Dailies` | **Nueva** | Board | `Estado daily` | `label:daily` |
| `Historial de dailys` | **Nueva** | Table | — | `label:daily`, orden por fecha descendente |

### Cómo se mueven las tarjetas

| Qué | Campo | Cómo se mueve |
|---|---|---|
| Tareas, bugs y PRs | `Status` | **Automático**, con los workflows integrados de GitHub |
| Dailys | `Estado daily` | **Manual**, arrastrando la tarjeta |

Movimiento de una daily:

| Etapa | Quién | Cómo |
|---|---|---|
| `Inicio` | Automático | Valor por defecto del campo al entrar la tarjeta en el Project |
| `¿Ha pasado algo?` | Facilitador/a | Arrastra la tarjeta al empezar la ronda de cada persona |
| `Cierre` | Facilitador/a | Arrastra la tarjeta al terminar y cierra el issue |

Cerrar el issue **no** mueve la tarjeta a `Cierre`: los workflows integrados solo actúan sobre `Status`, no sobre
campos personalizados. Es a propósito, porque se pidió que las dailys se muevan a mano.

## Historial

| Qué | Dónde queda |
|---|---|
| Transiciones y notas de cada daily | Tabla **`Historial de la daily`** dentro del cuerpo del issue (`Etapa · Hora · Qué ha pasado · Responsable`) |
| Lista cronológica de dailys | Vista `Historial de dailys` (una fila por día) |
| Historial de tareas | En el **timeline del issue**, como hasta ahora: GitHub registra ahí los cambios de `Status` |

El cuerpo del issue es la fuente de verdad de cada daily y se conserva aunque la tarjeta cambie de columna.

## Puesta en marcha (una sola vez, en la interfaz)

1. **Etiquetas.** No hace falta crearlas a mano: el workflow crea `daily` y `scrum` en su primera ejecución.
2. **Campo `Estado daily`.** Project → `+` (Add field) → *Single select* → nombre `Estado daily`, opciones
   `Inicio`, `¿Ha pasado algo?`, `Cierre` → valor por defecto `Inicio`.
3. **Vista `Dailies`.** Project → *New view* → *Board* → nombre `Dailies` → en *Column field* elegir
   `Estado daily` → filtro `label:daily`.
4. **Vista `Historial de dailys`.** *New view* → *Table* → filtro `label:daily` → ordenar por la fecha.
5. **Auto-add.** El Project tiene que recoger los issues con `label:daily`.
6. **Workflows integrados** (Project → *Workflows*):
   - *Item added to project* → `Status`: el estado inicial que ya uséis.
   - *Item closed* → `Status`: `Done`.
   - *Pull request merged* → `Status`: `Done`.
7. **Primera prueba.** *Actions → Daily Scrum → Run workflow* (`forzar: false`) y comprobar que la tarjeta
   aparece en `Dailies` en la columna `Inicio`.

## Personalizar

| Qué | Dónde |
|---|---|
| Hora de creación | `cron` de `.github/workflows/daily-scrum.yml` (está en UTC) |
| Orden de la rotación o fecha ancla | Variables `nombres`, `usuarios` y `ancla` del paso *Calcular fecha, semana y facilitador/a* |
| Texto de la tarjeta (tablas y checklists) | `env.PLANTILLA` del mismo workflow. Marcadores: `{{FECHA}}`, `{{SEMANA}}`, `{{NOMBRE}}`, `{{USUARIO}}` |
| Columnas de la daily | Opciones del campo `Estado daily` y la vista `Dailies` (en la interfaz) |

Si se añade o cambia una persona: actualizar `nombres` y `usuarios` en el workflow **y** la tabla de
[`../../.agents/team.md`](../../.agents/team.md).

## Problemas frecuentes

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| No aparece la tarjeta de hoy | El cron no ha corrido o ya existía | *Run workflow* con `forzar`; si el repo estuvo inactivo, hacer cualquier push |
| La tarjeta existe pero no está en el Project | Falta el auto-add o el filtro no incluye `label:daily` | Revisar *Auto-add to project* y añadir la tarjeta a mano mientras |
| La tarjeta entra pero sin columna | El campo `Estado daily` no tiene valor por defecto `Inicio` | Editar el campo y ponerlo |
| No puedo arrastrar la tarjeta | La vista no tiene `Estado daily` como *Column field* | Cambiar el *Column field* de la vista `Dailies` |
| Se crea una tarjeta el sábado | El `cron` se editó a `* * *` | Volver a `30 6 * * 1-5` (1-5 = lunes a viernes) |
| El facilitador/a no es el esperado | El ancla no es el lunes de la semana 1 | Ajustar `ancla` en el workflow |

## Lo que esta rutina no hace

- No aprueba specs ni mergea PRs.
- No recalcula importes ni toca la app.
- No escribe secretos: el workflow no usa ninguno.
- No crea issues de trabajo: los bloqueos detectados en la daily se convierten en issues con etiqueta `bloqueado`
  y responsable, y entran en el tablero como cualquier tarea.