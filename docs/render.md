# Demo en Render con persistencia en Atlas (#33)

## Estado

Preparación en `feat/33-render-deployment`, desde `dev` en `ec04c87`.
**No hay un despliegue remoto ni una URL pública verificados todavía.**
Objetivo: ejecutar la app en Render Free y conservar predicciones y respuestas
en Atlas Free, independientemente de los reinicios de la app. Configurar una
variable no prueba la conexión; esa integración sigue pendiente.

## Modelo distribuido

Se autoriza distribuir `models/app_model.joblib` con el repositorio, incluso
cuando pase a ser público. Esto sustituye la exclusión anterior únicamente
para ese archivo. No se publican raw, credenciales, registros de uso ni otros
modelos por defecto. Borrarlo después no retira copias ya descargadas.

Es el champion XGBoost `xgboost-trial34-6e9bc3e`, generado previamente por el
entrenador de esa revisión y usado en las pruebas locales de MongoDB. Se
conserva su identificación original; no se atribuye el ajuste a esta entrega.
Tiene 1.040.255 bytes; SHA-256 y dependencias figuran en
`models/app_model.manifest.json`. La procedencia, parámetros y umbrales se
contrastan con los resultados actuales del champion.

Incluye el Pipeline ajustado, etiquetas, opciones del formulario, coordenadas
medianas por barrio y métricas agregadas existentes. No incluye una tabla de
anuncios ni identificadores de anfitriones. Esto no es una auditoría formal de
privacidad del modelo aprendido. Solo cargar artefactos de confianza:
`joblib.load` puede ejecutar código. Una huella detecta modificaciones, no
demuestra seguridad ni calidad del modelo.

**Solo contiene el champion.** La app no asigna sesiones al challenger si no
está en el artefacto. El entrenador actual genera ambos modelos: no se ejecuta
para esta preparación ni se reetiqueta el archivo anterior. Su sustitución
requiere revisar contenido, actualizar manifiesto y repetir pruebas de
compatibilidad; no basta con cambiar la huella.

Render copia el archivo al construir la imagen. Al arrancar, la app lo carga
sin entrenar ni acceder al snapshot. Generarlo es un proceso independiente
que sí requiere train. No se reevalúa el test reservado para esta entrega.

## Configuración prevista

- Render: repositorio del proyecto, rama `feat/33-render-deployment` durante
  la prueba previa a la PR; cambiar a `dev` solo después de revisión y merge.
  Docker, Frankfurt y plan Free.
  Sin disco ni extras de pago. La capacidad en sus 512 MB sigue sin comprobarse.
- Variable secreta `MONGODB_URI`: usuario de app con `readWrite` solo sobre
  `bcn_price_classifier`, incluyendo `/bcn_price_classifier` en la URI. Nunca
  el administrador. No publicar credenciales en Git, chat, capturas o logs.
- No necesita `FEEDBACK_DIR`; el Dockerfile ya utiliza `PORT`. No añadir
  entrenamiento al comando de inicio.
- Autorizar en Atlas los rangos de IP de salida mostrados por Render para
  este servicio. La IP local no autoriza a Render. No abrir `0.0.0.0/0`.
- Revisar el despliegue automático: publicar en la rama configurada podría
  provocar un despliegue remoto. No cambiar otros servicios.
- Render Free puede suspenderse por inactividad. Atlas Free limita espacio
  y aplica condiciones de pausa. Persistencia no significa retención indefinida
  ni respaldo; no activar copias u opciones de pago sin autorización.
- El MongoDB local y sus registros no se publican ni migran automáticamente.

## Verificación numerada

1. **Integridad local:** en la rama indicada y con dependencias del proyecto,
   ejecutar `python -m pytest -q tests/test_deployment_artifact.py tests/test_docker.py`.
   Esperado: huella, versión, esquema, parámetros y dependencias correctos.
   Solo lee el artefacto confiable y resultados locales; pytest puede crear
   cachés. No requiere raw, Atlas ni credenciales.
2. **Construcción local:** `docker build -t bcn-price-demo:local .` puede
   descargar dependencias y crea una imagen local, sin entrenar ni publicar.
   Debe superar la copia del modelo. Ante fallo, revisar archivos y versiones;
   no incorporar raw ni secretos al contexto para resolverlo.
3. **Publicación para pruebas:** commit y push de `feat/33-render-deployment`,
   cada operación con permiso explícito. Seleccionar esa rama en Render para
   comprobar el despliegue antes de abrir la PR. No pulsar Deploy hasta que
   el modelo esté disponible en la rama remota seleccionada. No integrar en
   `dev` sin revisión y autorización de merge.
4. **Creación remota pendiente:** revisar Free, rama, despliegue automático y
   variables secretas. Crear el servicio envía código y modelo a Render,
   consume cuota de construcción y puede exponer una URL pública. Requiere
   permiso. Consultar IP de salida y autorizarla en Atlas.
5. **Flujo remoto pendiente:** comprobar despliegue y formulario en HTTPS.
   Predecir con datos ficticios escribe en Atlas; enviar un precio ficticio
   guarda una respuesta. Verificar recuentos, sin publicar documentos. Un
   HTTP 200 por sí solo no acredita el funcionamiento de la base.
6. **Persistencia remota pendiente:** con permiso para interrumpir/reiniciar
   la app, anotar recuentos, reiniciar solo ese servicio y consultar Atlas.
   Deben mantenerse los registros. No borrar el clúster ni los datos.
7. **Cierre pendiente:** documentar URL, revisión y resultados comprobados.
   Ante fallo, revisar errores sanitizados, credenciales, base y acceso de red.
   No imprimir secretos ni cambiar a CSV para ocultar un fallo de MongoDB.

Esta preparación no acredita los pasos remotos ni da por terminada la issue.

## Validación local realizada

- Suite completa con snapshot disponible y MongoDB local aislado: **358 tests
  correctos**, incluidos los de integración local; tres avisos gráficos ya
  existentes. Los cuatro nuevos controles verifican el artefacto distribuido.
- Imagen Docker construida con Python 3.12 en Linux ARM64. AppTest ejecutado
  dentro de un contenedor sin red ni raw, con límites de 512 MB y 0,5 CPU:
  formulario, predicción champion y respuesta ficticia de 165 € guardados en
  CSV temporal, sin errores. El contenedor de prueba fue eliminado al terminar.
  Esto no acredita Atlas, Render Free ni comportamiento con varios usuarios.
- Huella y tamaño coinciden con el manifiesto; no se ha modificado el binario,
  reentrenado ni evaluado nuevamente el test reservado.
- La primera sonda manual buscaba el botón sin su icono y falló al localizarlo;
  corregida la sonda para el rótulo real, el flujo completo pasó sin cambios
  en la aplicación. No fue un defecto del artefacto.
