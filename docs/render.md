# Demo en Render con persistencia en Atlas (#33)

## Estado

Rama `feat/33-render-deployment`, desde `dev` en `ec04c87`.
**Demo:** https://bcn-price-classifier.onrender.com.
Código desplegado: `657afa5`; las actualizaciones documentales posteriores no
cambian esa identificación. Servicio Docker en Frankfurt, Render Free y Atlas
Free. El flujo de predicción y respuesta se comprobó con un ejemplo ficticio;
los registros permanecieron tras reiniciar la app, según confirmó el usuario.
Detalle de las evidencias y sus límites al final de esta guía.

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

## Configuración del despliegue

- Render: repositorio del proyecto, rama `feat/33-render-deployment` durante
  la prueba previa a revisión; cambiar a `dev` solo después de revisión y merge.
  Docker, Frankfurt y plan Free.
  Sin disco ni extras de pago. Se comprobó una sesión; no hay prueba de carga
  que garantice capacidad con múltiples usuarios en sus 512 MB y 0,1 CPU.
- Variable secreta `MONGODB_URI`: usuario de app con `readWrite` solo sobre
  `bcn_price_classifier`, incluyendo `/bcn_price_classifier` en la URI. Nunca
  el administrador. No publicar credenciales en Git, chat, capturas o logs.
- No necesita `FEEDBACK_DIR`; el Dockerfile ya utiliza `PORT`. No añadir
  entrenamiento al comando de inicio.
- Autorizar en Atlas los rangos de IP de salida mostrados por Render para
  este servicio. La IP local no autoriza a Render. No abrir `0.0.0.0/0`.
- Auto-Deploy: **Off**, observado en la captura de configuración. No cambiar
  otros servicios. Tras merge, cambiar la rama a `dev` y desplegar requiere
  autorización separada; no es consecuencia de solicitar revisión.
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
   comprobar el despliegue antes de solicitar la revisión final. No pulsar
   Deploy hasta que
   el modelo esté disponible en la rama remota seleccionada. No integrar en
   `dev` sin revisión y autorización de merge.
4. **Configuración remota, solo lectura si el servicio ya existe:** revisar
   Free, rama, despliegue automático y
   variables secretas. Crear el servicio envía código y modelo a Render,
   consume cuota de construcción y puede exponer una URL pública. Requiere
   permiso. Consultar IP de salida y autorizarla en Atlas.
5. **Flujo remoto:** comprobar despliegue y formulario en HTTPS.
   Predecir con datos ficticios escribe en Atlas; enviar un precio ficticio
   guarda una respuesta. Para reproducir el caso, usar 165 € por noche:
   categoría real según los umbrales actuales, Medio; el acierto depende de
   la categoría predicha, no de un resultado prefijado. En Data Explorer,
   consultar `feedback` con `{"real_price": 165}`, localizar la respuesta de
   esta prueba por su fecha y consultar `predictions` con su `prediction_id`.
   Verificar el enlace, sin publicar documentos. Un
   HTTP 200 por sí solo no acredita el funcionamiento de la base.
6. **Persistencia remota:** con permiso para interrumpir/reiniciar
   la app, anotar recuentos e identificador de prueba, usar Render → Manual
   Deploy → Restart service, esperar a Live y consultar ambas colecciones
   en Atlas con ese mismo `prediction_id`, sin enviar nuevas predicciones.
   Deben mantenerse los registros. No borrar el clúster ni los datos.
7. **Revisión humana pendiente:** contrastar URL, revisión y resultados documentados.
   Ante fallo, revisar errores sanitizados, credenciales, base y acceso de red.
   No imprimir secretos ni cambiar a CSV para ocultar un fallo de MongoDB.

Estas verificaciones no autorizan aprobación, merge ni cambio de rama del
servicio. La revisión humana y la integración en `dev` siguen pendientes.

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

## Validación remota realizada

- Capturas del usuario: Render indica despliegue Live de `657afa5` desde esta
  rama, plan Free, Docker y URL HTTPS indicada. Auto-Deploy quedó en Off.
- El usuario comprobó que el formulario abre y ejecutó una predicción
  ficticia. Las primeras escrituras fallaron; una comprobación local segura
  identificó rechazo de autenticación. Tras corregir las credenciales fuera
  del repositorio, el usuario confirmó el guardado. No se registran valores
  secretos ni se deduce qué contraseña se cambió.
- Evidencia mostrada por el usuario en Atlas: una respuesta de 165 € con
  categoría real Medio y `correct: true`, y su predicción enlazada por el
  mismo `prediction_id`, categoría predicha Medio, XGBoost champion versión
  `xgboost-trial34-6e9bc3e`. No se copian documentos, identificadores de sesión
  ni credenciales a esta guía.
- Tras el procedimiento guiado de reinicio de la app, el usuario confirmó
  que ambos registros seguían presentes. Esa verificación manual no fue
  ejecutada directamente por el agente en Atlas.
- El agente consultó directamente el endpoint HTTPS `/_stcore/health` y
  recibió **HTTP 200**, cuerpo `ok`. Esta lectura no prueba por sí sola el
  guardado: la evidencia de Atlas procede de las comprobaciones anteriores.
- No hay pruebas de carga, respaldo/restauración, retención prolongada ni
  garantía de disponibilidad continua. La cuenta Free puede suspender la app
  por inactividad. La base persistente tiene sus propios límites y política
  de inactividad; guardar registros no los convierte en datos verificados
  apropiados para reentrenar sin revisión.
