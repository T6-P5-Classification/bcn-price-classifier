# Predicciones y feedback en MongoDB (#27)

## Alcance y decisiones

Rama: `feat/27-mongodb-feedback`, desde `dev` en `6e9bc3e`.

Se conserva el contrato de [feedback.md](feedback.md): la app llama a
`src/feedback_store.py`; este selecciona CSV o delega en `src/database.py`.
Las tareas de métricas, datos nuevos y comparación de variantes conservan
las funciones públicas y columnas. No se cambia entrenamiento ni se evalúa
de nuevo el test reservado del modelo.

| Configuración | Backend |
|---|---|
| `MONGODB_URI` configurado, sin argumento `directory=` | MongoDB |
| URI ausente o vacía | CSV local, con aviso visible en la app |
| `directory=` explícito en una llamada | CSV en esa carpeta, para aislar pruebas/consumidores locales |

`FEEDBACK_DIR` solo cambia la carpeta CSV. La app no pasa `directory=` y
respeta la selección por URI. Si MongoDB está configurado pero no responde,
**no** se usa CSV como sustitución silenciosa: se muestra la predicción,
se informa del fallo de guardado y no se confirma una respuesta no guardada.
El mensaje «configurado» no afirma que el servidor ya se haya conectado.

MongoDB recibe los mismos campos, sin identificadores de anfitriones,
direcciones exactas, texto libre ni campos adicionales. Las coordenadas de
la app siguen siendo las referencias por barrio de su artefacto. No se
guardan datasets de entrenamiento, modelos ni predicciones del test.

## Conexión y garantías

- `MONGODB_URI` identifica el servidor y, opcionalmente, la base. Docker
  Compose ya pasa `mongodb://mongo:27017/bcn_price_classifier` a la app.
  Si una URI no incluye base, se utiliza `bcn_price_classifier`.
- Colecciones `predictions` y `feedback`; cada una tiene un índice único
  `prediction_id_unique`. Se crean al primer acceso. Si hay duplicados
  anteriores que impidan crear el índice, se informa del fallo: no se borran
  ni reparan registros automáticamente.
- Un cliente y su conjunto de conexiones se reutilizan dentro del proceso.
  No se abre una conexión al importar el módulo. Los tiempos de selección
  del servidor, conexión y espera de respuesta se limitan a 3.000 ms cada
  uno; no se promete que toda una operación termine en exactamente 3 s.
- La escritura de feedback es un insert con unicidad en el servidor; no
  depende de un bloqueo local ni sustituye la respuesta anterior.
- Las lecturas devuelven DataFrames con el esquema común, sin `_id` de
  MongoDB ni campos ajenos al contrato. Las lecturas incompletas se rechazan.
- Los errores del driver se transforman en `StorageError` con mensajes
  genéricos, sin mostrar URI, credenciales ni documentos del servidor.

No se migran CSV existentes: siguen intactos y no aparecen automáticamente
en las consultas MongoDB. Cambiar de backend no une ambos historiales. Una
migración posterior requiere alcance y autorización propios.

Atlas y la configuración de un servidor remoto pertenecen a #28. Aquí se
usa exclusivamente una instancia local de prueba; no se crean cuentas,
credenciales ni recursos de pago. El MongoDB de Compose no tiene autenticación
y no publica puertos: es una configuración local, no una receta de producción.

## Verificación manual con la app y Compose

Prerrequisitos: rama indicada, entorno del proyecto, Docker en ejecución,
snapshot local del README y artefacto champion compatible. Usar un entorno
de pruebas propio, no un Compose que esté atendiendo a otras personas.

1. **Pruebas locales:** desde la raíz, ejecutar
   `.venv/bin/python -m pytest -q`. Las pruebas pueden crear cachés locales.
   Las de integración se omiten si no se proporciona `TEST_MONGODB_URI`.
   La suite existente incluye una comprobación que necesita el snapshot raw;
   no lo añadas a Git ni confundas su ausencia con una regresión del driver.
2. **Preparar el artefacto, solo si hace falta:**
   `python -m src.train_app_model` lee train y reemplaza
   `models/app_model.joblib`. Conserva antes una copia si necesitas mantener
   el artefacto existente. No evalúa test ni publica el archivo.
3. **Arrancar el entorno local:** `docker compose up --build --wait`.
   Descarga imágenes/dependencias si faltan, construye la imagen y crea
   contenedores y volúmenes locales. Esperado: app y MongoDB sanos. Abrir
   `http://localhost:8501`: debe indicar «Almacenamiento configurado: MongoDB».
4. **Guardar un ejemplo:** introducir características ficticias coherentes
   y pulsar **Predecir**. Esto escribe una predicción en la base local.
   Escribir un precio de ejemplo de 165 € y pulsar **Enviar**: escribe una
   respuesta. Con los umbrales actuales, su categoría real debe ser Medio;
   el mensaje de acierto depende de la categoría predicha, no está prefijado.
5. **Comprobar recuentos, solo lectura:** ejecutar

   ```sh
   docker compose exec mongo mongosh bcn_price_classifier --quiet --eval 'printjson({predictions: db.predictions.countDocuments({}), feedback: db.feedback.countDocuments({})})'
   ```

   Los recuentos deben aumentar en una predicción y una respuesta respecto
   al inicio. No imprimir documentos individuales ni datos reales en chat.
6. **Persistencia:** `docker compose down` detiene los servicios de este
   entorno y conserva los volúmenes. `docker compose up --wait` los recrea;
   repetir el paso de lectura debe mantener los recuentos. **No usar `down
   -v`: borraría los datos de esos volúmenes.**
7. **Modo CSV:** en otra ejecución local, sin `MONGODB_URI`, abrir Streamlit
   con `streamlit run app/app.py`. Debe avisar «CSV local» y «MongoDB no está
   configurado». Pulsar Predecir/Enviar escribe en la carpeta CSV, no en
   MongoDB. No mezclar este historial con el anterior al interpretar métricas.

Ante fallo, comprobar primero que se reconstruyó la imagen con PyMongo, el
backend indicado en pantalla y los servicios del entorno. No pegar la URI
con credenciales en chat ni borrar datos/volúmenes para resolver el problema.
Los fallos y envíos simultáneos se verifican automáticamente sin detener una
base compartida.

## Integración automatizada con MongoDB local

`tests/test_database_integration.py` necesita una instancia local de MongoDB.
Configurar `TEST_MONGODB_URI` con su dirección de loopback (`localhost`,
`127.0.0.1` o `::1`) y ejecutar:

```sh
python -m pytest -q tests/test_database_integration.py
```

Este comando **crea y elimina** únicamente bases temporales con un nombre
aleatorio `test_issue27_*` y datos sintéticos. Se niega a usar servidores
remotos. Si se suministra una dirección pero el servidor no responde, falla
en lugar de presentar la integración como comprobada. No usar credenciales
ni bases de una persona o entorno ajeno. Las pruebas normales eliminan de
su entorno `MONGODB_URI` para no acceder accidentalmente a la base de quien
las ejecuta; cada prueba de integración configura su base aislada.

## Límites

- Las pruebas de servidor local no acreditan conexión a Atlas ni despliegue
  en Render. El panel de métricas y las otras tareas se integran por separado.
- No se ejecuta una migración, borrado de datos de usuario, retención ni
  reentrenamiento. No se oculta un fallo de MongoDB escribiendo en otro sitio.
- Las lecturas cargan el historial completo en memoria, como la interfaz
  CSV anterior. Agregación/paginación y grandes volúmenes quedan pendientes.
- Una pérdida de conexión durante un insert puede dejar incierto si se
  guardó. No se promete entrega exactamente una vez de todas las predicciones;
  sí se impide guardar dos respuestas para el mismo `prediction_id`.
- Abrir una PR, publicar, aprobar y fusionar siguen siendo acciones distintas
  y no quedan autorizadas por ejecutar estas comprobaciones.

## Validación realizada en esta entrega

- Suite completa con recursos disponibles: **228 tests correctos**, incluidos
  ocho tests contra MongoDB local real. Tres avisos de gráficos no interactivos
  proceden de pruebas sintéticas ya existentes del notebook del champion.
- Pruebas unitarias: selección de backend, compatibilidad CSV, conversión de
  tipos NumPy, índices, duplicados, errores sanitizados y ausencia de cambio a
  CSV cuando falla MongoDB. No acreditan conexión a un servidor remoto.
- Pruebas reales: MongoDB 7.0.14 local; guardado/lectura y umbrales exactos,
  unicidad con cuatro inserts concurrentes y reconexión sin perder registros.
- Docker Compose construido con PyMongo 4.19.0 y ejecutado en Linux ARM64,
  Python 3.12: ambos servicios sanos, MongoDB sin puertos publicados y app con
  HTTP 200. El entorno Python local de tests usa Python 3.15.
- AppTest dentro del contenedor con el champion real: una predicción y un
  precio ficticio de 165 € generan documentos enlazados, sin archivos CSV.
  Se conservaron tras `down`/`up`, manteniendo el volumen de MongoDB.
- MongoDB detenido solo en el entorno aislado: la app conserva la predicción,
  informa del fallo, no ofrece feedback para un guardado fallido y no crea CSV.
  Sin URI, se comprobó el aviso de CSV y el guardado local en carpeta temporal.
- El usuario confirmó la prueba manual en la app local. La consulta posterior,
  solo de recuentos, verificó una predicción y una respuesta en MongoDB.

Estas comprobaciones no implican despliegue en Render, integración con Atlas,
publicación ni una nueva evaluación de test. Solo se usaron entradas y precios
de ejemplo para los registros operativos. No se alteraron CSV de usuario.
