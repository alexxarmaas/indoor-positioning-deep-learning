# Integración del MLP del TFG en IPSCampaign

Rama: `feat/tfg-mlp-comparison`. Esta extensión convive con los métodos originales;
no modifica el preprocesamiento ni el código interno de `WKNN`, `SVR` o `DeepLearning`.

## Métodos nuevos

- `MLP_TFG_ROBUST`: ocho RSSI, imputación de -100 dBm y escalado de RSSI.
- `MLP_TFG_MASK`: ocho RSSI más ocho indicadores binarios de presencia; solo los RSSI se escalan.
- `WKNN_TFG`: reutiliza `applyWKNNmethod` de IPSCampaign, con las mismas ocho características RSSI y las mismas capturas de referencia/evaluación que los MLP.

Los MLP tienen la arquitectura (64, 32), ReLU, Adam, alpha 0.0001,
learning rate 0.001, 2000 iteraciones máximas, early stopping 15%,
paciencia 50 y semilla 42, coherente con `tfg-ips/scripts/04_train_mlp.py`.
Se entrenan **de cero para cada comparación** a partir de la campaña de referencia:
nunca se aplica a una evaluación PR un `.joblib` que pudiera haber visto esas capturas.

## Preparación de huellas

`TFGFeaturesService.py` lee las tablas importadas `Capture` y
`Beacon_BLE_Signal`. Construye **una fila por captura**, agregando por mediana
las lecturas de cada una de las ocho MAC de `configs/tfg_beacon_mapping.json`.
No se filtra por canal (37/38/39) ni por protocolo: es el contrato del
dataset de `tfg-ips/scripts/02_build_dataset.py`.

Cada fila tiene `[rssi_b01..rssi_b08, present_b01..present_b08, x, y, z]`.
La ausencia de una MAC se representa mediante RSSI -100 y presencia 0. Una
lectura observada de -100 conserva presencia 1. Se omiten únicamente las
capturas sin señales BLE. Se mantienen las mediciones individuales y sus
orientaciones, sin mezclar las huellas durante el entrenamiento.

El MLP aprende únicamente `(x,y)`. Se exige que `z` sea constante y
coincida en las campañas, y se añade esa Z fija a cada predicción para
ajustarse al contrato tridimensional de IPSCampaign. Las predicciones
de las diferentes capturas del mismo punto físico se fusionan mediante
mediana coordenada a coordenada, reutilizando la función existente
`fusePredictionsByPointMedian`.

**Diferencia importante:** el `WKNN` estándar del fork utiliza una referencia
agregada por punto y valores RSSI basados en máximos y canales seleccionados.
Por eso la comparación de algoritmos sobre un mismo procesamiento debe usar
`WKNN_TFG` frente a `MLP_TFG_ROBUST`/`MLP_TFG_MASK`. El `WKNN` original
permanece disponible, pero sus resultados no son directamente equivalentes
al protocolo TFG.

## Preparar automáticamente la campaña PR del 16 de abril

Desde PowerShell, en la raíz del fork, ejecutar:

```powershell
python IPSCM/experiments/prepare_tfg_same_day_campaign.py --source ../tfg-ips/data/raw/toma_datos_16_4.sqlite3 --output Campañas/TFG_16_4
```

El script usa la misma cuadrícula canónica que
`tfg-ips/scripts/07_same_day_evaluation.py`: las 60 posiciones
se ordenan por `MIN(Capture.Id)`, entrenamos con los puntos físicos
impares y evaluamos en los pares. La copia filtrada preserva las tablas
SQLite necesarias y no altera el original.

Aparecerán:
- `tfg_reference_odd.sqlite3` (referencia PR).
- `tfg_evaluation_even.sqlite3` (evaluación PR).
- `01_ble.conf`, `02_ale_points.conf` y `03_ref_points.conf` en ese orden.
- `manifest.json`, con conteos y SHA-256 del origen.

En la página de importación de IPSCampaign selecciona como **puntos
aleatorios** `tfg_evaluation_even.sqlite3`, como **referencia**
`tfg_reference_odd.sqlite3` y como los **tres archivos de configuración**
los `.conf` de arriba, exactamente en el orden `01`, `02`, `03`.
La interfaz solo admite `.conf`, no `.ini`. Si se seleccionan en bloque,
comprueba el orden en el resumen antes de guardar.

`Duration=0` es un campo requerido por el importador y no modifica las
lecturas BLE originales.

## Importar campañas

1. Preparar **fuera del programa** las bases SQLite de referencia y evaluación,
   preservando las tablas originales y sus coordenadas/orientaciones. No
   importar todavía las 59 muestras PA finales para ajustar modelos.
2. En IPSCampaign (http://localhost:4200) usar la opción de importar campaña,
   indicando la base de evaluación, al menos una base de referencia y las
   configuraciones cuando corresponda.
3. Seleccionar `WKNN_TFG` y un rango de k (ej. 1 a 10) con métrica euclídea;
   después `MLP_TFG_ROBUST` y `MLP_TFG_MASK`.
4. Comparar los errores por punto físico sobre **el mismo par de campañas**.
   Documentar semillas, k, muestras y geometría. El campo
   `overlappingCoordinates` de la respuesta informa de posiciones compartidas
   entre referencia y evaluación; no equivale a un comprobante de ausencia
   de fuga temporal o por campañas.

**Advertencias metodológicas:**

- La aplicación diferencia referencia y evaluación por la relación entre
  campañas, pero **no conoce automáticamente** las etiquetas PR y PA de
  `tfg-ips`; es obligatorio que quien importe respete esa separación.
- En `tfg-ips` el preprocesamiento parte de las coordenadas originales
  del SQLite; IPSCampaign puede reasignar coordenadas mediante ficheros de
  configuración. Es necesario comprobar que el sistema de coordenadas sea
  idéntico al contrastar métricas entre ambos proyectos.
- El flujo TFG ignora los selectores antiguos de canales, protocolos y número
  máximo de muestras, pues la entrada reproduce la mediana de todas las
  lecturas BLE de cada captura. La interfaz oculta estos campos.
- Las pruebas unitarias del adaptador validan el contrato de entrada, pero
  aún hay que comprobar la igualdad fila a fila contra
  `tfg-ips/data/interim/fingerprints/fingerprints_all.csv` usando una campaña
  real importada. Hasta entonces no afirmar que los resultados se reproducen
  numéricamente respecto al repositorio original.

## Construcción y tests

Esta rama usa PyTorch CPU por defecto para evitar las descargas de CUDA,
manteniendo los métodos PyTorch funcionales sin aceleración por GPU.

```powershell
docker compose --progress plain build server client
docker compose up -d
docker compose exec server python -m unittest discover -s tests -p "test_tfg_*.py" -v
docker compose exec server python -m unittest discover -s tests -v
```

No borres el volumen `ipscm_ipscm-data` si deseas conservar las campañas
importadas. `docker compose up -d` preserva ese volumen.
