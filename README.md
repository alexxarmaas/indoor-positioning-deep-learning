# Estudio de técnicas de Machine Learning basadas en fingerprinting aplicadas al posicionamiento en interiores

Repositorio desarrollado como parte de un **Trabajo de Fin de Grado del Grado en Ingeniería Informática**. El proyecto estudia técnicas de aprendizaje automático y aprendizaje profundo para posicionamiento en interiores mediante huellas de señal BLE o fingerprinting.

El objetivo principal es comparar **Weighted K-Nearest Neighbours (WKNN)** con tres métodos neuronales implementados en el sistema:

- Perceptrón multicapa con Scikit-learn.
- Red neuronal residual con PyTorch.
- Modelo híbrido multicanal que combina los canales BLE 37, 38 y 39.

La comparación incluye un escenario controlado y otro más próximo a un despliegue real, con capturas de varias jornadas y una baliza eliminada completamente.

## Memoria del Trabajo de Fin de Grado

[Estudio de técnicas de Machine Learning basadas en fingerprinting aplicadas al posicionamiento en interiores](./Estudio_de_técnicas_de_Machine_Learning_basadas_en_fingerprinting_aplicadas_al_posicionamiento_en_interiores_v2.pdf)

## Motivación

Los sistemas basados en fingerprinting construyen una base de referencia a partir de las intensidades de señal recibidas desde diferentes balizas. Una nueva huella RSSI se compara con esa base para estimar la posición.

Aunque WKNN es una solución sencilla y extendida, su rendimiento puede verse afectado por la variabilidad temporal, la orientación del receptor, la propagación multicamino, la presencia de obstáculos y la pérdida de una baliza. Este TFG analiza si los modelos neuronales pueden aprender representaciones más robustas y conservar su exactitud cuando las condiciones dejan de ser ideales.

## Objetivos

### Objetivo general

Diseñar, implementar y evaluar métodos de aprendizaje profundo para posicionamiento en interiores basado en fingerprinting BLE, comparándolos con WKNN en distintos escenarios experimentales.

### Objetivos específicos

- Integrar los métodos neuronales en IPS Campaign Manager.
- Mantener WKNN como método de referencia.
- Procesar campañas de una o varias jornadas.
- Permitir varias orientaciones por punto y fusionar sus predicciones.
- Analizar individualmente los canales BLE 37, 38 y 39.
- Desarrollar un modelo híbrido multicanal.
- Estudiar el comportamiento ante la ausencia de una baliza.
- Evaluar media, mediana, percentil 90 y error máximo.
- Proporcionar herramientas reproducibles para repetir los experimentos.

## Métodos de posicionamiento

### WKNN

WKNN selecciona las huellas más próximas a la muestra de evaluación y calcula la posición mediante una media ponderada por la inversa de la distancia.

La interfaz permite configurar la métrica euclídea, Manhattan, Chebyshev o Minkowski; el número de vecinos; el algoritmo de búsqueda; el canal BLE y el límite de muestras RSSI.

WKNN construye una huella por punto de referencia. Para cada baliza obtiene el RSSI máximo de cada orientación y promedia los máximos disponibles.

### Perceptrón multicapa con Scikit-learn

El método DeepLearning utiliza un ensemble de perceptrones multicapa para aprender una regresión entre las huellas RSSI y las coordenadas tridimensionales.

Incluye selección automática de configuraciones, estandarización, representación de RSSI absoluto y relativo, máscara de señales ausentes, sustitución de valores ausentes por -110 dBm, agregación del ensemble mediante mediana y recorte espacial de las predicciones.

### Red neuronal residual con PyTorch

DeepLearningPyTorch emplea una red multicapa con conexiones residuales. Puede ejecutarse mediante CUDA o CPU e incorpora normalización, tratamiento de valores ausentes, entrenamiento reproducible y agregación mediante mediana.

### Modelo híbrido multicanal

DeepLearningHybrid concatena las huellas de los canales 37, 38 y 39. Las tres observaciones complementarias reducen la dependencia respecto a las fluctuaciones de un único canal. Este método presenta el comportamiento más robusto en el escenario multijornada sin la baliza 5.

## Escenarios experimentales

### Escenario 1: jornada única con todas las balizas

Representa unas condiciones cercanas al caso ideal:

- Capturas del 16 de abril.
- Puntos impares como referencia.
- Puntos pares como evaluación.
- Ocho balizas disponibles.
- Canales 37, 38 y 39.
- Cuatro orientaciones por punto.
- Variabilidad temporal reducida.

La referencia y la evaluación ocupan posiciones distintas, pero proceden de la misma jornada. Estas condiciones resultan especialmente favorables para WKNN.

### Escenario 2: entrenamiento multijornada sin la baliza 5

Representa un caso más próximo a la realidad:

- Referencia con puntos impares de los días 9, 10, 13, 16, 17, 20 y 22 de abril.
- Evaluación con puntos pares del 16 de abril.
- Eliminación de la baliza 5 en entrenamiento y evaluación.
- Siete balizas disponibles.
- Canales 37, 38 y 39.
- Variabilidad procedente de distintas jornadas.

El experimento simula un sistema que se vuelve a entrenar después de que una baliza deje de estar disponible.

## Tratamiento de orientaciones

Las configuraciones aceptan el formato antiguo Rotation para una orientación y el formato Rotations para varias orientaciones.

Los métodos neuronales conservan las huellas reales de todas las orientaciones de referencia. Durante la evaluación se genera una predicción por captura válida y las predicciones del mismo punto se fusionan mediante la mediana de cada coordenada. Las métricas finales se calculan por punto físico, no por orientación individual.

## Métricas

Se calcula el error euclídeo tridimensional entre la posición real y la estimada.

| Métrica | Interpretación |
|---|---|
| Media | Error promedio |
| Mediana | Error por debajo del cual queda el 50 % de las estimaciones |
| P90 | Error por debajo del cual queda el 90 % de las estimaciones |
| Máximo | Peor error registrado |

Todos los valores se expresan en metros y un valor menor indica mayor exactitud. La vista denominada precisión representa la distribución acumulada del error posicional, no la precisión propia de clasificación.

## Resultados principales

Las tablas utilizan una configuración completa de baja media para cada método, sin mezclar métricas de configuraciones WKNN diferentes.

### Jornada única con todas las balizas

| Método | Media | Mediana | P90 | Máximo |
|---|---:|---:|---:|---:|
| WKNN, euclídea, canal 37, k=2 | 1,06 m | 1,00 m | 1,62 m | 2,27 m |
| MLP Scikit-learn, canal 39 | 0,98 m | 0,90 m | **1,56 m** | 2,39 m |
| Residual PyTorch, canal 39 | 1,09 m | 0,94 m | 1,63 m | 2,31 m |
| Híbrido, canales 37+38+39 | **0,87 m** | **0,67 m** | 1,79 m | **2,24 m** |

WKNN continúa siendo competitivo. El híbrido obtiene la mejor media y mediana, mientras que el MLP consigue el mejor P90.

### Entrenamiento multijornada sin la baliza 5

| Método | Media | Mediana | P90 | Máximo |
|---|---:|---:|---:|---:|
| WKNN, Chebyshev, canal 39, k=4 | 1,52 m | 1,43 m | 2,90 m | 3,28 m |
| MLP Scikit-learn, canal 39 | 1,10 m | 1,03 m | 1,75 m | 2,49 m |
| Residual PyTorch, canal 39 | 1,07 m | 0,92 m | 1,67 m | 2,56 m |
| Híbrido, canales 37+38+39 | **0,85 m** | **0,73 m** | **1,50 m** | **1,64 m** |

Los tres métodos neuronales superan a WKNN. El híbrido reduce aproximadamente a la mitad los errores de WKNN.

La diferencia entre escenarios representa el efecto conjunto de la variabilidad multijornada y la eliminación de la baliza 5. Para aislar ambos factores serían necesarios experimentos adicionales de ablación.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| IPSCM/ | Backend Flask, procesamiento y métodos de posicionamiento |
| IPSCM/services/ | WKNN, modelos neuronales, fusión y tratamiento de campañas |
| IPSCM/controllers/ | Consultas y persistencia |
| IPSCM/tests/ | Pruebas unitarias del backend |
| IPSCM/experiments/ | Preparación reproducible de conjuntos de datos |
| IPSCM.WebClient/ | Interfaz Angular y gráficas |
| Campañas/ | Bases de datos locales, excluidas de Git |
| Resultados/ | Resultados experimentales locales |
| docker-compose.yaml | Contenedores del backend y frontend |

## Preparación del escenario multijornada

El script IPSCM/experiments/prepare_multicampaign_dataset.py:

1. Usa la cuadrícula del 16 de abril como referencia.
2. Normaliza las coordenadas de las jornadas.
3. Selecciona puntos impares para entrenamiento.
4. Selecciona puntos pares del 16 de abril para evaluación.
5. Conserva las orientaciones 0, 90, 180 y 270.
6. Elimina la baliza 5.
7. Comprueba que no exista fuga espacial.
8. Genera un manifiesto y sumas SHA-256.

Las bases de datos originales y los resultados generados se mantienen fuera del repositorio mediante .gitignore.

## Requisitos y ejecución

Requisitos:

- Docker Desktop.
- Docker Compose.
- GPU compatible con CUDA opcional.

Construcción y arranque:

~~~bash
docker compose up --build
~~~

Servicios:

- Interfaz web: http://localhost:4200
- API: http://localhost:5000

## Flujo básico de uso

1. Importar una base de evaluación.
2. Importar una o varias bases de referencia.
3. Cargar la configuración general y las configuraciones de puntos.
4. Seleccionar una campaña.
5. Elegir WKNN, DeepLearning, DeepLearningPyTorch o DeepLearningHybrid.
6. Configurar protocolo, canal y límite de muestras.
7. Ejecutar el método.
8. Generar las gráficas y comparar las métricas.

La opción Sin límite utiliza todas las observaciones disponibles.

## Pruebas

~~~bash
docker compose build server
docker compose run --rm server python -m unittest discover -s tests -v
~~~

Las pruebas cubren la construcción de huellas, compatibilidad entre Rotation y Rotations, evaluación multiorientación, fusión por mediana, intersección de balizas, valores RSSI ausentes, reproducibilidad y cálculo común del error.

## Tecnologías

- Python, Flask, SQLAlchemy y SQLite.
- NumPy, Scikit-learn y PyTorch.
- Angular, TypeScript, Angular Material y Chart.js.
- Docker, Docker Compose, Git y GitHub.

## Conclusiones

- WKNN es una referencia sencilla y competitiva en condiciones homogéneas.
- Los modelos neuronales aprovechan mejor la diversidad multijornada.
- La red residual mantiene un rendimiento más estable ante el cambio de escenario.
- La fusión de los canales 37, 38 y 39 aumenta la robustez.
- El híbrido obtiene los mejores resultados cuando aumenta la variabilidad y se elimina una baliza.
- El tratamiento de orientaciones es esencial para realizar una comparación justa.

## Origen y atribución

Este repositorio amplía IPS Campaign Manager, desarrollado originalmente por [Simon Rusu](https://github.com/SimonRusu) como parte de su Trabajo de Fin de Grado.

La extensión de aprendizaje profundo, el procesamiento multijornada, la evaluación multiorientación, la fusión multicanal y los experimentos comparativos de este TFG han sido desarrollados por [juanfradelrm](https://github.com/juanfradelrm).

Se conserva el historial de Git para reconocer el trabajo previo sobre el que se construye esta investigación.

## Licencia

El proyecto original no incluía un archivo de licencia en la raíz cuando se creó esta extensión. Los derechos pertenecen a sus respectivos autores. Se recomienda contactar con ellos antes de reutilizar o redistribuir el código fuera de las condiciones permitidas por GitHub.