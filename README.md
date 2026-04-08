# Predicción de retrasos de vuelos

## Problema que resuelve

El proyecto predice si un vuelo tendrá un retraso de salida de 15 minutos o más. Para ello, utiliza Apache Kafka para transmitir los vuelos simulados y Apache Spark para entrenar un modelo y realizar predicciones en tiempo real.

## Dataset

Se utiliza el dataset de retrasos de vuelos de Kaggle:

https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022

El dataset no se incluye en el repositorio porque ocupa demasiado espacio.

## Orden de ejecución

Ejecutar los scripts en este orden:

```text
Producer.py → EntrenarModelo.py → Inferencia.py
```

1. `Producer.py` envía los vuelos a Kafka.
2. `EntrenarModelo.py` entrena y guarda el modelo.
3. `Inferencia.py` recibe los vuelos y muestra las predicciones.

## Resultado principal

El sistema muestra para cada vuelo si se prevé un retraso y compara la predicción con el resultado real cuando está disponible.
