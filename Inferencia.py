from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType
from pyspark.sql.functions import from_json, col, when
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, when



# Iniciamos la sesión de psark
spark = SparkSession.builder \
    .appName("ProyectoVuelosStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

model = PipelineModel.load("modelo_clasificacion_vuelos")

# Definimos el esquema
esquema_vuelos = StructType([
    StructField("FlightDate", StringType(), True),
    StructField("Airline", StringType(), True),
    StructField("Marketing_Airline_Network", StringType(), True),
    StructField("Origin", StringType(), True),
    StructField("Dest", StringType(), True),
    StructField("CRSDepTime", LongType(), True),
    StructField("DayOfWeek", IntegerType(), True),
    StructField("DayofMonth", IntegerType(), True),
    StructField("Month", IntegerType(), True),
    StructField("Distance", DoubleType(), True),
    StructField("Cancelled", StringType(), True),
    StructField("Diverted", StringType(), True),
    StructField("DepDel15", DoubleType(), True)
])
# Leemos el stream
df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "vuelos_topic") \
    .option("startingOffsets", "latest") \
    .load()

vuelos_base = df_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), esquema_vuelos).alias("data")) \
    .select("data.*")

vuelos_features = vuelos_base.withColumn("DepHour", (col("CRSDepTime") / 100).cast("int")) \
                             .withColumn("IsWeekend", when(col("DayOfWeek") >= 6, 1).otherwise(0)) \
                             .withColumn("Cancelled", col("Cancelled").cast("boolean")) \
                             .withColumn("Diverted", col("Diverted").cast("boolean"))


predicciones = model.transform(vuelos_features)

predicciones_ajustadas = predicciones.withColumn(
    "Probabilidad_Retraso_Pura", 
    vector_to_array(col("probability")).getItem(1) 
).withColumn(
    "Prevision_Retraso_Ajustada",
    when(col("Probabilidad_Retraso_Pura") >= 0.35, 1.0).otherwise(0.0) 
)


# Clasificamos la preidicción
predicciones_auditadas = predicciones_ajustadas.withColumn(
    "Resultado",
    when(col("DepDel15").isNull(), "Desconocido") \
    .when((col("Prevision_Retraso_Ajustada").cast("int") == 1) & (col("DepDel15").cast("int") == 1), "ACIERTO") \
    .when((col("Prevision_Retraso_Ajustada").cast("int") == 1) & (col("DepDel15").cast("int") == 0), "FALLO") \
    .when((col("Prevision_Retraso_Ajustada").cast("int") == 0) & (col("DepDel15").cast("int") == 0), "ACIERTO") \
    .when((col("Prevision_Retraso_Ajustada").cast("int") == 0) & (col("DepDel15").cast("int") == 1), "FALLO") \
    .otherwise("Desconocido")
)

# Salida por consola
query = predicciones_auditadas.select(
        "FlightDate", "Airline", "Origin", "Dest", "DepHour", 
        "Prevision_Retraso_Ajustada", "DepDel15", "Resultado"
    ) \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

print("Esperando datos de Kafka...")
query.awaitTermination()
