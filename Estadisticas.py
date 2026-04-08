import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType
from pyspark.sql.functions import from_json, col, avg, count, round, date_format

spark = SparkSession.builder \
    .appName("VuelosStaticDashboard") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("OFF")

esquema_vuelos = StructType([
    StructField("FlightDate", StringType(), True),
    StructField("Airline", StringType(), True),
    StructField("Origin", StringType(), True),
    StructField("Dest", StringType(), True),
    StructField("DepDel15", DoubleType(), True)
])

df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "vuelos_topic") \
    .load()

vuelos = df_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), esquema_vuelos).alias("data")) \
    .select("data.*")

df_acumulado = None

def actualizar_dashboard(df_batch, batch_id):
    global df_acumulado
    if df_batch.count() == 0: return

    # Unir y cachear
    if df_acumulado is None:
        df_acumulado = df_batch.cache()
    else:
        df_acumulado = df_acumulado.union(df_batch).cache()

    # Usamos una cadena de texto para imprimir todo de una sola vez
    output = []
    output.append("\033[H\033[J") # Código para limpiar pantalla sin parpadeo
    output.append("="*60)
    output.append(f"REGISTROS: {df_acumulado.count()} | BATCH: {batch_id}")
    output.append("="*60 + "\n")
    
    res_dia = df_acumulado.withColumn("DiaSemana", date_format(col("FlightDate"), "EEEE")) \
        .groupBy("FlightDate", "DiaSemana") \
        .agg(
            count("*").alias("Vuelos"),
            round(avg("DepDel15") * 100, 2).alias("% Ret")
        ).orderBy("FlightDate").limit(10)

    res_air = df_acumulado.groupBy("Airline").agg(
        count("*").alias("Total"),
        round(avg("DepDel15") * 100, 2).alias("% Ret")
    ).filter(col("Total") > 5).orderBy(col("% Ret").desc()).limit(5)

    output.append("[ EVOLUCIÓN POR FECHA ]")
    output.append(res_dia._jdf.showString(10, 20, False))
    
    output.append("\n[ TOP 5 AEROLÍNEAS IMPUNTUALES ]")
    output.append(res_air._jdf.showString(5, 20, False))
    
    output.append("\n" + "-"*60)
    output.append(" Actualizando datos desde Kafka...")
    
    sys.stdout.write("\n".join(output))
    sys.stdout.flush()

query = vuelos.writeStream \
    .foreachBatch(actualizar_dashboard) \
    .trigger(processingTime='1 seconds') \
    .start()

query.awaitTermination()