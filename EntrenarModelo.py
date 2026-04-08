from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml import PipelineModel


spark = SparkSession.builder \
    .appName("Entrenamiento_Vuelos_Eficiente") \
    .master("local[3]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.network.timeout", "1000s") \
    .config("spark.executor.heartbeatInterval", "100s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("--- Iniciando proceso de entrenamiento ligero ---")
df_2021 = spark.read.parquet("vuelos_2021.parquet")

df_clf = df_2021.filter((F.col("Cancelled") == False) & (F.col("Diverted") == False)) \
                .withColumn("label", F.col("DepDel15").cast("int")) \
                .dropna(subset=["label"])

# Realizamos undersampling
df_clf = df_clf.sampleBy("label", fractions={0: 0.38, 1: 1.0}, seed=42)

df_clf = df_clf.withColumn("DepHour", (F.col("CRSDepTime") / 100).cast("int")) \
               .withColumn("IsWeekend", F.when(F.col("DayOfWeek") >= 6, 1).otherwise(0))

train_data, test_data = df_clf.randomSplit([0.8, 0.2], seed=42)

# Pipeline
categorical_cols = ["Airline", "Origin", "Dest", "Marketing_Airline_Network"]
numeric_cols = ["DepHour", "DayOfWeek", "DayofMonth", "Month", "Distance", "IsWeekend"]

stages = []
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name+"_Index", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol=col_name+"_Index", outputCol=col_name+"_Vec")
    stages += [indexer, encoder]

assembler = VectorAssembler(
    inputCols=[c + "_Vec" for c in categorical_cols] + numeric_cols, 
    outputCol="features", 
    handleInvalid="keep"
)
stages.append(assembler)

rf = RandomForestClassifier(
    featuresCol="features", 
    labelCol="label", 
    numTrees=20,   
    maxDepth=6,     
    seed=42
)
stages.append(rf)

spark.catalog.clearCache()

print("Entrenando modelo... ")
pipeline = Pipeline(stages=stages)
model_eval = pipeline.fit(train_data)

model_eval.write().overwrite().save("modelo_clasificacion_vuelos")
print("Modelo guardado en: 'modelo_clasificacion_vuelos'")


#model_eval = PipelineModel.load("modelo_clasificacion_vuelos")


print("Calculando métricas finales...")
test_predictions = model_eval.transform(test_data)

evaluator_binary = BinaryClassificationEvaluator(labelCol="label")
evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

test_roc = evaluator_binary.evaluate(test_predictions, {evaluator_binary.metricName: "areaUnderROC"})
test_pr = evaluator_binary.evaluate(test_predictions, {evaluator_binary.metricName: "areaUnderPR"})
test_acc = evaluator_multi.evaluate(test_predictions, {evaluator_multi.metricName: "accuracy"})

print(f"\nMétricas de Rendimiento (TEST)")
print(f"Accuracy:      {test_acc:.4f}")
print(f"Area bajo ROC: {test_roc:.4f}")
print(f"Area bajo PR:  {test_pr:.4f}")

print("\nBuscando el Umbral Óptimo")

# Extraemos la probabilidad 
preds_prob = test_predictions.withColumn(
    "prob_1", 
    vector_to_array(F.col("probability")).getItem(1)
)
preds_prob.cache()

evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="pred_custom", metricName="f1")

best_threshold = 0.5
max_f1 = 0.0

umbrales = [x / 100.0 for x in range(20, 65, 5)]

print("Evaluando umbrales...")
for umbral in umbrales:
    preds_custom = preds_prob.withColumn(
        "pred_custom", 
        F.when(F.col("prob_1") >= umbral, 1.0).otherwise(0.0)
    )
    
    current_f1 = evaluator_f1.evaluate(preds_custom)
    
    print(f"  -> Umbral {umbral:.2f} | F1-Score: {current_f1:.4f}")
    
    if current_f1 > max_f1:
        max_f1 = current_f1
        best_threshold = umbral

print("--------------------------------------------------")
print(f"El umbral óptimo es: {best_threshold:.2f} (F1-Score: {max_f1:.4f})")
print("--------------------------------------------------")