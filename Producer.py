import polars as pl
import json
import time
from kafka import KafkaProducer

########################## 
HORAS_ANTELACION = 2
FACTOR_VELOCIDAD = 360
ARCHIVO = "vuelos_2022.parquet"
TOPIC = "vuelos_topic"
##########################

# Configurar Productor de Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8')
)

print("Iniciando programa")
fechas_unicas = (
    pl.read_parquet(ARCHIVO, columns=["FlightDate"])
    .unique()
    .sort("FlightDate")
    .to_series()
    .to_list()
)

#Para saber cuanto esperar entre envios 
last_send_time = None   

#Main 
for fecha in fechas_unicas:
    print(f"Cargando y procesando el día: {fecha}")
    
    #Para trabajar con un solo dia 
    lf_dia = pl.scan_parquet(ARCHIVO).filter(pl.col("FlightDate") == fecha)

    #Eliminamos lascolumnas que no sabemso todavia,quitado la objetivo para enviarla y validar
    columnas_futuras = [
        'Cancelled', 'Diverted', 'DepTime', 'DepDelayMinutes', 'DepDelay', 
        'ArrTime', 'ArrDelayMinutes', 'ArrDelay', 'AirTime', 'ActualElapsedTime', 
         'DepartureDelayGroups', 'TaxiOut', 'WheelsOff', 'WheelsOn', 
        'TaxiIn', 'ArrDel15', 'ArrivalDelayGroups', 'DivAirportLandings'
    ]
    lf_dia = lf_dia.drop(columnas_futuras)
    
    # Transformaciones temporales
    time_str = pl.col("CRSDepTime").cast(pl.String).str.zfill(4)#POner 4 digitos todas fecha para formato 
    datetime_str = pl.col("FlightDate").dt.strftime("%Y-%m-%d") + " " + time_str#Juntar en un solo 
    
   #Se formatea la columna nueva se le resta las horas de antes y se ordena para saer cuando avisar 
    lf_procesado = (
        lf_dia
        .with_columns([
            datetime_str.str.to_datetime("%Y-%m-%d %H%M", strict=False).alias("Scheduled_Departure")
        ])
        .filter(pl.col("Scheduled_Departure").is_not_null())
        .with_columns([
            (pl.col("Scheduled_Departure") - pl.duration(hours=HORAS_ANTELACION)).alias("Send_Time")
        ])
        .sort("Send_Time") 
    )
    
    # Se coge en memoria y hacen todos las operaciones
    df_dia = lf_procesado.collect()
    vuelos_del_dia = len(df_dia)
    
    print(f"Preparados {vuelos_del_dia} vuelos.")
    
    
    #Envios 
    for index, row in enumerate(df_dia.iter_rows(named=True)):
        current_send_time = row['Send_Time']

        # Calculamos la pausa si procede
        if last_send_time is not None:
            diferencia_segundos = (current_send_time - last_send_time).total_seconds()
            tiempo_a_esperar = diferencia_segundos / FACTOR_VELOCIDAD
            
            if tiempo_a_esperar > 0:
                time.sleep(tiempo_a_esperar)
        
        # Eliminamos las columnas auxiliares antes de enviar
        del row['Scheduled_Departure']
        del row['Send_Time']
        
        producer.send(TOPIC, value=row)
        
        # Actualizamos la marca de tiempo
        last_send_time = current_send_time
        
        #Comprobacion
        if index == 0:
            print(f"[DEBUG] -> Primer envío del día {fecha} realizado.")
        elif index > 0 and index % 500 == 0:
            print(f"  -> Se han enviado {index}/{vuelos_del_dia} vuelos de este día...")
        elif index == vuelos_del_dia - 1:
            print(f"[DEBUG] -> Último envío del día {fecha} completado.")

    # Para nviar el final del dia aunque no sea un oaquete cmpleto de kafka
    producer.flush()

producer.close()

# Iniciar Kafka por consola
#KAFKA_CLUSTER_ID="$(bin/kafka-storage.sh random-uuid)" 
#bin/kafka-storage.sh format -t $KAFKA_CLUSTER_ID -c config/kraft/server.properties 
#bin/kafka-server-start.sh config/kraft/server.properties 
