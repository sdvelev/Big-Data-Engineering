import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
from pymongo import MongoClient

def load_csv_to_mongo():
    df = pd.read_csv("/opt/airflow/data/employers.csv")

    df['salary'] = df['salary'].astype(int)

    records = df.to_dict(orient='records')

    # Connect to MongoDB
    client = MongoClient("mongodb://mongo:27017/")
    db = client["company"]
    collection = db["employers"]

    collection.delete_many({})

    collection.insert_many(records)
    print("Data is inserted into MongoDB")

with DAG(
    dag_id="load_csv_to_mongodb",
    start_date=datetime(2025, 4, 1),
    schedule_interval="* * * * *",
    catchup=False,
    tags=["mongo", "csv"],
) as dag:

    load_task = PythonOperator(
        task_id="load_csv",
        python_callable=load_csv_to_mongo
    )

    trigger_process_dag = TriggerDagRunOperator(
        task_id="trigger_process_dag",
        trigger_dag_id="process_data_from_mongodb"
    )

    load_task >> trigger_process_dag
