import gspread
from google.oauth2.service_account import Credentials
from pymongo import MongoClient
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
from datetime import datetime

def process_data_from_mongo():
    hook = MongoHook(conn_id="mongo_conn")
    client = hook.get_conn()
    db = client["company"]
    collection = db["employers"]

    data = list(collection.find())

    department_stats = {}
    for record in data:
        dept = record['department']
        salary = int(record['salary'])
        
        if dept not in department_stats:
            department_stats[dept] = {"total_salary": 0, "staff_count": 0}
        
        department_stats[dept]["total_salary"] += salary
        department_stats[dept]["staff_count"] += 1

    for dept in department_stats:
        total = department_stats[dept]["total_salary"]
        count = department_stats[dept]["staff_count"]
        avg = total / count
        department_stats[dept]["average_salary"] = avg

    print("Department-wise statistics:", department_stats)

    db["department_stats"].delete_many({})
    db["department_stats"].insert_many([
        {"department": k, **v} for k, v in department_stats.items()
    ])

with DAG(
    dag_id="process_data_from_mongodb",
    start_date=datetime(2025, 4, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mongo", "process"]
) as dag:

    start_task = DummyOperator(task_id="start_task")
    
    process_task = PythonOperator(
        task_id="process_data",
        python_callable=process_data_from_mongo
    )

    trigger_upload_dag = TriggerDagRunOperator(
        task_id="trigger_upload_dag",
        trigger_dag_id="mongo_upload_to_gsheet"
    )


    start_task >> process_task >> trigger_upload_dag
