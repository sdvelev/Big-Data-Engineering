import gspread
import mlflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime
from google.oauth2.service_account import Credentials
from pymongo import MongoClient

def upload_stats_to_gsheet():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("BigDataEngineeringCoursework2")

    with mlflow.start_run(run_name="UploadStatsToGoogleSheets"):
        mlflow.log_param("source", "MongoDB")
        mlflow.log_param("destination", "Google Sheets")

        mongo_client = MongoClient("mongodb://mongo:27017/")
        db = mongo_client["company"]
        stats = list(db["department_stats"].find({}, {"_id": 0}))

        if not stats:
            print("No statistics can be found")
            return

        for stat in stats:
            department = stat["department"]
            avg_salary = stat.get("average_salary")
            count = stat.get("staff_count")

            if avg_salary is not None:
                mlflow.log_metric(f"{department}_avg_salary", avg_salary)
            if count is not None:
                mlflow.log_metric(f"{department}_staff_count", count)

        SERVICE_ACCOUNT_FILE = '/opt/airflow/secrets/bigdataengineeringcoursework2-6782d28a1616.json'
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

        credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        client = gspread.authorize(credentials)
        sheet = client.open("BigDataEngineeringCoursework2").sheet1

        sheet.clear()
        headers = list(stats[0].keys())
        data = [headers] + [[row.get(h, "") for h in headers] for row in stats]
        sheet.update('A1', data)

        print("Data uploaded to Google Sheets and logged to MLflow.")


with DAG(
    dag_id="mongo_upload_to_gsheet",
    start_date=datetime(2025, 4, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mongo", "upload", "gcp"]
) as dag:

    start = DummyOperator(task_id="start")

    upload = PythonOperator(
        task_id="upload_to_gsheet",
        python_callable=upload_stats_to_gsheet
    )

    start >> upload
