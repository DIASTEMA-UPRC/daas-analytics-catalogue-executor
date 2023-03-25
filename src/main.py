import datetime
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from flask import Flask, request, g
from werkzeug.serving import make_server
from io import StringIO
from pymongo import MongoClient

app_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class Model:
    def __init__(self, pipeline: PipelineModel, job_id: str):
        self.pipeline = pipeline
        self.job_id = job_id


def get_spark() -> SparkSession:
    session = SparkSession.builder.appName(f"diastema-{app_id}").getOrCreate()
    ctx = session.sparkContext
    host = session.conf.get("spark.executorEnv.MINIO_HOST", "0.0.0.0")
    port = session.conf.get("spark.executorEnv.MINIO_PORT", "9000")
    username = session.conf.get("spark.executorEnv.MINIO_USER", "minioadmin")
    password = session.conf.get("spark.executorEnv.MINIO_PASS", "minioadmin")
    ctx._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    ctx._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "false")
    ctx._jsc.hadoopConfiguration().set("fs.s3a.endpoint", f"http://{host}:{port}")
    ctx._jsc.hadoopConfiguration().set("fs.s3a.access.key", username)
    ctx._jsc.hadoopConfiguration().set("fs.s3a.secret.key", password)
    ctx._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")

    return session


def get_db(session: SparkSession) -> MongoClient:
    mongo_host = session.conf.get("spark.executorEnv.MONGO_HOST", "0.0.0.0")
    mongo_port = session.conf.get("spark.executorEnv.MONGO_PORT", "27017")

    return MongoClient(f"mongodb://{mongo_host}:{mongo_port}")


app = Flask(__name__)

model = None


@app.route("/health")
def health():
    return f"Running on {g.spark.version}"


@app.route("/start/<job_id>", methods=["GET"])
def start(job_id: str):
    global model
    pipeline = PipelineModel.load(f"s3a://diastemamodels/{job_id}")
    model = Model(pipeline, job_id)

    return "Loaded model successfully"


@app.route("/stop/<job_id>", methods=["GET"])
def stop(job_id: str):
    global model
    model = None
    g.pop("db", None)
    g.spark.stop()
    g.pop("spark", None)

    exit(0)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded", 500

    match = g.db["Diastema"]["Analytics"].find_one({ "job_id": model.job_id })

    if not match or not "columns" in match:
        return "Missing metadata for job-id", 500

    columns = match["columns"]
    raw = request.data.decode("utf-8")
    df_columns = [c for c in columns if c != "prediction"]
    df = pd.read_csv(StringIO(raw), header=None, names=df_columns)
    data = g.spark.createDataFrame(df)
    pred = model.pipeline.transform(data)
    out = pred.select(*columns)

    return out.toPandas().to_csv(index=False)


@app.before_first_request
def before_first_request():
    g.spark = get_spark()


@app.before_request
def before_request():
    if "spark" not in g:
        g.spark = get_spark()

    if "db" not in g:
        g.db = get_db(g.spark)


@app.errorhandler(404)
def page_not_found(_):
    return "Not found", 404


@app.errorhandler(Exception)
def error(ex: Exception):
    g.spark.stop()
    g.pop("db", None)
    g.pop("spark", None)

    return str(ex), 500


if __name__ == "__main__":
    spark = get_spark()
    host = spark.conf.get("spark.executorEnv.FLASK_HOST", "0.0.0.0")
    port = spark.conf.get("spark.executorEnv.FLASK_PORT", "5000")
    server = make_server(host, port, app)
    server.serve_forever()
