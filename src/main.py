import os
import datetime
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from flask import Flask, request, g
from io import StringIO

app_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_spark():
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


app = Flask(__name__)

model = None


@app.route("/health")
def health():
    return f"Running on {g.spark.version}"


@app.route("/load/<job_id>", methods=["POST"])
def load(job_id: str):
    global model
    model = PipelineModel.load(f"s3a://diastemamodels/{job_id}")

    return "Loaded model successfully"


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded", 500

    raw = request.data.decode("utf-8")
    df = pd.read_csv(StringIO(raw))
    data = g.spark.createDataFrame(df)
    pred = model.transform(data)
    out = pred.select(*[c for c in pred.columns if (c in data.columns and c != "features") or c == "prediction"])

    return out.toPandas().to_csv(index=False)


@app.before_first_request
def before_first_request():
    g.spark = get_spark()


@app.before_request
def before_request():
    if "spark" not in g:
        g.spark = get_spark()


@app.errorhandler(404)
def page_not_found(_):
    return "Not found", 404


@app.errorhandler(Exception)
def error(ex: Exception):
    g.pop("spark", None)
    return f"{ex}", 500


if __name__ == "__main__":
    app.run("0.0.0.0", 5000, debug=False)
