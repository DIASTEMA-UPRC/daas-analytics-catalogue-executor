FROM konvoulgaris/diastema-spark-base
RUN apt update -y
RUN apt upgrade -y
RUN pip install -U pip
WORKDIR /app
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gunicorn
ADD src/ .
EXPOSE 5000
