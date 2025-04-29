FROM python:3.13-slim

RUN pip install -U pip 

WORKDIR /app

COPY [ "requirements.txt", "./" ]

RUN pip install -r requirements.txt

COPY [ "reddit_model_pipeline.joblib", "redditApp.py", "./"]

EXPOSE 8000

# --host=0.0.0.0 is necessary when using uvicorn inside a docker container so that you can connect to localhost or 127.0.0.1 from outside of container
ENTRYPOINT [ "uvicorn", "redditApp:app", "--host", "0.0.0.0"]