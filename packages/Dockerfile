FROM python:3.7-slim-buster
MAINTAINER Anuj Sable

COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

RUN pip install pytest && pytest tests/test_api_client.py

EXPOSE 5000

CMD ["bash", "./run.sh"]
