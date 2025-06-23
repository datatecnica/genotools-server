FROM python:3.11-slim

EXPOSE 8080

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
