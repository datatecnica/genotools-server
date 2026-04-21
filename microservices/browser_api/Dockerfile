FROM python:3.11-slim


WORKDIR /app

COPY requirements.txt app/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r app/requirements.txt

# RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]