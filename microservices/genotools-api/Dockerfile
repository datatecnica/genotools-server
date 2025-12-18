FROM python:3.11-slim
LABEL maintainer="Dan Vitale <dan@datatecnica.com>"

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN pip install poetry

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --verbose \
    && apt-get remove -y gcc build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y wget
# for KING
RUN wget https://www.kingrelatedness.com/Linux-king.tar.gz -O /tmp/Linux-king.tar.gz

# Create a directory for KING and extract the executable
RUN mkdir -p /root/.genotools/misc/executables /opt/king && \
    tar -xzvf /tmp/Linux-king.tar.gz -C /root/.genotools/misc/executables

# Add KING to the system's PATH
ENV PATH="/root/.genotools/misc/executables:${PATH}"

COPY . /app
EXPOSE 8080

CMD ["uvicorn", "genotools_api.main:app", "--host", "0.0.0.0", "--port", "8080"]