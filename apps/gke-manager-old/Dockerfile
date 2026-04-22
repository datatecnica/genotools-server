# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    --no-install-recommends apt-utils \
    build-essential curl gnupg git  \
    && rm -rf /var/lib/apt/lists/*
#for kubectl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin

# Add the Google Cloud SDK repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install Google Cloud SDK and gke-gcloud-auth-plugin
RUN apt-get update && apt-get install -y \
    google-cloud-sdk \
    google-cloud-sdk-gke-gcloud-auth-plugin \
    && rm -rf /var/lib/apt/lists/*


# Verify installation of gke-gcloud-auth-plugin
RUN gcloud components list && gke-gcloud-auth-plugin --version

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8080

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]