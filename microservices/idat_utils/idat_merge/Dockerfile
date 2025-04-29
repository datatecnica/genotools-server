# Use Python 3.11 base image
FROM python:3.11-slim

# Set environment variables to take away user interaction in install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/datatecnica/genotools-server.git /opt/genotools-server

# Set working directory
WORKDIR /opt/genotools-server/microservices/idat_utils

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy and make entrypoint script executable
COPY docker-setup.sh /opt/genotools-server/microservices/idat_utils/docker-setup.sh
RUN chmod +x /opt/genotools-server/microservices/idat_utils/docker-setup.sh
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/i/icu/libicu70_70.1-2_amd64.deb
RUN dpkg -i libicu70_70.1-2_amd64.deb

# Set default shell entry point
# ENTRYPOINT ["/bin/sh"]
# Set default entrypoint
ENTRYPOINT ["/opt/genotools-server/microservices/idat_utils/docker-setup.sh"]