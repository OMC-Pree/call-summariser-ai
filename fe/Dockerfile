# fe/Dockerfile (ARM64)
# syntax=docker/dockerfile:1

FROM --platform=linux/arm64 public.ecr.aws/lambda/python:3.11

# Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 \
     /lambda-adapter /opt/extensions/lambda-adapter

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /var/task

# Install deps (pin gradio to latest v4)
COPY fe/requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --only-binary=:all: -r requirements.txt -v

# Copy app code
COPY fe/ ./

# Env for Gradio + Web Adapter
ENV AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/ \
    AWS_LWA_LOG_LEVEL=info \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080 \
    GRADIO_ANALYTICS_ENABLED=false \
    PORT=8080 \
    MPLCONFIGDIR=/tmp/mpl

# Start via Lambda handler
CMD ["server.handler"]
