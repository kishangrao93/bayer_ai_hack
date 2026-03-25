# =============================================================================
# Bayer AI Incident Response System — Lambda Container Image
# =============================================================================
#
# Build:
#   docker build -t bayer-incident-rca:latest .
#
# Test locally (requires AWS credentials in environment):
#   docker run --rm \
#     --env-file .env \
#     -p 9000:8080 \
#     bayer-incident-rca:latest
#
#   # In a second terminal, send a test CloudWatch alarm event:
#   curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
#     -d @tests/sample_alarm_event.json
#
# Push to ECR and deploy:
#   See deploy.sh for the full push + Lambda update workflow.
# =============================================================================

# AWS-managed Lambda Python 3.11 base image.
# Always pin to a specific tag so builds are reproducible.
FROM public.ecr.aws/lambda/python:3.11

# ---------------------------------------------------------------------------
# System dependencies
# libgomp (OpenMP) is needed by numpy/scipy at runtime.
# The Lambda base image (Amazon Linux 2) uses yum — NOT dnf (that is AL2023).
# ---------------------------------------------------------------------------
RUN yum install -y libgomp && \
    yum clean all && \
    rm -rf /var/cache/yum

# ---------------------------------------------------------------------------
# Python dependencies
# requirements-lambda.txt is the Lambda-specific subset of requirements.txt:
#   - sentence-transformers and rank_bm25 are EXCLUDED (not used at runtime;
#     they pull in scikit-learn → numpy>=2 which requires GCC>=9.3 to build
#     from source — the AL2 image only ships GCC 7.3.1).
#   - numpy is pinned to <2.0 so pip resolves a pre-built manylinux wheel
#     instead of compiling from source.
# --only-binary=numpy is an extra safety net that makes pip refuse to build
# numpy from source even if a source dist somehow wins dependency resolution.
# ---------------------------------------------------------------------------
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir \
        --only-binary=numpy,tiktoken \
        -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY lambda_handler.py   ${LAMBDA_TASK_ROOT}/
COPY orchestrator.py     ${LAMBDA_TASK_ROOT}/
COPY sub_agents.py       ${LAMBDA_TASK_ROOT}/
COPY tools.py            ${LAMBDA_TASK_ROOT}/
COPY rca_generator.py    ${LAMBDA_TASK_ROOT}/
COPY orch_2.py           ${LAMBDA_TASK_ROOT}/

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

# Log level — override at Lambda function level via environment variables.
ENV LOG_LEVEL=INFO

# LangChain / LangSmith — can be overridden at deploy time.
# Do NOT hard-code secrets here; pass them as Lambda environment variables.
# ENV LANGCHAIN_VERBOSE=false

# ---------------------------------------------------------------------------
# Lambda handler entrypoint
# Format: <module_name>.<function_name>
# ---------------------------------------------------------------------------
CMD ["lambda_handler.handler"]
