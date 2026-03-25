#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Build, push, and deploy the Lambda container image
# =============================================================================
#
# Prerequisites:
#   - AWS CLI v2 installed and configured (aws configure)
#   - Docker Desktop running
#   - Sufficient IAM permissions:
#       ecr:GetAuthorizationToken, ecr:CreateRepository,
#       ecr:BatchCheckLayerAvailability, ecr:PutImage,
#       lambda:CreateFunction, lambda:UpdateFunctionCode,
#       lambda:UpdateFunctionConfiguration, iam:PassRole
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Override defaults by exporting variables before running:
#   AWS_REGION=eu-west-1 LAMBDA_FUNCTION_NAME=my-fn ./deploy.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these or export them before calling the script
# ---------------------------------------------------------------------------

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"

ECR_REPO_NAME="${ECR_REPO_NAME:-bayer-incident-rca}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-bayer-incident-rca}"
LAMBDA_ROLE_ARN="${LAMBDA_ROLE_ARN:-}"           # Required for first-time create
LAMBDA_TIMEOUT="${LAMBDA_TIMEOUT:-900}"          # 15 minutes (max for Lambda)
LAMBDA_MEMORY="${LAMBDA_MEMORY:-2048}"           # MB — sentence-transformers needs headroom

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Step 1 — Ensure ECR repository exists
# ---------------------------------------------------------------------------
log "Ensuring ECR repository '${ECR_REPO_NAME}' exists..."
aws ecr describe-repositories \
    --repository-names "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" > /dev/null 2>&1 \
  || aws ecr create-repository \
        --repository-name "${ECR_REPO_NAME}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability MUTABLE \
        --query 'repository.repositoryUri' \
        --output text
log "ECR repository ready: ${ECR_URI}"

# ---------------------------------------------------------------------------
# Step 2 — Authenticate Docker to ECR
# ---------------------------------------------------------------------------
log "Authenticating Docker to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login \
      --username AWS \
      --password-stdin \
      "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ---------------------------------------------------------------------------
# Step 3 — Build the Docker image
# ---------------------------------------------------------------------------
log "Building Docker image..."
docker build \
  --platform linux/amd64 \
  --tag "${ECR_REPO_NAME}:${IMAGE_TAG}" \
  --tag "${ECR_URI}" \
  .

# ---------------------------------------------------------------------------
# Step 4 — Push to ECR
# ---------------------------------------------------------------------------
log "Pushing image to ECR: ${ECR_URI}"
docker push "${ECR_URI}"

# ---------------------------------------------------------------------------
# Step 5 — Create or update the Lambda function
# ---------------------------------------------------------------------------
FUNCTION_EXISTS=$(aws lambda get-function \
  --function-name "${LAMBDA_FUNCTION_NAME}" \
  --region "${AWS_REGION}" \
  --query 'Configuration.FunctionName' \
  --output text 2>/dev/null || echo "NOT_FOUND")

if [ "${FUNCTION_EXISTS}" = "NOT_FOUND" ]; then
  log "Lambda function does not exist — creating..."
  [ -z "${LAMBDA_ROLE_ARN}" ] && die "LAMBDA_ROLE_ARN must be set when creating a new function."

  aws lambda create-function \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --region "${AWS_REGION}" \
    --package-type Image \
    --code "ImageUri=${ECR_URI}" \
    --role "${LAMBDA_ROLE_ARN}" \
    --timeout "${LAMBDA_TIMEOUT}" \
    --memory-size "${LAMBDA_MEMORY}" \
    --architectures x86_64 \
    --environment "Variables={
      AWS_REGION=${AWS_REGION},
      LOG_LEVEL=INFO,
      TOKENIZERS_PARALLELISM=false
    }"

  log "Lambda function created: ${LAMBDA_FUNCTION_NAME}"
else
  log "Lambda function exists — updating image..."
  aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --region "${AWS_REGION}" \
    --image-uri "${ECR_URI}"

  log "Waiting for update to complete..."
  aws lambda wait function-updated \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --region "${AWS_REGION}"

  log "Lambda function code updated: ${LAMBDA_FUNCTION_NAME}"
fi

# ---------------------------------------------------------------------------
# Step 6 — Update function configuration (timeout, memory, env vars)
#          Set your real env vars here or manage them separately in the console.
# ---------------------------------------------------------------------------
log "Updating Lambda configuration..."
aws lambda update-function-configuration \
  --function-name "${LAMBDA_FUNCTION_NAME}" \
  --region "${AWS_REGION}" \
  --timeout "${LAMBDA_TIMEOUT}" \
  --memory-size "${LAMBDA_MEMORY}" \
  --environment "Variables={
    AWS_REGION=${AWS_REGION},
    BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID:-anthropic.claude-3-5-sonnet-20241022-v2:0},
    BEDROCK_KB_ID=${BEDROCK_KB_ID:-},
    KB_CONFIDENCE_THRESHOLD=${KB_CONFIDENCE_THRESHOLD:-0.6},
    RCA_S3_BUCKET=${RCA_S3_BUCKET:-},
    SES_SENDER=${SES_SENDER:-},
    ALERT_EMAIL_RECIPIENT=${ALERT_EMAIL_RECIPIENT:-},
    LANGSMITH_API_KEY=${LANGSMITH_API_KEY:-},
    LANGSMITH_PROJECT=${LANGSMITH_PROJECT:-bayer-incident-rca},
    LOG_LEVEL=${LOG_LEVEL:-INFO},
    TOKENIZERS_PARALLELISM=false
  }"

aws lambda wait function-updated \
  --function-name "${LAMBDA_FUNCTION_NAME}" \
  --region "${AWS_REGION}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "============================================================"
log "Deployment complete!"
log "  Function : ${LAMBDA_FUNCTION_NAME}"
log "  Region   : ${AWS_REGION}"
log "  Image    : ${ECR_URI}"
log "============================================================"
log "Next steps:"
log "  1. Set all remaining env vars in the Lambda console or re-run"
log "     this script after exporting the missing variables."
log "  2. Attach the Lambda to an SNS topic that your CloudWatch"
log "     alarm publishes to."
log "  3. Verify the Lambda execution role has the IAM policies"
log "     listed in README — Bedrock, CloudWatch Logs, S3, SES,"
log "     CodeDeploy read access."
