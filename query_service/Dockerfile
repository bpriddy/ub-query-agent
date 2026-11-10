# The query agent service (ub-query-agent#92). The build context is this
# folder (deployment/deploy-service.sh: gcloud run deploy --source query_service),
# so the image holds only the service: no ADK, no agent code.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements.txt query_service/requirements.txt
RUN pip install --no-cache-dir -r query_service/requirements.txt

COPY . query_service/

RUN useradd --uid 10001 --no-create-home svc
USER svc

CMD ["sh", "-c", "exec uvicorn query_service.app:app --host 0.0.0.0 --port ${PORT:-8080} --no-access-log"]
