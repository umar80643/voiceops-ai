# Deployment

## Local (verified working)

```bash
make install
cp .env.example .env
make run          # uvicorn on :8000, single dev worker
```

Visit `http://localhost:8000/docs` for interactive API docs, or
`http://localhost:8000/demo` for the demo UI.

## Docker Compose (written, not build-tested in this sandbox)

```bash
make docker-up
make docker-down
```

`docker-compose.yml` provisions the API plus Postgres, Redis, Kafka,
Qdrant, MinIO, Prometheus, and Grafana. **Not build-tested here** — no
Docker daemon was available in the sandbox this was built in. The API
Dockerfile (`infra/docker/api.Dockerfile`) follows standard multi-stage,
non-root, healthcheck-equipped conventions; validate with `docker build`
and `docker compose up` in an environment with Docker before trusting it
in production.

## Kubernetes (written, not validated)

```bash
kubectl apply -f infra/kubernetes/
```

or via Helm:

```bash
helm install voiceops infra/helm/voiceops -f infra/helm/voiceops/values-dev.yaml
```

**Not validated against a live cluster** — no `kubectl`/cluster access
in this sandbox. Run `kubectl apply --dry-run=client -f infra/kubernetes/`
and `helm lint infra/helm/voiceops` at minimum before applying to a real
cluster, and confirm GPU scheduling (`nvidia.com/gpu` resource requests
in `asr-deployment.yaml`) against your actual node pool's device plugin.

## AWS (Terraform written, not run)

```bash
cd infra/terraform
terraform init
terraform plan   # NOT run in this sandbox — no AWS credentials available
terraform apply
```

Provisions: VPC, EKS (general + tainted GPU node group), RDS Postgres,
S3 bucket, ECR repository. See ADR-008 for the reasoning and explicit
caveat that this has not been executed against real AWS infrastructure.

## Rollback / operational notes

- The API is stateless aside from the SQLite dev DB (`voiceops_dev.db`)
  and in-memory RAG index — in production (Postgres + persistent Qdrant),
  rolling back a bad deploy is a standard Kubernetes rollout undo; no
  special migration step is implemented here yet (no Alembic migrations
  exist — `Base.metadata.create_all()` is used for simplicity in this
  build, which is not a production-safe migration strategy).
