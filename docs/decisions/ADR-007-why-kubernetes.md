# ADR-007 — Why Kubernetes

## Context
CPU-bound services (API, NLP) and GPU-bound services (ASR) need to scale independently, with rolling deploys and self-healing.

## Decision
Kubernetes (EKS in production, see ADR-008).

## Alternatives considered
- Docker Swarm — simpler, but a weaker ecosystem and autoscaling story.
- Plain ECS/Fargate — a reasonable AWS-native alternative; Kubernetes was chosen for portability and its more mature GPU node-pool/HPA support.

## Trade-offs
Kubernetes HPA plus separate node pools (general vs GPU, see `infra/terraform/main.tf`) map directly onto this system's uneven resource needs across services, at the cost of operational complexity relative to simpler PaaS options.

## Consequences
Manifests in `infra/kubernetes/` and the Helm chart in `infra/helm/` are real, reviewable YAML, but NOT validated against a live cluster in this sandbox — no `kubectl`/cluster access was available. Validate with `kind`/`minikube` or a real cluster (`kubectl apply --dry-run=client`, `helm lint`) before treating any of it as tested infrastructure.
