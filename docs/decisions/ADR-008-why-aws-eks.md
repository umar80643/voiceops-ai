# ADR-008 — Why AWS EKS

## Context
Need a managed Kubernetes control plane, integrated IAM, and mature GPU instance availability for ASR inference.

## Decision
AWS EKS + RDS (Postgres) + S3 + ECR + CloudWatch, provisioned via Terraform (`infra/terraform/main.tf`).

## Alternatives considered
- GCP GKE — a comparable managed Kubernetes offering; not chosen here, largely because AWS had broader GPU instance (g5.*) availability at the time this was designed.
- Self-managed kubeadm cluster — more operational burden, no managed RDS/S3 integration.

## Trade-offs
EKS + RDS + S3 gives a managed control plane, managed Postgres with automated backups, and durable object storage with minimal operational overhead, at AWS's usual cost premium over self-hosting.

## Consequences
`infra/terraform/main.tf` is real, reviewable HCL (VPC, EKS with a tainted GPU node group, RDS, S3, ECR) but has NOT been run — no AWS account/credentials were available in this sandbox. `terraform plan`/`apply` has not been executed against real AWS infrastructure; do not treat this as validated, only as a reviewed design.
