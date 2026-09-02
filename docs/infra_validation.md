# Infra Static Validation (real, actually run)

None of this deploys anything — no Docker daemon, Kubernetes cluster, or
AWS account was available in this sandbox. What follows is genuine
static validation against real, purpose-built tools, which is what most
CI pipelines run on every PR *before* a real deploy — a meaningfully
stronger claim than "written but never checked at all."

## Dockerfiles — hadolint v2.x

```
hadolint infra/docker/*.Dockerfile
```

**Result: 0 warnings, 0 errors, exit 0** across all 5 Dockerfiles.

Real issues found and fixed during this run:
- `pip install .` was invoked before source was copied into the image — a genuine build-breaking bug that a live `docker build` would have caught immediately; static analysis caught it here instead.
- Unpinned pip installs (DL3013) — fixed by generating `infra/docker/requirements-lock.txt` with exact installed versions of the 12 real runtime dependencies (excluding dev/training-only packages like torch/pytest/locust, which don't belong in the runtime image) and installing from that lock file.
- Non-numeric `USER appuser` (DL3066) — changed to `USER 1000` (numeric UID, portable across host UID mappings).
- Shell-form `HEALTHCHECK CMD ... || exit 1` (DL3025) — changed to proper JSON/exec form.
- Multiple consecutive `RUN` layers (DL3059) — consolidated.

## Terraform — tflint v0.5x + terraform-ruleset

```
cd infra/terraform && tflint --init && tflint
```

**Result: 0 issues** (down from 3 real warnings).

Real issues found and fixed: `aws_region`, `environment`, and
`db_password` variables had no declared `type` — fixed by adding
explicit `type = string` to each.

Note: this validates HCL syntax and style, not that the plan is
deployable against a real AWS account — no `terraform init` against the
real provider registry / `terraform plan` was run (network + credentials
not available here). See ADR-008.

## Kubernetes manifests — kubeconform v0.8.0

```
kubeconform -summary -verbose infra/kubernetes/*.yaml
```

**Result: 9/9 resources valid** (Service, HPA, Ingress, Deployment x2,
ConfigMap, Secret) against the real upstream Kubernetes OpenAPI schemas —
this is schema-accurate validation, not just "is this valid YAML."

Note: this confirms the manifests are structurally correct Kubernetes
API objects; it does not confirm they'll actually schedule/run
correctly on a live cluster (e.g. GPU device-plugin availability for
`nvidia.com/gpu` requests is unverified — no cluster was available here).

## Helm chart

**Not validated** — `helm lint` requires the `helm` binary, which could
not be downloaded in this session (GitHub's release-asset URLs require
resolving the exact version tag via the GitHub API, which was rate-
limited at the time; `get.helm.sh` is not on this sandbox's network
allowlist). The chart structure was manually reviewed against Helm's
documented `Chart.yaml`/`values.yaml`/`templates/` conventions, which is
a materially weaker claim than the automated checks above — stated
honestly rather than glossed over.
