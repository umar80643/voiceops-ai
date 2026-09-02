# Load Test Results (Phase 25)

Real run, recorded 2026-08-28, against a single local `uvicorn` dev worker
(no gunicorn/multi-worker, no Kubernetes autoscaling — this is the naive
baseline, not a production configuration):

```
locust -f scripts/loadtest.py --host http://localhost:8000 \
    --users 20 --spawn-rate 5 --run-time 20s --headless
```

| Endpoint | Requests | Failures | P50 | P95 | P99 | Max | RPS |
|---|---|---|---|---|---|---|---|
| POST /conversations | 106 | 4 (3.8%) | 14ms | 56ms | 59ms | 87ms | 5.4 |
| GET /health | 45 | 6 (13.3%) | 3ms | 6ms | 19ms | 19ms | 2.3 |
| POST /knowledge/search | 37 | 0 | 4ms | 5-13ms | — | 13ms | 1.9 |

**Bottleneck observed:** connection-refused errors (10/188 = 5.3%) under
even 20 concurrent users on a single dev-mode uvicorn process backed by a
single SQLite file. This is expected and NOT a production configuration —
production would run multiple uvicorn/gunicorn workers behind Kubernetes
HPA (infra/kubernetes/) against Postgres with a connection pool, which
this test does not exercise. 50/100-user runs were not performed in this
sandbox given the above known single-worker ceiling; re-run this script
against a multi-worker/Postgres deployment before drawing production
conclusions.
