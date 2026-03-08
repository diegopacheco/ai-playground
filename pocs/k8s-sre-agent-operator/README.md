# Kovalski: K8s SRE Agent Operator

Kubernetes SRE Agent Operator written in Rust 1.93+ (edition 2024) that runs inside a Kind cluster. It exposes REST endpoints to inspect pod logs and automatically fix broken deployments using Claude CLI as the AI reasoning engine.

<img src="logo.png" width="400">

## Stack

* Rust 1.93+ (edition 2024)
* kube-rs
* axum
* tokio
* Claude CLI (child process)
* Kind (Kubernetes in Docker)

## Endpoints

* `GET /logs` - reads all pod logs across namespaces
* `POST /fix` - collects diagnostics from failing pods/deployments, sends context to Claude CLI, gets corrected YAML, applies fixes to the cluster

## Broken Specs

The `specs/` folder contains intentionally broken K8s manifests:

* `broken-deployment-wrong-image.yaml` - references non-existent image `nginxxxxx:latesttttt`
* `broken-deployment-bad-port.yaml` - readiness/liveness probes on port 9999 while nginx listens on 80
* `broken-deployment-missing-env.yaml` - postgres without required `POSTGRES_PASSWORD` env var

## How to Run

```bash
export ANTHROPIC_API_KEY="your-key"
./start.sh
```

```bash
curl http://localhost:30080/logs
curl -X POST http://localhost:30080/fix
```

```bash
./stop.sh
```

## Build

```bash
cd operator
cargo build --release
```

### Result

```
./build.sh
./start.sh
```

```
--> 81322263762e
[2/2] STEP 1/7: FROM debian:bookworm-slim
[2/2] STEP 2/7: RUN apt-get update && apt-get install -y curl nodejs npm ca-certificates && rm -rf /var/lib/apt/lists/* &&     ARCH=$(uname -m) &&     if [ "$ARCH" = "aarch64" ]; then KARCH="arm64"; else KARCH="amd64"; fi &&     curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/${KARCH}/kubectl" &&     chmod +x kubectl && mv kubectl /usr/local/bin/ &&     npm install -g @anthropic-ai/claude-code
--> Using cache 4bc3064ce730201c4fee822a4d4ebbbac90ec2ffd95274da5d98d6253b891182
--> 4bc3064ce730
[2/2] STEP 3/7: RUN useradd -m -s /bin/bash sre && mkdir -p /home/sre/.kube && chown -R sre:sre /home/sre
--> Using cache a3ce17f97ec394b37adaf7440eba3990435817be34c976338a63ddc1fb1ced68
--> a3ce17f97ec3
[2/2] STEP 4/7: COPY --from=builder /app/target/release/sre-agent /usr/local/bin/sre-agent
--> Using cache ac410eb3914721f5fe84b1d0c087897b0cc40946002e0290aae490e94372a17c
--> ac410eb39147
[2/2] STEP 5/7: USER sre
--> Using cache 29a8bfa67c2bf44ee0a2a121bf3a1b973e298c75fd56352d1783ac7cebc267c2
--> 29a8bfa67c2b
[2/2] STEP 6/7: EXPOSE 8080
--> Using cache 5189fded07381190d9c23aff3381582b362f7df160ef7c40a269ba03b9ace255
--> 5189fded0738
[2/2] STEP 7/7: CMD ["sre-agent"]
--> Using cache 5386ac153f112077f295f3ed9f5aedbef4a365939d706dcec2e8d3ee6206ba83
[2/2] COMMIT sre-agent-operator:latest
--> 5386ac153f11
Successfully tagged localhost/sre-agent-operator:latest
5386ac153f112077f295f3ed9f5aedbef4a365939d706dcec2e8d3ee6206ba83
enabling experimental podman provider
Applying /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/k8s-sre-agent-operator/specs/broken-deployment-bad-port.yaml
deployment.apps/broken-bad-port created
Applying /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/k8s-sre-agent-operator/specs/broken-deployment-missing-env.yaml
deployment.apps/broken-missing-env created
Applying /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/k8s-sre-agent-operator/specs/broken-deployment-wrong-image.yaml
deployment.apps/broken-wrong-image created
Applying /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/k8s-sre-agent-operator/specs/sre-agent-operator.yaml
serviceaccount/sre-agent created
clusterrole.rbac.authorization.k8s.io/sre-agent-role created
clusterrolebinding.rbac.authorization.k8s.io/sre-agent-binding created
deployment.apps/sre-agent-operator created
service/sre-agent-operator created
Waiting for sre-agent-operator pod to be ready...
Forwarding from 127.0.0.1:30080 -> 8080
Forwarding from [::1]:30080 -> 8080

SRE Agent Operator is running.
  GET  /logs -> kovalski logs
  POST /fix  -> kovalski fix
  Port-forward PID: 69022

NAMESPACE            NAME                                                      READY   STATUS              RESTARTS   AGE
default              broken-bad-port-f66875f78-7wddx                           0/1     ContainerCreating   0          4s
default              broken-missing-env-6c4b848f69-gndv5                       0/1     ContainerCreating   0          4s
default              broken-wrong-image-b8cd65b56-vktjg                        0/1     ContainerCreating   0          4s
default              sre-agent-operator-5c8bc95796-4hhzc                       1/1     Running             0          4s
kube-system          coredns-7d764666f9-lvf5v                                  1/1     Running             0          85s
kube-system          coredns-7d764666f9-ts8wn                                  1/1     Running             0          85s
kube-system          etcd-sre-agent-cluster-control-plane                      1/1     Running             0          94s
kube-system          kindnet-4h27v                                             1/1     Running             0          85s
kube-system          kindnet-wb8l6                                             1/1     Running             0          80s
kube-system          kube-apiserver-sre-agent-cluster-control-plane            1/1     Running             0          93s
kube-system          kube-controller-manager-sre-agent-cluster-control-plane   1/1     Running             0          93s
kube-system          kube-proxy-75bmw                                          1/1     Running             0          85s
kube-system          kube-proxy-wxvh9                                          1/1     Running             0          80s
kube-system          kube-scheduler-sre-agent-cluster-control-plane            1/1     Running             0          93s
local-path-storage   local-path-provisioner-67b8995b4b-8mdmj                   1/1     Running             0          85s
```

kovalski
```
❯ operator/target/release/kovalski
Usage: kovalski <command>

Commands:
  logs    Read all pod logs from the cluster
  fix     Fix broken deployments using Claude AI
  status       Show all resources in the cluster (kubectl get all)
  logs-summary Summarize logs using Claude AI

Environment:
  KOVALSKI_URL  Base URL of the SRE agent (default: http://localhost:30080)
```

kovalski status
```
❯ operator/target/release/kovalski status
Handling connection for 30080
NAMESPACE            NAME                                                          READY   STATUS             RESTARTS       AGE
default              pod/broken-bad-port-f66875f78-7wddx                           0/1     Running            4 (18s ago)    2m58s
default              pod/broken-missing-env-6c4b848f69-gndv5                       0/1     Error              4 (2m6s ago)   2m58s
default              pod/broken-wrong-image-b8cd65b56-vktjg                        0/1     ImagePullBackOff   0              2m58s
default              pod/sre-agent-operator-5c8bc95796-4hhzc                       1/1     Running            0              2m58s
kube-system          pod/coredns-7d764666f9-lvf5v                                  1/1     Running            0              4m19s
kube-system          pod/coredns-7d764666f9-ts8wn                                  1/1     Running            0              4m19s
kube-system          pod/etcd-sre-agent-cluster-control-plane                      1/1     Running            0              4m28s
kube-system          pod/kindnet-4h27v                                             1/1     Running            0              4m19s
kube-system          pod/kindnet-wb8l6                                             1/1     Running            0              4m14s
kube-system          pod/kube-apiserver-sre-agent-cluster-control-plane            1/1     Running            0              4m27s
kube-system          pod/kube-controller-manager-sre-agent-cluster-control-plane   1/1     Running            0              4m27s
kube-system          pod/kube-proxy-75bmw                                          1/1     Running            0              4m19s
kube-system          pod/kube-proxy-wxvh9                                          1/1     Running            0              4m14s
kube-system          pod/kube-scheduler-sre-agent-cluster-control-plane            1/1     Running            0              4m27s
local-path-storage   pod/local-path-provisioner-67b8995b4b-8mdmj                   1/1     Running            0              4m19s

NAMESPACE     NAME                         TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                  AGE
default       service/kubernetes           ClusterIP   10.96.0.1       <none>        443/TCP                  4m28s
default       service/sre-agent-operator   NodePort    10.96.169.246   <none>        8080:30080/TCP           2m58s
kube-system   service/kube-dns             ClusterIP   10.96.0.10      <none>        53/UDP,53/TCP,9153/TCP   4m27s

NAMESPACE            NAME                                     READY   UP-TO-DATE   AVAILABLE   AGE
default              deployment.apps/broken-bad-port          0/1     1            0           2m58s
default              deployment.apps/broken-missing-env       0/1     1            0           2m58s
default              deployment.apps/broken-wrong-image       0/1     1            0           2m58s
default              deployment.apps/sre-agent-operator       1/1     1            1           2m58s
kube-system          deployment.apps/coredns                  2/2     2            2           4m27s
local-path-storage   deployment.apps/local-path-provisioner   1/1     1            1           4m25s

NAMESPACE            NAME                                                DESIRED   CURRENT   READY   AGE
default              replicaset.apps/broken-bad-port-f66875f78           1         1         0       2m58s
default              replicaset.apps/broken-missing-env-6c4b848f69       1         1         0       2m58s
default              replicaset.apps/broken-wrong-image-b8cd65b56        1         1         0       2m58s
default              replicaset.apps/sre-agent-operator-5c8bc95796       1         1         1       2m58s
kube-system          replicaset.apps/coredns-7d764666f9                  2         2         2       4m20s
local-path-storage   replicaset.apps/local-path-provisioner-67b8995b4b   1         1         1       4m20s
```

