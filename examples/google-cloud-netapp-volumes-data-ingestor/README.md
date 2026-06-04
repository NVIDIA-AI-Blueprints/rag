# Google Cloud NetApp Volumes (GCNV) Data Ingestor Helm Chart

This chart packages the deployment of the GCNV Data Ingestor that integrates with the NVIDIA Foundational RAG pipeline into a reusable Helm chart at `examples/google-cloud-netapp-volumes-data-ingestor`.

Create or target the namespace externally with `--namespace ... --create-namespace`. Chart-managed namespace creation is intentionally not supported because Helm cannot reliably create the release namespace from within the same chart.

## Prerequisites

Before installing this chart, make sure the cluster can provision or expose the required PVCs from NetApp Google Cloud NetApp Volumes (GCNV).

1. Install and configure NetApp Trident in the target cluster.
2. Create or use a Trident `StorageClass` that maps to your GCNV backend.
3. Decide how you want the chart to get storage:
   - Let the chart create PVCs by setting `appData.storageClassName` and `sourceData.storageClassName` to Trident-backed classes.
   - Or create the PVCs ahead of time with Trident and set `appData.create=false`, `appData.existingClaim=<claim>`, `sourceData.create=false`, and `sourceData.existingClaim=<claim>`.
   - If you set `create=false`, the matching `existingClaim` is required.
4. Make sure the Docker Hub image and tag you want to deploy are available.
5. If the Docker Hub repository is private, create an image pull secret in the target namespace and set `image.pullSecrets`.

## Chart Layout

```text
examples/google-cloud-netapp-volumes-data-ingestor/
├── Chart.yaml
├── values.yaml
├── values.schema.json
├── README.md
└── templates/
    ├── _helpers.tpl
    ├── deployment.yaml
    ├── pvc.yaml
    ├── service.yaml
    ├── validate.yaml
    └── serviceaccount.yaml
```

## Important Values

Update these values before install:

- `image.repository`: set to your Docker Hub image path
- `image.tag`: set to the image tag you want to deploy
- `appData.storageClassName`: set to your Trident-backed app PVC class when the chart creates the PVC
- `appData.size`: app PVC size request, defaults to `50Gi`
- `appData.existingClaim`: use an already-created PVC instead of letting the chart create one; required when `appData.create=false`
- `sourceData.storageClassName`: set to your Trident-backed GCNV source PVC class when the chart creates the PVC
- `sourceData.size`: source PVC size request, defaults to `200Gi`
- `sourceData.existingClaim`: use an already-created source PVC instead of letting the chart create one; required when `sourceData.create=false`
- `env.nvIngestEndpoint`: set to the reachable NVIDIA ingestor-server `/v1` base URL

The chart validates required values during `helm lint`, `helm template`, `helm install`, and `helm upgrade`.

## Install

You can either edit `values.yaml` directly or use an override file.

Example override file:

```yaml
image:
  repository: docker.io/acme/netapp_volumes_rag_ingestor
  tag: "REPLACE_WITH_REAL_TAG"

appData:
  storageClassName: trident-app

sourceData:
  storageClassName: trident-gcnv

env:
  nvIngestEndpoint: http://YOUR_INGESTOR_SERVER:8082/v1
```

Install with:

```bash
helm install gcnv-data-ingestor ./examples/google-cloud-netapp-volumes-data-ingestor \
  --namespace gcnv-data-ingestor \
  --create-namespace \
  -f my-values.yaml
```

If your Docker Hub repository is private, add an image pull secret. `image.pullSecrets` must be a YAML list of secret names:

```yaml
image:
  pullSecrets:
    - dockerhub-secret
```

## Common Overrides

Resize the chart-managed PVCs:

```yaml
appData:
  size: 100Gi
  storageClassName: trident-app

sourceData:
  size: 500Gi
  storageClassName: trident-gcnv
```

## Use Existing PVCs

If Trident or another workflow already created the claims you want to mount, use overrides like this:

```yaml
appData:
  create: false
  existingClaim: gcnv-ingestor-config-data

sourceData:
  create: false
  existingClaim: gcnv-data-for-rag
```

The chart will mount those existing claims into the Pod instead of creating new PVCs.

If you set `create: false` and leave `existingClaim` empty, the chart now fails fast during Helm validation instead of creating a broken release.

Expose the service differently:

```yaml
service:
  type: ClusterIP
  port: 8000
```

Tune runtime resources:

```yaml
resources:
  requests:
    cpu: 1
    memory: 2Gi
  limits:
    cpu: 4
    memory: 8Gi
```

Pass extra environment variables using normal Kubernetes `env` list syntax:

```yaml
env:
  extra:
    - name: EXTRA_FLAG
      value: "1"
```

## Supported Values

The chart supports overrides for the following areas in `values.yaml`:

- Naming: `nameOverride`, `fullnameOverride`
- Labels: `selectorLabels`, `podLabels`, `podAnnotations`
- Deployment: `replicaCount`, `strategy`
- Image: `image.repository`, `image.tag`, `image.pullPolicy`, `image.pullSecrets`
- Service account: `serviceAccount.create`, `serviceAccount.name`, `serviceAccount.automount`, `serviceAccount.annotations`
- Service: `service.type`, `service.port`, `service.annotations`
- App PVC: `appData.create`, `appData.existingClaim`, `appData.name`, `appData.accessModes`, `appData.size`, `appData.storageClassName`, `appData.mountPath`
- Source PVC: `sourceData.create`, `sourceData.existingClaim`, `sourceData.name`, `sourceData.accessModes`, `sourceData.size`, `sourceData.storageClassName`, `sourceData.mountPath`, `sourceData.readOnly`
- Environment: `env.scanOutputRoot`, `env.appDbPath`, `env.defaultIncrementalSchedulerMins`, `env.nvIngestMode`, `env.nvIngestEndpoint`, `env.extra`
- Health checks: `probes.liveness.*`, `probes.readiness.*`
- Scheduling and placement: `nodeSelector`, `tolerations`, `affinity`
- Resource limits: `resources`

## Verify

```bash
helm template gcnv-data-ingestor ./examples/google-cloud-netapp-volumes-data-ingestor -n gcnv-data-ingestor
kubectl get pods,svc,pvc -n gcnv-data-ingestor
```

The service defaults to `NodePort` with service port `8000`, matching the source manifest. Kubernetes assigns the external node port automatically unless you customize the Service separately.

The default PVC access modes are `ReadWriteOnce`, so increasing `replicaCount` beyond `1` may require different storage semantics or pod placement constraints.
