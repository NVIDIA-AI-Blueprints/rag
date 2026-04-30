{{/*
Expand the name of the chart.
*/}}
{{- define "nvidia-blueprint-rag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "nvidia-blueprint-rag.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "nvidia-blueprint-rag.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{ include "nvidia-blueprint-rag.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "nvidia-blueprint-rag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nvidia-blueprint-rag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Service account name
*/}}
{{- define "nvidia-blueprint-rag.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "nvidia-blueprint-rag.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Generate DockerConfigJson for image pull secrets
*/}}
{{- define "imagePullSecret" -}}
{{- printf "{\"auths\":{\"%s\":{\"auth\":\"%s\"}}}" .Values.imagePullSecret.registry (printf "%s:%s" .Values.imagePullSecret.username .Values.imagePullSecret.password | b64enc) | b64enc -}}
{{- end -}}

{{/*
Create secret to access NGC Api
*/}}
{{- define "ngcApiSecret" -}}
{{- printf "%s" .Values.ngcApiSecret.password | b64enc -}}
{{- end -}}

{{/*
Get API keys secret name (either existing or created)
*/}}
{{- define "apiKeysSecretName" -}}
{{- if .Values.apiKeysSecret.existingSecret -}}
{{- .Values.apiKeysSecret.existingSecret -}}
{{- else -}}
{{- .Values.apiKeysSecret.name -}}
{{- end -}}
{{- end -}}

{{/*
Elasticsearch resource base name
*/}}
{{- define "nvidia-blueprint-rag.elasticsearchFullname" -}}
{{- $esCfg := index .Values "eck-elasticsearch" -}}
{{- default (printf "%s-eck-elasticsearch" .Release.Name) $esCfg.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Elasticsearch elastic user secret name
*/}}
{{- define "nvidia-blueprint-rag.elasticsearchUserSecretName" -}}
{{- printf "%s-es-elastic-user" (include "nvidia-blueprint-rag.elasticsearchFullname" .) -}}
{{- end -}}

{{/*
Whether bundled ECK Elasticsearch has xpack security enabled.
*/}}
{{- define "nvidia-blueprint-rag.elasticsearchSecurityEnabled" -}}
{{- $esCfg := index .Values "eck-elasticsearch" -}}
{{- $nodeSets := default (list) $esCfg.nodeSets -}}
{{- $firstNodeSet := dict -}}
{{- if gt (len $nodeSets) 0 -}}
{{- $firstNodeSet = index $nodeSets 0 -}}
{{- end -}}
{{- $esNodeCfg := default (dict) (index $firstNodeSet "config") -}}
{{- $enabled := true -}}
{{- if hasKey $esNodeCfg "xpack.security.enabled" -}}
{{- $enabled = index $esNodeCfg "xpack.security.enabled" -}}
{{- end -}}
{{- if eq (toString $enabled) "true" -}}true{{- else -}}false{{- end -}}
{{- end -}}

{{/*
SeaweedFS resource base name
*/}}
{{- define "nvidia-blueprint-rag.seaweedfsFullname" -}}
{{- $cfg := .Values.seaweedfs -}}
{{- if $cfg.fullnameOverride -}}
{{- $cfg.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (default "seaweedfs" $cfg.nameOverride) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/*
SeaweedFS all-in-one resource name
*/}}
{{- define "nvidia-blueprint-rag.seaweedfsAllInOneName" -}}
{{- printf "%s-all-in-one" (include "nvidia-blueprint-rag.seaweedfsFullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
SeaweedFS labels
*/}}
{{- define "nvidia-blueprint-rag.seaweedfsLabels" -}}
app.kubernetes.io/name: {{ default "seaweedfs" .Values.seaweedfs.nameOverride }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: seaweedfs-all-in-one
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end -}}

{{/*
SeaweedFS selector labels
*/}}
{{- define "nvidia-blueprint-rag.seaweedfsSelectorLabels" -}}
app.kubernetes.io/name: {{ default "seaweedfs" .Values.seaweedfs.nameOverride }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: seaweedfs-all-in-one
{{- end -}}

{{/*
Redis resource base name used by the nv-ingest topology
*/}}
{{- define "nvidia-blueprint-rag.redisFullname" -}}
{{- $redis := (index .Values "nv-ingest").redis -}}
{{- default "rag-redis" $redis.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Redis master service/deployment name
*/}}
{{- define "nvidia-blueprint-rag.redisMasterName" -}}
{{- printf "%s-master" (include "nvidia-blueprint-rag.redisFullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
