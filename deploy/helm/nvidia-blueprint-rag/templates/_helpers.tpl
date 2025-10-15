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
