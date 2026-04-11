{{/*
Expand the chart name.
*/}}
{{- define "gcnv-data-ingestor.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "gcnv-data-ingestor.fullname" -}}
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
Chart name and version.
*/}}
{{- define "gcnv-data-ingestor.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "gcnv-data-ingestor.labels" -}}
helm.sh/chart: {{ include "gcnv-data-ingestor.chart" . }}
{{ include "gcnv-data-ingestor.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{/*
Selector labels copied from the source deployment.
*/}}
{{- define "gcnv-data-ingestor.selectorLabels" -}}
app.kubernetes.io/name: {{ .Values.selectorLabels.name }}
app.kubernetes.io/instance: {{ .Values.selectorLabels.instance }}
{{- end -}}

{{/*
Service account name.
*/}}
{{- define "gcnv-data-ingestor.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (printf "%s-sa" (include "gcnv-data-ingestor.fullname" .)) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
App PVC name.
*/}}
{{- define "gcnv-data-ingestor.appPvcName" -}}
{{- if .Values.appData.existingClaim -}}
{{- .Values.appData.existingClaim -}}
{{- else -}}
{{- .Values.appData.name -}}
{{- end -}}
{{- end -}}

{{/*
Source PVC name.
*/}}
{{- define "gcnv-data-ingestor.sourcePvcName" -}}
{{- if .Values.sourceData.existingClaim -}}
{{- .Values.sourceData.existingClaim -}}
{{- else -}}
{{- .Values.sourceData.name -}}
{{- end -}}
{{- end -}}
