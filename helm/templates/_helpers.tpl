{{/*
Expand the name of the chart.
*/}}
{{- define "ols.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "redis.name" -}}
{{- default "redis-stack" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "ols.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ols.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ols.labels" -}}
helm.sh/chart: {{ include "ols.chart" . }}
{{ include "ols.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}


{{/*
Selector labels
*/}}
{{- define "ols.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ols.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ols.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ols.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper Redis image name
*/}}
{{- define "redis.image" -}}
{{- printf "%s/%s:%s" .Values.redis.image.registry .Values.redis.image.repository .Values.redis.image.tag | quote -}}
{{- end -}}

{{/*
Return the proper OLS image name
*/}}
{{- define "ols.image" -}}
{{- printf "%s/%s:%s" .Values.ols.api.image.registry .Values.ols.api.image.repository .Values.ols.api.image.tag | quote -}}
{{- end -}}

{{/*
Return the proper kube-rbac-proxy image name
*/}}
{{- define "rbac-proxy.image" -}}
{{- printf "%s/%s:%s" (index .Values.ols "rbac-proxy" "image" "registry") (index .Values.ols "rbac-proxy" "image" "repository") (index .Values.ols "rbac-proxy" "image" "tag") | quote -}}
{{- end -}}
