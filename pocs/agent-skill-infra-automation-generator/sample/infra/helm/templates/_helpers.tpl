{{- define "agent-werewolf.fullname" -}}
{{- .Release.Name }}-{{ .Chart.Name }}
{{- end }}
{{- define "agent-werewolf.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
{{- end }}
{{- define "agent-werewolf.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
