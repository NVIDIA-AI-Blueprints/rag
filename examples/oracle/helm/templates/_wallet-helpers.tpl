{{/*
Oracle wallet volume + volumeMount + env helpers.
When oracle.wallet.secretName is set, these inject the wallet Secret
as a read-only volume at /app/wallet and set TNS_ADMIN.
*/}}

{{- define "oracle.wallet.volumes" -}}
{{- if (((.Values.oracle).wallet).secretName) }}
- name: oracle-wallet
  secret:
    secretName: {{ .Values.oracle.wallet.secretName | quote }}
    defaultMode: 0444
{{- end }}
{{- end -}}

{{- define "oracle.wallet.volumeMounts" -}}
{{- if (((.Values.oracle).wallet).secretName) }}
- name: oracle-wallet
  mountPath: /app/wallet
  readOnly: true
{{- end }}
{{- end -}}

{{- define "oracle.wallet.env" -}}
{{- if (((.Values.oracle).wallet).secretName) }}
- name: TNS_ADMIN
  value: "/app/wallet"
{{- end }}
{{- end -}}
