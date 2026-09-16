# Ficha de entrada para EEG real

Esta ficha deve ser preenchida pelo laboratório antes de conectar um equipamento
ao BrainSniffer. O resultado esperado é um stream de pesquisa anonimizado e
documentado; preencher a ficha não autoriza uso clínico.

## Identificação do equipamento

- Fabricante e modelo:
- Versão de firmware/software:
- SDK, protocolo ou bridge utilizado:
- Licença e autorização para exportar o sinal:
- Data, operador e versão do BrainSniffer:
- Hash do checkpoint usado:

## Definição do sinal

- Taxa nominal e taxa medida (Hz):
- Número de canais exportados:
- Canal usado pelo modelo:
- Nome e posição do canal:
- Montagem e referência (por exemplo, referencial/montagem bipolar):
- Unidade da amostra:
- Faixa nominal, resolução e saturação:
- Sinal cru ou já processado:
- Filtro/notch aplicado pelo equipamento:
- Relógio dos timestamps e sincronização com o computador:
- Política para amostras faltantes, chunks perdidos e reconexão:

Se qualquer campo essencial estiver desconhecido, o stream deve permanecer em
modo de caracterização e não alimentar uma avaliação em paciente.

## Pacote mínimo de bancada

Entregar, sem identificadores pessoais:

1. metadados acima em texto;
2. pelo menos cinco minutos de stream com timestamps por amostra;
3. uma captura curta com sinal sintético conhecido ou entrada de teste do
   fabricante;
4. uma captura com o eletrodo desconectado, se isso for seguro no equipamento;
5. registro dos eventos de perda, pausa e reconexão;
6. o JSONL bruto ou a gravação LSL e o relatório de `audit-json`.

O bridge deve converter o sinal para a unidade documentada e remover qualquer
prontuário, nome, número de registro ou outro identificador que não seja
necessário para o experimento.

## Critérios de aceite antes do replay

| Verificação | Evidência mínima | Decisão |
|---|---|---|
| Taxa | taxa medida compatível com o metadado e constante | aceitar/rejeitar |
| Integridade | todas as amostras finitas; nenhuma regressão temporal | aceitar/rejeitar |
| Temporização | lacunas quantificadas e explicadas | aceitar/rejeitar |
| Unidade/escala | unidade em microvolt compatível, faixa nominal/saturação e processamento documentados; distribuição comparada com o treino, sem conversão arbitrária | aceitar/rejeitar |
| Canal | posição, referência e montagem confirmadas | aceitar/rejeitar |
| Falhas | desconexão, perda e reconexão reproduzidas em bancada | aceitar/rejeitar |
| Qualidade | saturação, linha plana e ruído identificados | aceitar/rejeitar |
| Inferência | replay causal sem valor stale após abstenção | aceitar/rejeitar |
| Latência | p50/p95 medidos no hardware e no host alvo | aceitar/rejeitar |

Comandos de referência:

```bash
uv run brainsniffer validate-intake \
  --metadata-file examples/stream_metadata.local.json
cat captura.jsonl | uv run brainsniffer audit-json --source-rate 256 \
  --metadata-file examples/stream_metadata.local.json \
  --require-metadata --require-intake --require-timestamps
uv run brainsniffer benchmark-latency --iterations 30
cat captura.jsonl | uv run brainsniffer stream-json \
  --checkpoint models/brainsniffer_cnn.pt --source-rate 256 \
  --metadata-file examples/stream_metadata.local.json \
  --require-metadata --require-intake --require-timestamps --fail-on-audit \
  --report registros/sessao.json
```

Para transformar uma captura finita em gate automatizado, acrescente
`--metadata-file ... --require-metadata --require-intake --fail-on-audit --report ...`; uma sessão
reprovada termina com código de saída 1 e preserva o relatório parcial para
investigação.

O `audit-json` é um preflight de engenharia, e `benchmark-latency` mede somente
o caminho disponível no ambiente em que é executado. O laboratório deve repetir
a medição com o equipamento, o bridge, a visualização e a carga do host alvo.

## Gate para estudo prospectivo

Somente depois do aceite técnico devem ser fechados o protocolo do estudo, a
referência clínica complementar ao BIS, o alinhamento do atraso, a política de
abstenção, o plano de incidentes, a anonimização, a revisão do anestesiologista,
ética, segurança e avaliação regulatória aplicável. O BrainSniffer não controla
bombas, não recomenda dose e não substitui monitor aprovado ou julgamento
clínico.
