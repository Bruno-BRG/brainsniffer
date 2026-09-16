# Bridge JSONL opcional para pesquisa local

O TCC é exclusivamente retrospectivo: arquivos existentes e replay, sem conexão
com EEG físico. O núcleo mantém JSON Lines pela entrada padrão como contrato
opcional de chunks para testes locais. Não há driver de monitor, integração
validada com hardware ou compromisso de aquisição futura. LSL foi removido;
seus artefatos históricos são identificados ao final deste documento.

Os exemplos abaixo usam `chunks.jsonl`, um arquivo local previamente preparado
com sinal sintético ou gravação autorizada. Não são comandos de captura.

## Bridge JSON

O bridge deve enviar uma linha por chunk. `samples` é uma lista de números de um único canal; `sampling_rate` deve informar a taxa real do chunk (ou use `--source-rate`); `timestamps` é opcional, mas recomendado:

```json
{"samples": [0.1, 0.2, 0.0], "sampling_rate": 256, "timestamps": [1710000000.0, 1710000000.00390625, 1710000000.0078125], "metadata": {"unit": "uV", "channel_name": "Fpz", "reference": "linked ears", "montage": "frontal referenced"}}
```

```bash
uv run brainsniffer stream-json < chunks.jsonl \
  --checkpoint models/brainsniffer_cnn.pt \
  --require-metadata --require-timestamps
```

O objeto `metadata` pode ser enviado no primeiro chunk; os campos também podem
ser fornecidos pela CLI. O processo registra esses valores no relatório, permite
complementar um manifesto já iniciado e rejeita conflitos durante a sessão. Para
uma aquisição que precisa do gate completo:

```bash
uv run brainsniffer stream-json < chunks.jsonl \
  --checkpoint models/brainsniffer_cnn.pt \
  --unit uV --channel-name Fpz \
  --reference "linked ears" --montage "frontal referenced" \
  --require-metadata --require-timestamps --fail-on-audit
```

Quando o manifesto já foi revisado pelo laboratório, ele pode ser mantido como
arquivo versionado. O modelo [`../examples/stream_metadata.template.json`(../../examples/stream_metadata.template.json)
contém os campos essenciais e campos opcionais do equipamento:

```bash
cp examples/stream_metadata.template.json examples/stream_metadata.local.json
# preencher e revisar o arquivo antes da captura
uv run brainsniffer stream-json < chunks.jsonl \
  --checkpoint models/brainsniffer_cnn.pt \
  --metadata-file examples/stream_metadata.local.json \
  --require-metadata --require-timestamps --fail-on-audit
```

O mesmo `--metadata-file` funciona em `audit-json` e `stream-json`. Os valores do
arquivo são combinados com flags e metadados dos chunks;
qualquer divergência é rejeitada, preservando a proveniência do sinal.

Antes do primeiro stream de um equipamento, valide a ficha completa sem iniciar
o modelo:

```bash
uv run brainsniffer validate-intake \
  --metadata-file examples/stream_metadata.local.json
```

Esse gate exige fabricante, modelo, firmware/software, bridge, taxa nominal,
unidade em microvolt, canal, referência, montagem, faixa nominal/saturação e
processamento/ganho aplicado. `ready_for_bench` autoriza apenas a
próxima verificação técnica em bancada; não autoriza aquisição em paciente.
Em uma sessão, use `--require-intake` junto com `--require-metadata` para aplicar
essa exigência antes da inferência e registrá-la no relatório.

O `audit-json` aceita as mesmas opções e inclui `metadata_complete` e
`metadata_missing` no preflight. Sem `--require-metadata`, a ausência é registrada
como advertência de documentação, mas continua sendo uma lacuna para o protocolo
de hardware. Se o metadata incluir `sampling_rate`, ela precisa coincidir com a
taxa efetiva dos chunks; uma divergência encerra a sessão antes da inferência.
O consumo JSONL é feito pela CLI `stream-json`, não pelo Dash. Use explicitamente
`--require-timestamps` para exigir timestamps válidos por amostra e
`--fail-on-audit` para rejeitar a sessão quando a auditoria falhar. Essas flags
não são ativadas por padrão na CLI; sua omissão deixa o modo exploratório e não
representa aceite do equipamento. O Dash oferece apenas inspeção retrospectiva
nessa cadeia, sem controles de captura ou validação da ficha.

Para registrar somente metadados da sessão, sem guardar o EEG bruto, acrescente
`--report registros/sessao.json`. O relatório inclui auditoria de taxa, lacunas,
qualidade, número de predições, abstenções, o SHA-256 do checkpoint, o ambiente,
o pré-processamento efetivo, o stride da sessão e as flags efetivas de gate,
incluindo `fail_on_audit`. Se houver
falha de contrato ou interrupção, o arquivo é salvo como `status="error"` com a
causa e os diagnósticos parciais; a exceção ainda encerra o processo.
O campo `scope` declara `intended_use="research_only"`,
`clinical_decision_support=false` e `controls_anesthetic_delivery=false`.
Com `--fail-on-audit`, uma auditoria reprovada durante a captura ou no encerramento
termina o processo e mantém o relatório parcial para investigação. Sem essa opção,
o stream pode continuar em modo exploratório e o estimador deve se abster em
janelas de baixa qualidade.

Antes de conectar esse bridge ao estimador, faça um preflight finito ou de uma
gravação de bancada:

```bash
cat chunks.jsonl | uv run brainsniffer audit-json --source-rate 256
```

O relatório verifica a estrutura do stream, taxa, timestamps, lacunas, valores
não finitos, saturação, linha plana e uma qualidade heurística. `ok` é somente um
critério de engenharia; não é confiança do modelo, não é SQI clínico e não libera
o uso em pacientes. Quando `ok=false`, o comando termina com código de saída 1;
isso permite interromper automaticamente um gate de bancada ou CI.

No caminho de inferência, amostras `NaN` ou `Inf` são rejeitadas antes de entrar
no filtro causal ou no resampler. Isso é deliberadamente diferente da análise
offline, que pode imputar valores ausentes apenas para inspeção do dataset. Um
chunk inválido encerra o stream com erro e, se `--report` estiver ativo, deixa um
relatório parcial com os diagnósticos acumulados.

Se o bridge não incluir `sampling_rate` em cada linha, informe-a explicitamente, por exemplo `--source-rate 256`. A taxa deve permanecer constante durante o stream; para mudar a taxa, reinicie o processo e o resampler. O resampler mantém estado entre chunks e libera a cauda ao chegar ao fim de um JSONL finito; se nenhuma das duas formas informar a taxa, o processo assume 128 Hz e avisa no stderr.

O limite de lacuna pode ser ajustado com `--max-gap-factor`; use o mesmo valor no
`audit-json` e no stream que será aceito, e preserve-o junto ao relatório. O
padrão 1,5 é um critério de engenharia, não uma tolerância clínica universal.

O bridge é o lugar correto para converter unidades do equipamento para a unidade documentada no treino, escolher referência/montagem, verificar taxa e remover identificadores. Nunca envie prontuário, nome, número de registro ou outros dados pessoais para o processo se eles não forem necessários.

Timestamps fornecidos ao resampler/estimador precisam ser finitos e estritamente
crescentes; uma regressão faz o caminho online falhar antes de avançar o estado do
filtro. Uma lacuna positiva é reportada pelo `audit-json` e deve ser tratada no
protocolo de aquisição.

## Limites de uma eventual pesquisa separada — fora do TCC

- Confirmar com o fabricante/engenheiro clínico o SDK, licença, taxa, unidade e montagem do canal.
- Verificar em bancada se o número de amostras por segundo realmente coincide com o metadado.
- Testar sinal sintético, ruído, eletrodo desconectado, saturação, linha plana, perda e reconexão.
- Medir latência p50/p95 com `uv run brainsniffer benchmark-latency` e com o hardware alvo, incluindo o atraso do resampler quando a fonte não for 128 Hz.
- Registrar estado do filtro/resampler, versão do modelo, SHA-256 do checkpoint e configuração do stream. O comando `stream-json` inclui esse hash em cada registro de predição.
- Manter o relatório de sessão versão 2 junto ao registro do estudo: ele inclui a
  configuração de pré-processamento, ambiente Python, stride, gate de metadata e
  o manifesto do sinal, sem armazenar as amostras EEG.
- Validar o limiar de abstenção usando SQI/anotação do monitor, não apenas a heurística atual.
- Fazer revisão de anestesiologista, ética, proteção de dados, análise de risco e requisitos regulatórios.

## Artefatos históricos LSL

LSL, o extra `live`, `pylsl` e o comando `stream-lsl` foram removidos. Os relatos
em [project_status.md(../project_status.md) e os relatórios
`reports/lsl_synthetic_session.json` e `reports/lsl_synthetic_intake_session.json`
registram execuções sintéticas passadas, não capacidades atuais nem testes JSONL.

A fixture `examples/stream_metadata.bench.synthetic.json` foi preservada byte a
byte: `bridge` identifica o publisher histórico removido
`examples/lsl_synthetic_publisher.py` e `timestamp_clock` identifica
`liblsl local_clock`. Esses campos são proveniência, não links/instruções de
execução, equipamento atual ou um manifesto para JSONL. Não os reatribua a outro
transporte; para novos testes locais, use o template e descreva a fonte real.
