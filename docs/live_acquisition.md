# Integração com aquisição EEG ao vivo

O BrainSniffer não contém um driver de monitor cirúrgico específico. O núcleo aceita uma fonte de chunks; para pesquisa, há duas entradas:

1. Lab Streaming Layer (LSL), usando `pylsl` opcional.
2. JSON Lines pela entrada padrão, para um bridge local que leia o SDK ou protocolo do fabricante.

## Caminho LSL

O equipamento ou software de aquisição deve publicar um outlet com:

- tipo `EEG`;
- taxa regular declarada;
- canais documentados, unidades e referência no metadado do stream;
- timestamps por amostra;
- uma política clara para desconexão, perda de chunk e reconexão.

Quando o descritor XML LSL fornecer `channels/channel/label`, `unit`,
`reference` e `montage`, o adaptador importa esses campos automaticamente para o
manifesto da sessão. Flags explícitas da CLI podem complementar ou confirmar os
valores; conflitos são rejeitados.

Instalação e execução:

```bash
uv sync --extra dev --extra live
uv run brainsniffer stream-lsl \
  --stream-name "Nome do stream" \
  --stream-type EEG \
  --channel 0 \
  --checkpoint models/brainsniffer_cnn.pt \
  --unit uV --channel-name Fpz \
  --reference "linked ears" --montage "frontal referenced" \
  --require-metadata --require-timestamps
```

Se o nome não for conhecido, omita `--stream-name` e filtre por `--stream-type`. O primeiro canal é uma escolha de configuração, não uma garantia de que ele é o canal frontal correto.

Para um smoke test de bancada reproduzível, o repositório inclui
`examples/lsl_synthetic_publisher.py`. Ele publica uma senoide documentada como
EEG sintético, não representa um paciente e não substitui o teste do hardware:

```bash
uv run python examples/lsl_synthetic_publisher.py --duration 30
uv run brainsniffer stream-lsl \
  --stream-name BrainSnifferSyntheticEEG \
  --stale-timeout 2 \
  --require-metadata --require-timestamps --fail-on-audit \
  --checkpoint models/brainsniffer_cnn.pt
```

`--stale-timeout` define quanto tempo sem um chunk é tolerado. O padrão de
engenharia é 2 s. Com `--fail-on-audit`, o silêncio encerra a sessão e grava o
relatório parcial; sem essa flag, o núcleo limpa o estado causal, invalida a
última saída com `ABSTAIN` e aguarda uma janela completa depois que o stream
retorna. O timeout precisa ser medido novamente no bridge e no hardware alvo.

## Bridge JSON

O bridge deve enviar uma linha por chunk. `samples` é uma lista de números de um único canal; `sampling_rate` deve informar a taxa real do chunk (ou use `--source-rate`); `timestamps` é opcional, mas recomendado:

```json
{"samples": [0.1, 0.2, 0.0], "sampling_rate": 256, "timestamps": [1710000000.0, 1710000000.00390625, 1710000000.0078125], "metadata": {"unit": "uV", "channel_name": "Fpz", "reference": "linked ears", "montage": "frontal referenced"}}
```

```bash
meu-bridge-do-fabricante | uv run brainsniffer stream-json \
  --checkpoint models/brainsniffer_cnn.pt \
  --require-metadata --require-timestamps
```

O objeto `metadata` pode ser enviado no primeiro chunk; os campos também podem
ser fornecidos pela CLI. O processo registra esses valores no relatório, permite
complementar um manifesto já iniciado e rejeita conflitos durante a sessão. Para
uma aquisição que precisa do gate completo:

```bash
meu-bridge-do-fabricante | uv run brainsniffer stream-json \
  --checkpoint models/brainsniffer_cnn.pt \
  --unit uV --channel-name Fpz \
  --reference "linked ears" --montage "frontal referenced" \
  --require-metadata --require-timestamps --fail-on-audit
```

Quando o manifesto já foi revisado pelo laboratório, ele pode ser mantido como
arquivo versionado. O modelo [`../examples/stream_metadata.template.json`](../examples/stream_metadata.template.json)
contém os campos essenciais e campos opcionais do equipamento:

```bash
cp examples/stream_metadata.template.json examples/stream_metadata.local.json
# preencher e revisar o arquivo antes da captura
meu-bridge-do-fabricante | uv run brainsniffer stream-json \
  --checkpoint models/brainsniffer_cnn.pt \
  --metadata-file examples/stream_metadata.local.json \
  --require-metadata --require-timestamps --fail-on-audit
```

O mesmo `--metadata-file` funciona em `audit-json` e `stream-lsl`. Os valores do
arquivo são combinados com flags, metadados dos chunks e descritor XML do LSL;
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
Na aba LSL da interface, timestamps válidos por amostra são sempre obrigatórios;
o processo falha antes de avançar o filtro se essa condição não for atendida. A
opção de rejeitar a sessão quando a auditoria falhar durante a captura ou no
encerramento vem ativada por padrão;
ela pode ser desmarcada apenas para caracterização exploratória.

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

Em uma captura LSL com `--duration`, zero amostras é tratado como erro (e não
como uma sessão concluída), para que uma descoberta tardia ou um outlet parado
não seja confundido com aquisição válida.

Timestamps fornecidos ao resampler/estimador precisam ser finitos e estritamente
crescentes; uma regressão faz o caminho online falhar antes de avançar o estado do
filtro. Uma lacuna positiva é reportada pelo `audit-json` e deve ser tratada no
protocolo de aquisição.

## Checklist antes de qualquer estudo com voluntários/pacientes

- Confirmar com o fabricante/engenheiro clínico o SDK, licença, taxa, unidade e montagem do canal.
- Verificar em bancada se o número de amostras por segundo realmente coincide com o metadado.
- Testar sinal sintético, ruído, eletrodo desconectado, saturação, linha plana, perda e reconexão.
- Medir latência p50/p95 com `uv run brainsniffer benchmark-latency` e com o hardware alvo, incluindo o atraso do resampler quando a fonte não for 128 Hz.
- Registrar estado do filtro/resampler, versão do modelo, SHA-256 do checkpoint e configuração do stream. Os comandos `stream-json` e `stream-lsl` incluem esse hash em cada registro de predição.
- Manter o relatório de sessão versão 2 junto ao registro do estudo: ele inclui a
  configuração de pré-processamento, ambiente Python, stride, gate de metadata e
  o manifesto do sinal, sem armazenar as amostras EEG.
- Validar o limiar de abstenção usando SQI/anotação do monitor, não apenas a heurística atual.
- Fazer revisão de anestesiologista, ética, proteção de dados, análise de risco e requisitos regulatórios.

O LSL é uma camada de transporte e sincronização de pesquisa; ele não valida o EEG, não calcula BIS e não autoriza uso clínico. A documentação oficial descreve streams, chunks, metadados e timestamps em [Lab Streaming Layer](https://labstreaminglayer.readthedocs.io/info/intro.html).
