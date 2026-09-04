# Model card — BrainSniffer CNN 0.1.0

## Resumo

O BrainSniffer é um protótipo de pesquisa que recebe uma janela de um canal de EEG
frontal e estima um valor contínuo entre 0 e 100, tratado no projeto como uma
**referência BIS estimada**. O valor é convertido em quatro faixas de pesquisa:
`deep`, `general`, `light` e `awake`.

Este artefato não é um dispositivo médico, não mede consciência diretamente, não
recomenda dose, não controla bomba e não deve orientar uma cirurgia.

## Uso pretendido

- Reproduzir experimentos com o dataset público EEG and BIS.
- Comparar uma CNN 1-D compacta com baselines espectrais.
- Exercitar inferência causal em replay e em streams de pesquisa LSL/JSON.
- Expor qualidade do sinal e abster-se quando a janela não é confiável pela
  heurística atual.

## Uso não pretendido

- Diagnóstico ou monitorização clínica autônoma.
- Controle ou ajuste de anestésicos.
- Substituição do anestesiologista, de monitor aprovado ou de protocolo clínico.
- Generalização para outra marca, montagem, unidade, população ou fármaco sem
  validação específica.

## Dados e rótulo

O treinamento demonstrativo usa 24 arquivos anônimos do dataset [EEG and BIS raw
data](https://doi.org/10.6084/m9.figshare.5589841.v1), publicado no Figshare sob
CC BY 4.0. O corpus é descrito com EEG frontal a 128 Hz e BIS a cada 5 s.
O `case24` é baixado, mas fica fora do checkpoint padrão porque a auditoria
encontra escala bruta incompatível (0–4095) sem metadado seguro para conversão.

O alvo é o BIS publicado, um índice processado de monitor. Ele é uma referência
operacional e não uma verdade clínica universal; atrasos, artefatos, EMG, drogas,
estímulo cirúrgico e contexto do paciente podem alterar sua interpretação.

## Modelo e pré-processamento

- Entrada: `(1, 640)` amostras, equivalente a 5 s a 128 Hz.
- Pré-processamento: clipping documentado, passa-banda 0,5–45 Hz, notch de 50 Hz,
  escala de amplitude e filtros causais com estado entre chunks; o checkpoint
  usa `label_offset_seconds=0`.
- Modelo: quatro blocos Conv1D com BatchNorm, GELU, pooling e regressão limitada
  por sigmoide ao intervalo 0–100.
- Arquitetura experimental disponível em `RobustConv1DDepthEstimator`: blocos
  residuais depthwise-separable, GroupNorm independente do tamanho do batch,
  dilatação temporal, pooling médio/máximo e MC Dropout para incerteza
  exploratória. Ela é opt-in e não substitui o checkpoint acima.
- O loop de treino registra split/tamanho, métricas por época, taxa de
  aprendizado e ambiente; inclui early stopping, `ReduceLROnPlateau`, clipping
  de gradiente, seed determinística e mixed precision somente quando há CUDA.
- Suavização: EWMA somente na apresentação em fluxo; o valor bruto permanece
  disponível.
- Abstenção: qualidade heurística abaixo de 0,20 produz `stage="abstain"` e não
  mantém um BIS antigo visível.

## Avaliação reproduzível atual

O checkpoint demonstrativo foi treinado com seed 42, dez épocas e separação por
caso: 13 casos em treino, 5 em validação e 5 em teste. Foram usadas 31.055
janelas válidas; 23 casos passaram o gate de qualidade.

| Métrica no teste por caso | CNN |
|---|---:|
| MAE BIS | 7,03 |
| RMSE BIS | 11,08 |
| Viés | 0,79 |
| Correlação de Pearson | 0,784 |
| Acurácia das faixas | 0,585 |
| Macro-F1 das faixas | 0,548 |

Um bootstrap exploratório de 1.000 reamostragens de cirurgias inteiras no
holdout estimou Pearson médio de 0,789 (95%: 0,703–0,881) e MAE médio de 7,11
(95%: 6,38–8,24). Esses intervalos têm somente cinco casos e não são
intervalos clínicos, validação externa, calibração, estudo prospectivo,
comparação multicêntrica ou evidência de segurança.

Uma avaliação exploratória sem retreino nos 15 casos VitalDB compatíveis
(1–10, 12–14, 16 e 17) teve MAE 12,43, Pearson 0,024 e macro-F1 0,398 em 38.730
janelas. A expansão ocorreu depois do primeiro piloto 1–5, portanto não é um
resultado pré-registrado. A variação por caso foi grande; isso é compatível com
mudança de domínio e não deve ser usado como estimativa final de desempenho. O
piloto histórico de dez casos permanece documentado em
`docs/vitaldb_external_validation.md` para preservar a evolução do protocolo.

O catálogo VitalDB declara `BIS/EEG1_WAV` em µV a 128 Hz; a rotina de normalização
preserva essa unidade, os nomes dos tracks e seus IDs. Os 15 arquivos observados
possuem caudas de amplitude aproximadamente entre −1,477×10³ e 1,800×10³ µV. Como não há
uma especificação equivalente de ganho/montagem no registro Figshare usado, o
pipeline não faz conversão arbitrária e a comparação permanece exploratória.
Os diagnósticos da avaliação registraram pontos não finitos nos 15 arquivos; a
preparação offline pode imputá-los para inspeção das janelas, mas o caminho online
os rejeita antes do filtro causal. Esse tratamento diferente é deliberado e deve
ser considerado ao comparar os resultados.
Arquivos VitalDB gerados antes desta mudança que não tenham esses campos devem
ser recriados com `--overwrite`.

O bootstrap de 1.000 casos inteiros nos 15 casos, com seed determinística 42,
estimou MAE 12,52 [11,05–14,45] e Pearson 0,023 [−0,126–0,193] como intervalos
exploratórios de 95%. Quinze casos continuam insuficientes para uma conclusão
clínica ou uma validação externa definitiva.

O primeiro candidato do corpus misto foi treinado com 23 casos Figshare e 10
casos VitalDB elegíveis, excluindo do ajuste os cinco casos do holdout Figshare
histórico e mantendo os 15 VitalDB externos congelados. No holdout Figshare fixo,
obteve MAE 6,69 e Pearson 0,818; no VitalDB externo, MAE 8,60 e Pearson 0,688.
Esses resultados sugerem ganho de generalização de domínio, mas o artefato é
experimental e não foi promovido automaticamente ao checkpoint ativo. O protocolo
completo, gates e hashes estão em [`docs/mixed_corpus.md`](mixed_corpus.md).

## Riscos e validação necessária

Antes de qualquer contato com pacientes, o laboratório precisa confirmar taxa,
unidade em microvolt, faixa nominal/saturação, processamento/ganho, referência,
montagem, canal, timestamps e política de perda no hardware
alvo. A ficha mínima pode ser verificada sem EEG com
`uv run brainsniffer validate-intake --metadata-file ...`; no stream, o gate
equivalente é `--require-intake`. Isso só libera a bancada técnica. Também precisa fazer revisão por anestesiologista, ética, proteção de dados,
análise de riscos, validação externa e avaliação regulatória aplicável.

O próximo experimento deve incluir um segundo centro/aparelho, referência clínica
complementar ao BIS, alinhamento do atraso do monitor, SQI anotado, desempenho por
paciente/fármaco/qualidade e uma política de abstenção calibrada.
O plano de execução por etapas, incluindo bancada, validação externa travada e
modo sombra, está em [`docs/prospective_protocol.md`](prospective_protocol.md) e
segue as referências GMLP, DECIDE-AI e TRIPOD+AI.

## Artefatos e reprodução

- Checkpoint: `models/brainsniffer_cnn.pt`.
- Metadados, métricas e SHA-256 do checkpoint: `models/brainsniffer_cnn.json`.
- Relatório reproduzível do holdout Figshare: `reports/figshare_holdout_evaluation.json`.
- Relatório externo exploratório reproduzível: `reports/vitaldb_external_validation.json`.
- Manifesto e auditoria do corpus misto: `reports/corpus_manifest.json`.
- Avaliações do candidato misto: `reports/mixed_fixed_figshare_holdout.json` e
  `reports/mixed_vitaldb_external.json`.
- Checkpoint candidato (não ativo): `models/brainsniffer_corpus_fixed.pt` e
  `models/brainsniffer_corpus_fixed.json`.
- Novos checkpoints também registram o manifesto SHA-256/tamanho dos arquivos
  de entrada; `evaluate` verifica esses arquivos antes de recalcular métricas.
- O carregador verifica automaticamente o SHA-256 quando o `.json` acompanha o
  `.pt`; mantenha os dois arquivos juntos.
- Os comandos de treino, avaliação, replay, benchmark e streaming registram esse
  digest para vincular cada resultado ao modelo executado.
- Ambiente de execução (Python e versões das bibliotecas): campo `environment`
  no checkpoint e no JSON.
- Registro das decisões: `docs/decisions.md`.
- Ledger de fontes, hipóteses e limites: `docs/source_ledger.md`.
- Artigo técnico: `docs/article.md`.
- Roteiro falado: `docs/talk_script.md`.
- Contrato de aquisição: `docs/live_acquisition.md`.
- Protocolo de avanço e gates: `docs/prospective_protocol.md`.

Reprodução mínima:

```bash
uv sync --extra dev
uv run brainsniffer download-data
uv run brainsniffer train --epochs 10 --min-quality 0.2 \
  --checkpoint models/brainsniffer_cnn.pt
uv run brainsniffer evaluate --checkpoint models/brainsniffer_cnn.pt \
  --report reports/figshare_holdout_evaluation.json \
  --bootstrap-samples 1000 --bootstrap-seed 42
uv run pytest -q
```

O comando `evaluate` não treina novamente: ele lê a divisão por caso salva no
checkpoint, reconstrói as janelas com a configuração persistida e mostra as
métricas armazenadas ao lado das métricas recalculadas.

As decisões metodológicas são fundamentadas na literatura de DoA/CNN, no corpus
publicado, na documentação do LSL e nas limitações documentadas de índices pEEG;
as fontes completas estão no [README](../README.md) e em
[`docs/decisions.md`](decisions.md).
