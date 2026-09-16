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
- Exercitar inferência causal em replay retrospectivo; JSONL local é opcional
  para testes de chunks, não entrega de aquisição física. LSL foi removido.
- Expor qualidade do sinal e abster-se quando a janela não é confiável pela
  heurística atual.

## Uso não pretendido

- Diagnóstico ou monitorização clínica autônoma.
- Controle ou ajuste de anestésicos.
- Substituição do anestesiologista, de monitor aprovado ou de protocolo clínico.
- Generalização para outra marca, montagem, unidade, população ou fármaco sem
  validação específica.

## Dados e rótulo

O acervo demonstrativo contém 24 arquivos anônimos do dataset [EEG and BIS raw
data](https://doi.org/10.6084/m9.figshare.5589841.v1), publicado no Figshare sob
CC BY 4.0. O corpus é descrito com EEG frontal a 128 Hz e BIS a cada 5 s.
O `case24` é baixado, mas fica fora do checkpoint padrão porque a auditoria
encontra escala bruta incompatível (0–4095) sem metadado seguro para conversão.

O alvo é o BIS publicado, um índice processado de monitor. Ele é uma referência
operacional e não uma verdade clínica universal; atrasos, artefatos, EMG, drogas,
estímulo cirúrgico e contexto do paciente podem alterar sua interpretação.

A separação Figshare é por caso/arquivo; `subject_id` não está disponível no
manifesto e independência por pessoa não está comprovada. No VitalDB, o
agrupamento usa `subject_id` quando disponível, com fallback por caso da fonte.

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
  exploratória. É uma API experimental não integrada à CLI de treino/inferência;
  não substitui o checkpoint acima nem oferece incerteza validada ou calibrada.
- A receita histórica persistida registra dez épocas, seed 42, batch 128,
  learning rate 0,001 e weight decay 0,0001. Recursos posteriores do loop
  (early stopping, scheduler, clipping de gradiente ou mixed precision) não
  devem ser atribuídos ao treino histórico apenas por existirem no código atual;
  os JSON históricos não comprovam sua utilização.
- Suavização: EWMA somente na apresentação em fluxo; o valor bruto permanece
  disponível.
- Abstenção: qualidade heurística abaixo de 0,20 produz `stage="abstain"` e não
  mantém um BIS antigo visível. SQI é uma heurística de sinal, não probabilidade
  de acerto, confiança clínica ou calibração da incerteza.

Com `label_offset_seconds=0`, o alvo offline corresponde ao início da janela;
a emissão online só ocorre ao final de sua aquisição (5 s). Filtros causais não
eliminam essa diferença temporal nem o atraso do próprio monitor BIS. A imputação
linear offline pode usar amostras posteriores à lacuna, portanto os resultados
offline não demonstram equivalência causal ponta a ponta com o stream.

## Resultados históricos do artefato congelado

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

Uma avaliação cruzada de dataset exploratória, sem retreino, nos 15 casos VitalDB compatíveis
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

O pool elegível misto contém 23 Figshare + 10 VitalDB. O candidato `fixed`
exclui os cinco casos do holdout Figshare histórico: seu desenvolvimento contém
18 Figshare + 10 VitalDB = 28 casos/grupos, divididos em 16/6/6 para
treino/validação/teste interno; somente 12 Figshare + 4 VitalDB entram no ajuste
dos pesos. No holdout Figshare fixo, obteve MAE 6,69 e Pearson 0,818; no holdout
histórico VitalDB, MAE 8,60 e Pearson 0,688. Estes 15 casos não entram no ajuste,
mas já foram reutilizados para comparação após exposição ao domínio VitalDB;
não constituem validação externa confirmatória. A melhora sugere adaptação ao
domínio observado, não transporte comprovado a outra fonte. O candidato não
foi promovido ao checkpoint ativo. O protocolo
completo, gates e hashes estão em [`docs/mixed_corpus.md`](mixed_corpus.md).

## Riscos e validação necessária fora do TCC

O TCC termina na análise de arquivos existentes e replay, sem etapa de hardware
ou estudo prospectivo. As condições abaixo limitam qualquer projeto separado;
não são um plano de execução nem promessa de integração desta entrega.

Antes de qualquer contato com pacientes, o laboratório precisa confirmar taxa,
unidade em microvolt, faixa nominal/saturação, processamento/ganho, referência,
montagem, canal, timestamps e política de perda no hardware
alvo. A ficha mínima pode ser verificada sem EEG com
`uv run brainsniffer validate-intake --metadata-file ...`; no stream, o gate
equivalente é `--require-intake`. Isso só libera a bancada técnica. Também precisa fazer revisão por anestesiologista, ética, proteção de dados,
análise de riscos, validação externa e avaliação regulatória aplicável.

Um eventual projeto independente precisaria incluir um segundo centro/aparelho, referência clínica
complementar ao BIS, alinhamento do atraso do monitor, SQI anotado, desempenho por
paciente/fármaco/qualidade e uma política de abstenção calibrada.
O registro histórico da proposta por etapas, incluindo bancada, validação externa
travada e modo sombra (fora do escopo definitivo), está em [`docs/_archive/prospective_protocol.md`](_archive/prospective_protocol.md) e
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
- Contrato JSONL local opcional e histórico LSL: `docs/_archive/live_acquisition.md`.
- Proposta histórica fora do TCC: `docs/_archive/prospective_protocol.md`.

### Reprodução da avaliação, sem substituir o artefato

Exemplo proposto, não executado nesta revisão documental; requer ambiente,
arquivos de entrada e par `.pt`/`.json` compatíveis. Grave a nova avaliação em
outro caminho, sem sobrescrever a evidência histórica:

```bash
uv run brainsniffer evaluate --checkpoint models/brainsniffer_cnn.pt \
  --report /tmp/brainsniffer_figshare_recheck.json \
  --bootstrap-samples 1000 --bootstrap-seed 42
```

### Receita de treino histórico versus código atual

Os campos `training_config`, `preprocess_config`, `split`, `history` e
`environment` dos JSON descrevem os experimentos históricos. Executar `train`
com dez épocas no código atual é um novo experimento, não garantia de recuperar
os pesos históricos. Uma reprodução do treino requer também revisão/ambiente
históricos e todas as decisões de seleção; novos pesos devem usar outro nome.
Nenhum treino, teste, avaliação numérica ou build foi executado nesta revisão.
Relatórios preservados não provam o funcionamento da implementação atual.

O comando `evaluate` não treina novamente: ele lê a divisão por caso salva no
checkpoint, reconstrói as janelas com a configuração persistida e mostra as
métricas armazenadas ao lado das métricas recalculadas.

As decisões metodológicas são fundamentadas na literatura de DoA/CNN, no corpus
publicado e nas limitações documentadas de índices pEEG; a documentação LSL
fundamentou somente o transporte histórico removido, não o escopo atual;
as fontes completas estão no [README](../README.md) e em
[`docs/decisions.md`](decisions.md).
