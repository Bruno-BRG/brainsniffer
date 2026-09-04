# BrainSniffer

Protótipo de pesquisa em Python 3.12 para estimar, a partir de EEG frontal, um valor contínuo semelhante ao BIS e um estágio discreto relacionado ao BIS. O modelo inicial é uma CNN 1-D compacta; o sistema também inclui aquisição do dataset público, pré-processamento, divisão por paciente/caso, treino, replay em fluxo e dashboard Dash + Plotly.

> **Aviso clínico:** este repositório não é um dispositivo médico e não deve orientar, sozinho, dose anestésica ou conduta em uma cirurgia. O “ground truth” do MVP é o BIS publicado, que é um índice processado do monitor e não uma medida absoluta de consciência. Qualquer uso clínico exigiria protocolo prospectivo, aprovação ética, validação externa multicêntrica, análise de segurança, gestão de alarmes e avaliação regulatória.

## O que já existe

- Download sem credenciais do dataset [EEG and BIS raw data](https://doi.org/10.6084/m9.figshare.5589841.v1), publicado no Figshare sob CC BY 4.0, com conferência de tamanho e MD5 do manifesto oficial.
- Download seletivo de casos do [VitalDB Open Dataset](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md), com `subjectid` do mapa público para agrupar reoperações quando disponível.
- Manifesto de corpus misto Figshare + VitalDB com gates de finitude, lacunas, BIS válido, qualidade por janela, hashes e separação entre treino e holdout externo.
- A normalização VitalDB preserva unidade declarada, nomes/IDs dos tracks e origem do dataset; amplitude e montagem não são convertidas silenciosamente.
- Leitura de arquivos MATLAB v7.3 (`EEG` e `bis`) e fallback para MAT clássico.
- Janelas não sobrepostas de 5 s a 128 Hz, alinhadas ao ponto BIS correspondente.
- Filtro passa-banda, notch de 50 Hz, tratamento de amostras não finitas e heurística de qualidade exposta na interface.
- Gate padrão de qualidade 0,20 no treino e na inferência; o corpus baixado tem 24 arquivos, mas o `case24` é reprovado pela auditoria de escala e não entra no checkpoint padrão.
- CNN 1-D com regressão limitada a 0–100 e mapeamento para quatro estágios: `deep`, `general`, `light`, `awake`.
- Divisão por caso antes da avaliação para evitar que janelas vizinhas do mesmo paciente vazem entre conjuntos.
- Replay simulado com buffer, emissão configurável e suavização exponencial para demonstrar o caminho de inferência em fluxo.
- Entrada ao vivo vendor-neutral via Lab Streaming Layer (LSL), com seleção de stream/canal, timestamps, conversão de taxa e saída JSON; o driver do equipamento ou bridge LSL continua sendo responsabilidade do laboratório.
- Baseline espectral com potência por banda, entropia, frequência de borda, RMS e line length para comparar a CNN com uma referência interpretável.
- Benchmark de latência do caminho de streaming, com p50/p95 por chunk e fração do orçamento de tempo real.
- Dashboard Dash + Plotly para comparar CNN contra BIS e reproduzir um caso gravado com cursor temporal, play/pause, velocidade, EEG causal, saída bruta/suavizada, estágio e qualidade.
- Aba **Corpus** com composição por fonte, mapa finitude × janelas aproveitáveis e comparação do checkpoint ativo contra o candidato misto.
- Interface Streamlit legada em [`app.py`](app.py), mantida para compatibilidade com os fluxos de download, treino e LSL; a publicação principal usa [`dash_app.py`](dash_app.py).
- Artigo escrito do TCC em LaTeX no formato SBC em [`docs/tcc_brainsniffer.tex`](docs/tcc_brainsniffer.tex), com referências em [`docs/referencias.bib`](docs/referencias.bib), arquivos do template em [`docs/latex/`](docs/latex/) e PDF compilado em [`docs/tcc_brainsniffer.pdf`](docs/tcc_brainsniffer.pdf).
- Rascunho técnico de apoio em [`docs/article.md`](docs/article.md).
- Ledger que liga fontes, hipóteses, decisões e limites em [`docs/source_ledger.md`](docs/source_ledger.md).
- Catálogo de datasets e compatibilidade de cada fonte em [`docs/data_catalog.md`](docs/data_catalog.md).
- Protocolo e resultado do corpus misto em [`docs/mixed_corpus.md`](docs/mixed_corpus.md).
- Model card com uso pretendido, métricas, riscos e limites em [`docs/model_card.md`](docs/model_card.md).
- Guia da avaliação externa exploratória seletiva com VitalDB em [`docs/vitaldb_external_validation.md`](docs/vitaldb_external_validation.md).
- Ficha de entrada e aceite para equipamento EEG real em [`docs/real_eeg_intake.md`](docs/real_eeg_intake.md).
- Protocolo de avanço da bancada ao modo sombra prospectivo em [`docs/prospective_protocol.md`](docs/prospective_protocol.md).
- A aba Dados também oferece download e avaliação exploratória VitalDB sem retreino.
- A aba EEG ao vivo (LSL) mostra a auditoria da sessão e permite baixar um relatório
  JSON de metadados sem armazenar o EEG bruto.
- A mesma aba valida a ficha do equipamento sem conectar e permite baixar o
  manifesto JSON que será reutilizado na sessão.
- A ficha JSON do equipamento pode ser validada antes da bancada com
  `validate-intake`; `--require-intake` transforma essa ficha em gate antes da
  inferência em JSON/LSL.
- O gate exige unidade em microvolt compatível com o pré-processamento, faixa
  nominal/saturação e processamento/ganho documentados; não faz conversão de
  escala silenciosa.
- Na interface LSL, a rejeição por auditoria durante a captura e no encerramento vem ativada por padrão, mas pode
  ser desmarcada explicitamente para caracterização exploratória.
- Streams podem carregar manifesto de unidade, posição do canal, referência e
  montagem; `--require-metadata` bloqueia a inferência quando esses campos não
  estiverem documentados.
- No LSL, esses campos também podem ser lidos do descritor XML do stream; flags
  explícitas complementam ou confirmam o descritor e conflitos são rejeitados.
- A aba EEG ao vivo exige timestamps válidos por amostra, além do gate de
  metadados escolhido na tela.
- Para manter o manifesto versionado junto ao estudo, preencha o modelo
  [`examples/stream_metadata.template.json`](examples/stream_metadata.template.json)
  e passe `--metadata-file` aos comandos `audit-json`, `stream-json` ou
  `stream-lsl`. Conflitos entre arquivo, flags, chunks JSON e descritor LSL fazem
  a sessão falhar de forma explícita.
- Matriz de requisitos, evidências e lacunas em [`docs/project_status.md`](docs/project_status.md).

## Instalação

Com `uv`:

```bash
uv sync --extra dev
```

Para ativar a entrada LSL opcional mantendo as ferramentas de desenvolvimento:

```bash
uv sync --extra dev --extra live
```

O projeto fixa Python 3.12 em [`.python-version`](.python-version) e declara `requires-python = ">=3.12,<3.13"`. Em uma máquina sem Python 3.12, o `uv` pode baixar um interpretador isolado; não misture o ambiente do projeto com o Python do sistema.

## Uso rápido

```bash
# baixar todos os 24 casos para data/raw/
uv run brainsniffer download-data

# avaliação externa exploratória seletiva; repita --case para outros casos
uv run brainsniffer download-vitaldb --case 1 --out data/vitaldb

# baixar casos VitalDB destinados ao pool de treino (não use o diretório externo)
uv run brainsniffer download-vitaldb --case 18 --case 20 --out data/vitaldb_train

# auditar Figshare + VitalDB de treino e manter data/vitaldb como holdout congelado
uv run brainsniffer build-corpus --out reports/corpus_manifest.json

# treinar candidato com amostragem balanceada por grupo e por fonte
uv run brainsniffer train-corpus --manifest reports/corpus_manifest.json \
  --checkpoint models/brainsniffer_corpus_fixed.pt

# inspecionar os casos presentes
uv run brainsniffer inspect-data

# treino inicial; aumente as épocas apenas depois de validar o protocolo
uv run brainsniffer train --epochs 10 --min-quality 0.2 --checkpoint models/brainsniffer_cnn.pt

# se um protocolo medir atraso do BIS, registre-o explicitamente (segundos)
# uv run brainsniffer train --label-offset-seconds 5 --checkpoint models/offset.pt

# recomputar as métricas do checkpoint nos mesmos casos de teste e salvar auditoria
uv run brainsniffer evaluate --checkpoint models/brainsniffer_cnn.pt \
  --report reports/figshare_holdout_evaluation.json \
  --bootstrap-samples 1000 --bootstrap-seed 42

# recalcular o holdout salvo de um candidato treinado pelo manifesto
uv run brainsniffer evaluate --manifest reports/corpus_manifest.json \
  --checkpoint models/brainsniffer_corpus_fixed.pt

# sensibilidade exploratória ao alinhamento do BIS, sem retreinar
uv run brainsniffer evaluate-offset --data-dir data/raw \
  --checkpoint models/brainsniffer_cnn.pt \
  --offset-seconds -20 -15 -10 -5 0 5 10 15 20 \
  --report reports/offset_sensitivity.json

# avaliar arquivo externo normalizado sem retreinar
uv run brainsniffer evaluate-external --checkpoint models/brainsniffer_cnn.pt \
  --case data/vitaldb/vitaldb_case1.npz \
  --case data/vitaldb/vitaldb_case2.npz

# avaliar todos os VitalDB normalizados no diretório, sem esquecer casos
uv run brainsniffer evaluate-external --data-dir data/vitaldb \
  --checkpoint models/brainsniffer_cnn.pt --bootstrap-samples 1000 \
  --bootstrap-seed 42 --report reports/vitaldb_external_validation.json

# preflight de um bridge JSONL antes de alimentar a CNN
cat chunks.jsonl | uv run brainsniffer audit-json

# replay simulado de um caso como fluxo de EEG
uv run brainsniffer replay --case 1 --checkpoint models/brainsniffer_cnn.pt

# replay de um caso VitalDB normalizado como fluxo de EEG; falha fechado se houver NaN/Inf
uv run brainsniffer replay --case 1 --data-dir data/vitaldb \
  --checkpoint models/brainsniffer_cnn.pt

# comparar com a baseline espectral (split por caso)
uv run brainsniffer benchmark-baseline

# análise de sensibilidade com 5 folds por paciente
uv run brainsniffer benchmark-baseline --folds 5

# medir a latência do caminho de inferência
uv run brainsniffer benchmark-latency --iterations 30

# dashboard web local (produção/publicação)
uv run python dash_app.py

# servidor equivalente ao usado na VPS
uv run gunicorn --bind 0.0.0.0:8501 --workers 1 --threads 4 --timeout 120 dash_app:server
```

O replay da CLI simula o caminho online e, por isso, rejeita amostras `NaN`/`Inf`
antes do filtro causal. A trajetória e o replay visual do Dash são inspeções
retrospectivas de arquivos gravados: quando um VitalDB tem pontos ausentes, a
interface interpola esses pontos somente para a simulação e mantém o score de
qualidade calculado no sinal original. Essa conveniência não altera o estimador
online nem valida uma aquisição ao vivo.

### Ficha do equipamento

Antes de conectar um equipamento físico, copie e preencha o manifesto versionado:

```bash
cp examples/stream_metadata.template.json examples/stream_metadata.local.json
uv run brainsniffer validate-intake \
  --metadata-file examples/stream_metadata.local.json
```

O comando só retorna sucesso quando fabricante, modelo, firmware/software, bridge,
taxa, unidade, canal, referência, montagem, faixa nominal e processamento/ganho
estão documentados. A unidade precisa ser microvolt compatível com o modelo;
conversões devem ocorrer no bridge. `ready_for_bench` significa pronto para o
preflight técnico; não significa validação clínica.
Para aplicar o mesmo gate ao stream:

```bash
meu-bridge-do-fabricante | uv run brainsniffer stream-json \
  --checkpoint models/brainsniffer_cnn.pt \
  --metadata-file examples/stream_metadata.local.json \
  --require-metadata --require-intake --require-timestamps --fail-on-audit \
  --report registros/sessao.json
```

### Entrada ao vivo

Se o laboratório publicar o EEG como LSL, o processo pode consumir o canal escolhido:

```bash
uv run brainsniffer stream-lsl \
  --checkpoint models/brainsniffer_cnn.pt \
  --metadata-file examples/stream_metadata.synthetic.json \
  --stream-type EEG \
  --channel 0 \
  --require-metadata
```

Para integrar um driver próprio sem LSL, o modo JSON aceita uma linha por chunk pela entrada padrão:

```json
{"samples": [0.1, 0.2, 0.0], "sampling_rate": 128, "timestamps": [1710000000.0, 1710000000.0078125, 1710000000.015625], "metadata": {"unit": "uV", "channel_name": "Fpz", "reference": "linked ears", "montage": "frontal referenced"}}
```

```bash
cat chunks.jsonl | uv run brainsniffer stream-json --checkpoint models/brainsniffer_cnn.pt

# bloqueio explícito para aquisição sem manifesto de sinal completo
cat chunks.jsonl | uv run brainsniffer stream-json \
  --checkpoint models/brainsniffer_cnn.pt --require-metadata
```

Para gerar um relatório de sessão sem guardar o EEG bruto:

```bash
cat chunks.jsonl | uv run brainsniffer stream-json \
  --checkpoint models/brainsniffer_cnn.pt \
  --report registros/sessao.json
```

O relatório versionado inclui o hash do checkpoint, ambiente Python, configuração
de pré-processamento, stride, gate de metadata e diagnósticos do sinal; ele não
contém as amostras EEG.
Ele também declara explicitamente `intended_use=research_only`, sem apoio à
decisão clínica e sem controle de entrega anestésica.
Com `--fail-on-audit`, uma captura finita ou contínua termina com erro assim que a
auditoria rejeita uma amostra, lacuna, timestamp, qualidade ou condição estrutural;
o relatório parcial é preservado. Sem essa opção, o modo de exploração continua
disponível e as janelas ruins geram abstenção.

Quando informado, `metadata.sampling_rate` também é comparado à taxa efetiva dos
chunks e uma divergência encerra o stream antes da inferência.
O critério de lacuna temporal é configurável com `--max-gap-factor` nos três
comandos de stream/preflight e é salvo no relatório de sessão.

No LSL, `--stale-timeout` controla o silêncio da fonte. O padrão de engenharia
é 2 s: com `--fail-on-audit`, a sessão termina e preserva um relatório parcial;
sem essa flag, a última estimativa é invalidada com `ABSTAIN`, a suavização é
limpa e uma janela nova precisa ser preenchida. Isso não substitui a auditoria
de timestamps, finitude, saturação e linha plana.

Para testar a cadeia LSL em bancada antes do equipamento, use o publisher
sintético documentado (somente pesquisa):

```bash
uv run python examples/lsl_synthetic_publisher.py --duration 30
uv run brainsniffer stream-lsl --stream-name BrainSnifferSyntheticEEG \
  --duration 8 --require-metadata --require-timestamps --fail-on-audit \
  --checkpoint models/brainsniffer_cnn.pt \
  --report reports/lsl_synthetic_session.json
```

Para exercitar também o gate completo da ficha de equipamento, use o manifesto
explicitamente sintético (ele não descreve um dispositivo clínico):

```bash
uv run python examples/lsl_synthetic_publisher.py --rate 256 --duration 25
uv run brainsniffer stream-lsl --stream-name BrainSnifferSyntheticEEG \
  --metadata-file examples/stream_metadata.bench.synthetic.json \
  --duration 8 --stale-timeout 2 --require-metadata --require-intake \
  --require-timestamps --fail-on-audit \
  --checkpoint models/brainsniffer_cnn.pt \
  --report reports/lsl_synthetic_intake_session.json
```

Esse smoke deve retornar `ready_for_bench=true` apenas porque a ficha sintética
está completa; ele não representa aceite de um EEG físico.

O JSON deve informar `sampling_rate` ou receber `--source-rate`; se ambos faltarem, o processo assume 128 Hz e avisa no stderr. A taxa deve permanecer constante durante o stream. Taxas diferentes são convertidas pelo mesmo resampler stateful usado no LSL, e a CLI libera a cauda ao terminar um JSONL finito.

O modelo foi treinado a 128 Hz. O adaptador LSL converte outras taxas com um resampler polifásico que mantém sobreposição/estado entre chunks e segura o pequeno atraso da cauda do filtro; esse caminho ainda precisa ser medido no hardware real antes de um estudo prospectivo. O LSL fornece uma camada de transporte/sincronização, não valida o sinal e não transforma um monitor de pesquisa em equipamento clínico. Consulte a [documentação oficial do LSL](https://labstreaminglayer.readthedocs.io/info/intro.html).

`evaluate-external` também mostra métricas por caso, diagnósticos de finitude do
sinal bruto e um bootstrap exploratório por agrupamento de caso. A CLI usa seed
42 por padrão, aceita `--bootstrap-seed` e registra a seed e o número de
reamostragens no JSON para permitir repetição auditável. `audit-json` é um
preflight de engenharia: ele reporta taxa, finitude, saturação, linha plana,
timestamps e lacunas. O score não é uma confiança clínica e um relatório `ok`
não autoriza uso em pacientes.
Quando o relatório do `audit-json` é `ok=false`, o comando termina com código de
saída 1, permitindo que o gate seja usado em automação de bancada/CI.

Novos checkpoints registram o manifesto SHA-256/tamanho dos arquivos de entrada;
`evaluate` verifica esses arquivos antes de recalcular métricas e sinaliza quando
um checkpoint de smoke test tem escopo de janelas diferente da avaliação completa.

Para um smoke test rápido, use `--max-windows 512`. O padrão de treino é `--min-quality 0.2`; use `--min-quality 0` somente para auditar o bruto, pois pode reintroduzir janelas saturadas. Os arquivos `.mat` e checkpoints ficam fora do Git por padrão.

## Interpretação do rótulo

O dataset fornece EEG e BIS, não um diagnóstico clínico independente de consciência. O código preserva a regressão BIS e cria uma taxonomia de pesquisa com intervalos semiabertos: `[0,40) deep`, `[40,60) general`, `[60,80) light`, `[80,100] awake`. Essa taxonomia acompanha a literatura que usa quatro faixas de BIS, mas a apresentação do resultado deve sempre mostrar o valor, o estágio, a qualidade do sinal e a origem do rótulo.

## Referências que orientam as decisões

1. Ma, L. (2017). *EEG and BIS raw data*. Figshare. [DOI e dataset](https://doi.org/10.6084/m9.figshare.5589841.v1).
2. Li et al. (2022). *Multiscale depth of anaesthesia prediction for surgery using frontal cortex electroencephalography*. [PMC9160818](https://pmc.ncbi.nlm.nih.gov/articles/PMC9160818/). Descreve EEG de 128 Hz, BIS a cada 5 s e quatro estados derivados do BIS.
3. Lee et al. (2022). *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients*. [PMC9178032](https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032/). Descreve o corpus perioperatório externo e sua estrutura de sinais.
4. Yang et al. (2020). *A Real-Time Depth of Anesthesia Monitoring System Based on Deep Neural Network...*. [PubMed 32746339](https://pubmed.ncbi.nlm.nih.gov/32746339/). Justifica investigar uma CNN compacta para inferência em dispositivo de baixo custo, sem transformar o resultado em alegação clínica.
5. Shi et al. (2023). *Estimating the Depth of Anesthesia from EEG Signals Based on a Deep Residual Shrinkage Network*. [PubMed 36679805](https://pubmed.ncbi.nlm.nih.gov/36679805/). Exemplo de regressão de índice de monitor a partir de quatro canais.
6. Hajat, Ahmad & Andrzejowski (2017). *The role and limitations of EEG-based depth of anaesthesia monitoring*. [PubMed 28044337](https://pubmed.ncbi.nlm.nih.gov/28044337/). Fundamenta a cautela com BIS, paralisação e contexto clínico.
7. *Open Reimplementation of the BIS Algorithms for Depth of Anesthesia* (2022). [PMC9481655](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481655/). Referência para distinguir índice proprietário, sinais intermediários e validação independente.
8. FDA, K202621. [Indications and comparison of EEG-based proprietary indices](https://www.accessdata.fda.gov/cdrh_docs/pdf20/K202621.pdf). Registro regulatório que descreve índices 0–100 e o intervalo de referência 40–60 como auxílio de monitoramento.
9. Kothe et al. (2025). *The Lab Streaming Layer for Synchronized Multimodal Recording*. [PMC12434378](https://pmc.ncbi.nlm.nih.gov/articles/PMC12434378/). Fundamenta o uso de timestamps, descoberta de streams e sincronização como camada de pesquisa.

As decisões detalhadas, as hipóteses e os itens que ainda dependem de validação estão em [`docs/decisions.md`](docs/decisions.md).
O contrato para uma fonte de EEG real está em [`docs/live_acquisition.md`](docs/live_acquisition.md); antes de usar um equipamento, preencha também a [`ficha de entrada de EEG real`](docs/real_eeg_intake.md).
O protocolo para avançar de replay para bancada, validação externa travada e modo sombra está em [`docs/prospective_protocol.md`](docs/prospective_protocol.md).

## Próxima etapa científica

O próximo experimento deve ser fechado antes de otimizar a arquitetura: pré-registrar a divisão por caso, comparar CNN crua contra baseline espectral/entropia, reportar MAE/RMSE/correlação e desempenho por estágio, medir latência no hardware alvo e testar generalização em outro centro/dataset. Não devemos publicar métricas de janelas aleatórias como se fossem generalização a pacientes novos.

O adaptador VitalDB baixa somente os tracks `BIS/EEG1_WAV` e `BIS/BIS` do caso solicitado e grava um `.npz` normalizado em 128 Hz. A licença/termos do VitalDB devem ser lidos antes do uso; esses arquivos não são misturados automaticamente ao treino Figshare.
