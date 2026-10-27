# BrainSniffer

[![CI](https://github.com/Bruno-BRG/brainsniffer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Bruno-BRG/brainsniffer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Protótipo de pesquisa em Python 3.12 para estimar, a partir de EEG frontal, um valor contínuo semelhante ao BIS e um estágio discreto relacionado ao BIS. O modelo inicial é uma CNN 1-D compacta; o sistema também inclui aquisição do dataset público, pré-processamento, divisão por paciente/caso, treino e replay em fluxo. A interface é a linha de comando (CLI); o projeto não inclui interface web.

> **Aviso clínico:** este repositório não é um dispositivo médico e não deve orientar, sozinho, dose anestésica ou conduta em uma cirurgia. O “ground truth” do MVP é o BIS publicado, que é um índice processado do monitor e não uma medida absoluta de consciência. Qualquer uso clínico exigiria protocolo prospectivo, aprovação ética, validação externa multicêntrica, análise de segurança, gestão de alarmes e avaliação regulatória.

## Estrutura do repositório

O mapa completo das pastas está em [`OVERVIEW.md`](OVERVIEW.md#estrutura-do-repositório); em resumo:

| Caminho | O que é |
|---|---|
| `src/brainsniffer/` | Código do produto: CLI e config + `data/` (leitura/pré-processamento), `nets/` (arquiteturas CNN) e `pipeline/` (treino, métricas, baseline, replay/stream) |
| `models/` | Artefatos treinados: `.pt` (pesos, fora do Git) + `.json` (ficha/hash, versionada) |
| `reports/` | JSONs de evidência do artigo (holdouts, bootstrap, Pk, calibração); congelados |
| `data/` | Dados locais baixados (Figshare/VitalDB); não redistribuídos, fora do Git |
| `tests/` | Testes do pipeline; rodam no CI junto com o `ruff` |
| `scripts/` | Build do PDF do TCC + reanálises one-off que geram `reports/` |
| `examples/` | Fichas de manifesto de sinal para bancada/stream (template + sintéticos) |
| `docs/` | Artigo do TCC, notas, submissão JHI e arquivo morto |

## Recursos e documentação de apoio

<details>
<summary>Inventário técnico e documentos do projeto</summary>


- Download sem credenciais do dataset [EEG and BIS raw data](https://doi.org/10.6084/m9.figshare.5589841.v1), publicado no Figshare sob CC BY 4.0, com conferência de tamanho e MD5 do manifesto oficial.
- Download seletivo de casos do [VitalDB Open Dataset](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md), com `subjectid` do mapa público para agrupar reoperações quando disponível.
- Manifesto de corpus misto Figshare + VitalDB com gates de finitude, lacunas, BIS válido, qualidade por janela, hashes e separação entre treino e holdout externo.
- A normalização VitalDB preserva unidade declarada, nomes/IDs dos tracks e origem do dataset; amplitude e montagem não são convertidas silenciosamente.
- Leitura de arquivos MATLAB v7.3 (`EEG` e `bis`) e fallback para MAT clássico.
- Janelas não sobrepostas de 5 s a 128 Hz, alinhadas ao ponto BIS correspondente.
- Filtro passa-banda, notch de 50 Hz, tratamento de amostras não finitas e heurística de qualidade exposta na CLI e nos relatórios.
- Gate padrão de qualidade 0,20 no treino e na inferência; o corpus baixado tem 24 arquivos, mas o `case24` é reprovado pela auditoria de escala e não entra no checkpoint padrão.
- CNN 1-D com regressão limitada a 0–100 e mapeamento para quatro estágios: `deep`, `general`, `light`, `awake`.
- Divisão por caso antes da avaliação para evitar que janelas vizinhas do mesmo paciente vazem entre conjuntos.
- Replay simulado com buffer, emissão configurável e suavização exponencial para demonstrar o caminho de inferência em fluxo.
- Entrada experimental por bridge JSONL na CLI, com timestamps, conversão de taxa e saída JSON; o driver do equipamento e o bridge continuam sendo responsabilidade do laboratório.
- Baseline espectral com potência por banda, entropia, frequência de borda, RMS e line length para comparar a CNN com uma referência interpretável.
- Benchmark de latência do caminho de streaming, com p50/p95 por chunk e fração do orçamento de tempo real.
- Download (`download-data`, `download-vitaldb`), treino (`train`, `train-corpus`), bridge JSONL (`stream-json`, `audit-json`) e ficha do equipamento (`validate-intake`) são operações da CLI.
- Artigo escrito do TCC em LaTeX no formato SBC em [`docs/artigo/tcc_brainsniffer.tex`](docs/artigo/tcc_brainsniffer.tex), com referências em [`docs/artigo/referencias.bib`](docs/artigo/referencias.bib), arquivos do template em [`docs/artigo/latex/`](docs/artigo/latex/) e PDF compilado em [`docs/artigo/tcc_brainsniffer.pdf`](docs/artigo/tcc_brainsniffer.pdf).
- Índice da documentação em [`docs/README.md`](docs/README.md): artigo, notas, submissão e arquivo morto.
- Rascunho técnico de apoio em [`docs/notas/article.md`](docs/notas/article.md).
- Ledger que liga fontes, hipóteses, decisões e limites em [`docs/notas/source_ledger.md`](docs/notas/source_ledger.md).
- Catálogo de datasets e compatibilidade de cada fonte em [`docs/notas/data_catalog.md`](docs/notas/data_catalog.md).
- Protocolo e resultado do corpus misto em [`docs/notas/mixed_corpus.md`](docs/notas/mixed_corpus.md).
- Model card com uso pretendido, métricas, riscos e limites em [`docs/notas/model_card.md`](docs/notas/model_card.md).
- Guia da avaliação externa exploratória seletiva com VitalDB em [`docs/notas/vitaldb_external_validation.md`](docs/notas/vitaldb_external_validation.md).
- Ficha de entrada e aceite para equipamento EEG real em [`docs/_archive/real_eeg_intake.md`](docs/_archive/real_eeg_intake.md).
- Protocolo de avanço da bancada ao modo sombra prospectivo em [`docs/_archive/prospective_protocol.md`](docs/_archive/prospective_protocol.md).
- Download e avaliação exploratória VitalDB sem retreino são feitos pela CLI (`download-vitaldb` e `evaluate-external`).
- A CLI `stream-json --report` registra a auditoria da sessão em JSON sem armazenar
  o EEG bruto.
- Preencha o manifesto JSON do equipamento e valide-o com `validate-intake`
  antes de reutilizá-lo na sessão via `--metadata-file`.
- A ficha JSON do equipamento pode ser validada antes da bancada com
  `validate-intake`; `--require-intake` transforma essa ficha em gate antes da
  inferência em JSONL.
- O gate exige unidade em microvolt compatível com o pré-processamento, faixa
  nominal/saturação e processamento/ganho documentados; não faz conversão de
  escala silenciosa.
- Na CLI de streaming, use explicitamente `--fail-on-audit` para interromper
  a captura quando a auditoria rejeitar a sessão; essa flag não é ativada por padrão.
- Streams podem carregar manifesto de unidade, posição do canal, referência e
  montagem; `--require-metadata` bloqueia a inferência quando esses campos não
  estiverem documentados.
- Na CLI, `--require-timestamps` exige timestamps válidos por amostra, além
  dos gates explícitos `--require-metadata` e `--require-intake`.
- Para manter o manifesto versionado junto ao estudo, preencha o modelo
  [`examples/stream_metadata.template.json`](examples/stream_metadata.template.json)
  e passe `--metadata-file` aos comandos `audit-json` ou `stream-json`.
  Conflitos entre arquivo, flags e chunks JSON fazem a sessão falhar de forma explícita.
- Matriz de requisitos, evidências e lacunas em [`docs/notas/project_status.md`](docs/notas/project_status.md).

</details>

## Instalação

Com `uv`:

```bash
uv sync --extra dev
```

O projeto fixa Python 3.12 em [`.python-version`](.python-version) e declara `requires-python = ">=3.12,<3.13"`. Em uma máquina sem Python 3.12, o `uv` pode baixar um interpretador isolado; não misture o ambiente do projeto com o Python do sistema.

## Citação e reprodutibilidade

Ao usar o BrainSniffer em trabalho acadêmico, cite o software com os metadados de
[`CITATION.cff`](CITATION.cff), também disponíveis em **Cite this repository** no
GitHub. O código-fonte é distribuído sob a [licença MIT](LICENSE); datasets e
artefatos de terceiros permanecem sujeitos às licenças e aos termos de suas fontes.

Para reconstruir o ambiente declarado e executar as verificações automatizadas:

```bash
uv sync --locked --extra dev
uv run --locked --extra dev pytest -q
```

Os comandos dos experimentos estão em [Uso rápido](#uso-rápido). A proveniência,
as hipóteses e os limites de interpretação são registrados em
[`docs/notas/source_ledger.md`](docs/notas/source_ledger.md) e
[`docs/notas/decisions.md`](docs/notas/decisions.md).

### Limites da reprodução

A instalação consome `uv.lock` e deve falhar se ele divergir do projeto, em vez
de resolver livremente com `pip install .`. O lock atual aponta Torch do PyPI,
incluindo dependências CUDA no Linux: o ambiente não é CPU-only nem tem garantia
de tamanho reduzido, e a disponibilidade dos wheels ainda precisa de validação
de integração.

Trate checkpoints e qualquer artefato serializado com pickle como entradas
**somente de origem confiável**. Pickle pode executar código ao desserializar;
não carregue uploads ou caminhos fornecidos por terceiros. Esta orientação não
atesta nem antecipa o contrato de segurança de cada loader.

Para uma apresentação, congele dados, checkpoint, relatórios e configuração antes
da sessão, registre seus hashes e mantenha-os somente para leitura. Não treine,
baixe dados ou substitua o modelo durante a sessão; prepare uma nova versão e
repita a verificação fora dela. Isto é um requisito operacional, não uma alegação
de que o filesystem já imponha imutabilidade. O
[checklist de aceite da demonstração](docs/_archive/research/demo_acceptance_checklist.md)
foi escrito para a interface web removida e permanece apenas como registro
histórico: os cenários que ele lista seguem **não executados** nesta revisão.
Correções paralelas e CI não são declarados aprovados por este texto.

## Uso rápido

### CLI — preparação e experimentos

Os comandos abaixo são opções independentes, **não uma sequência obrigatória**.
Download acessa a rede; treino e avaliações podem escrever artefatos. Execute
apenas a operação necessária, fora da sessão de apresentação.

<details>
<summary>Comandos de download, treino, avaliação e benchmark</summary>

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

```

</details>

O replay da CLI simula o caminho online e, por isso, rejeita amostras `NaN`/`Inf`
antes do filtro causal. O replay e as demais inspeções retrospectivas de arquivos
gravados preservam as lacunas do sinal: pontos ausentes permanecem ausentes, e o
score de qualidade continua o do sinal original. Essa conveniência não altera o
estimador online nem valida uma aquisição ao vivo.

### Experimentos de arquitetura e treino

O checkpoint padrão continua usando `Conv1DDepthEstimator`, preservando a
compatibilidade dos experimentos existentes. Para novos estudos, o pacote
também exporta `RobustConv1DDepthEstimator`, uma CNN residual com GroupNorm,
convoluções depthwise-separable, contexto dilatado e `predict_with_uncertainty`
via MC Dropout. A configuração de treino registra métricas por época e oferece
early stopping, redução adaptativa do learning rate, clipping de gradiente,
reprodutibilidade determinística e mixed precision apenas em CUDA. Esses
recursos são experimentais e não constituem validação clínica.

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

### Bridge JSONL experimental

Além do `replay` de gravações, a CLI aceita entrada em JSON, uma linha por chunk
pela entrada padrão, para integrar um driver próprio; não há captura de
equipamento.

O suporte LSL foi retirado. Relatórios e documentos históricos de ensaios LSL
preservam evidência passada, não representam capacidade atual do software.

Formato de entrada:

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
O critério de lacuna temporal é configurável com `--max-gap-factor` em
`stream-json` e `audit-json` e é salvo no relatório de sessão.

O JSON deve informar `sampling_rate` ou receber `--source-rate`; se ambos faltarem,
o processo assume 128 Hz e avisa no stderr. A taxa deve permanecer constante
durante o stream. O resampler polifásico mantém sobreposição/estado entre chunks
para converter outras taxas para os 128 Hz do modelo; a CLI libera a cauda ao
terminar um JSONL finito. Esse caminho ainda precisa ser medido no hardware real
antes de um estudo prospectivo. O bridge não valida o sinal nem transforma o
protótipo de pesquisa em equipamento clínico.

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

Para um smoke test rápido, use `--max-windows 512`. O padrão de treino é `--min-quality 0.2`; use `--min-quality 0` somente para auditar o bruto, pois pode reintroduzir janelas saturadas.

### Dados e artefatos rastreados

O código é MIT, mas os dados têm termos próprios. O repositório **não
redistribui dados derivados do VitalDB**: os arquivos `vitaldb_case*.npz` são
gerados localmente por `download-vitaldb`, sob o acordo de uso do VitalDB, e
ficam fora do Git (`data/vitaldb/*.npz` no `.gitignore`). Em 24/09/2026 os 15
arquivos derivados do VitalDB foram removidos do índice **e do histórico**;
clones anteriores a essa data ainda podem contê-los e devem ser tratados como
desatualizados.

Em 26/10/2026 o histórico foi reescrito de novo para remover também os 24
arquivos `.mat` do Figshare e os checkpoints `.pt` (`brainsniffer_cnn.pt` e
`brainsniffer_corpus_fixed.pt`). **Nenhum dado e nenhum peso treinado é publicado
neste repositório.** Clones anteriores a essa data ainda podem contê-los e devem
ser tratados como desatualizados.

- **Dados:** `download-data` baixa os 24 arquivos do corpus
  [EEG and BIS raw data](https://doi.org/10.6084/m9.figshare.5589841.v1)
  (Figshare, CC BY 4.0) para `data/raw/`, e `download-vitaldb` baixa casos do
  VitalDB para `data/vitaldb*/`. Ambos ficam fora do Git e são baixados por quem
  executa o sistema.
- **Pesos:** os `.pt` ficam só na máquina de quem treina (`train` e `train-corpus`
  os reconstroem). Os sidecars `.json` (configuração, métricas e SHA-256)
  continuam versionados para auditoria dos relatórios.

### Licença e termos do VitalDB

O VitalDB Open Dataset é distribuído sob **CC BY-NC-SA 4.0** e um
[Data Use Agreement](https://vitaldb.net/dataset/) (seção "Data Use Agreement").
Quem usa os dados ou pesos derivados deles deve observar, entre outros pontos do
acordo:

- uso **somente para pesquisa e desenvolvimento**; **uso comercial é vedado**;
- **não divulgar os dados** a quem não seja contratado pelo usuário, sem
  consentimento do provedor;
- **não tentar identificar pessoas** a partir dos dados;
- citar o provedor: Lee H-C et al., *VitalDB, a high-fidelity multi-parameter
  vital signs database in surgical patients*, Scientific Data, 2022.

Os pesos treinados com VitalDB (por exemplo, o candidato `corpus_fixed`) são
obra derivada desses dados e devem ser tratados como **restritos a pesquisa, sem
uso comercial e com a mesma licença da fonte (CC BY-NC-SA 4.0)**. Se o acordo for
alterado, vale a versão vigente em vitaldb.net.

A licença MIT cobre apenas o código. Ela não relicencia dados, checkpoints ou
relatórios de terceiros. Preserve a atribuição e os termos do Figshare
(CC BY 4.0), siga o acordo do VitalDB e não presuma que estar no Git equivalha a
autorização de redistribuição ou uso clínico.

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

As decisões detalhadas, as hipóteses e os itens que ainda dependem de validação estão em [`docs/notas/decisions.md`](docs/notas/decisions.md).
O contrato para uma fonte de EEG real está em [`docs/_archive/live_acquisition.md`](docs/_archive/live_acquisition.md); antes de usar um equipamento, preencha também a [`ficha de entrada de EEG real`](docs/_archive/real_eeg_intake.md).
O protocolo para avançar de replay para bancada, validação externa travada e modo sombra está em [`docs/_archive/prospective_protocol.md`](docs/_archive/prospective_protocol.md).

## Possíveis pesquisas futuras

Estudos posteriores podem ampliar a comparação com a baseline espectral/entropia,
medir latência no hardware alvo ou avaliar outro centro/dataset sob protocolo
pré-registrado. São possibilidades de continuidade, não uma lista de obrigações
para a demonstração de graduação. Qualquer nova conclusão deve respeitar a
separação por caso e reportar erros, associação e desempenho por faixa; janelas
aleatórias não demonstram generalização a pacientes novos.

O adaptador VitalDB baixa somente os tracks `BIS/EEG1_WAV` e `BIS/BIS` do caso solicitado e grava um `.npz` normalizado em 128 Hz. A licença/termos do VitalDB devem ser lidos antes do uso; esses arquivos não são misturados automaticamente ao treino Figshare.
