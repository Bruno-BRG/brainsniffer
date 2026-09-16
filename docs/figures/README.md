# Mapa das figuras científicas

As quatro figuras são geradas por `docs/generate_figures.py` a partir de uma
lista explícita de snapshots JSON rastreados pelo Git. A geração valida escopo,
holdouts, contagens de janelas, pré-processamento e parâmetros de bootstrap
antes de escrever os PNGs e PDFs vetoriais. O fluxograma também lê os dois
JSON de checkpoints explicitamente listados em `MODEL_FILES` e confere hashes,
contagens e separação do holdout Figshare no desenvolvimento fixo. Arquivos JSON locais fora dessa lista não entram nas
figuras.

## Contrato visual comum

- Renderizador: Matplotlib (backend Agg), sem desenho direto por Pillow,
  seaborn, cartões, flores, cabeçalhos decorativos ou caixas de conclusão.
- Superfície: largura física de 16 cm, incluída a `\\columnwidth` no SBC;
  PNG a 300 dpi e PDF vetorial com fontes TrueType incorporadas.
- Tipografia: DejaVu Sans distribuída com Matplotlib, 9–10 pt na escala final.
  Fundo branco, eixos pretos de 0,7 pt e grid cinza claro de 0,5 pt.
- Paleta: azul `#0072B2`, vermelhão `#D55E00` e neutros; sem gradientes.
- Distinção não cromática: círculo preenchido para o checkpoint ativo,
  quadrado aberto para o candidato misto; IC misto tracejado. Segmentos cinza
  unem os pares na comparação. No offset, métricas usam símbolos/linhas distintos.
- Escalas: MAE parte de zero quando compara magnitudes; Pearson usa o domínio
  completo de -1 a +1; eixos ampliados da análise de offset são declarados.
- Escopo: resultados exploratórios de pesquisa, sem interpretação como
  validação clínica ou medida direta de consciência.

## `pipeline.png`

- Pergunta: como o fluxo evita mistura entre desenvolvimento e holdout?
- Takeaway: a separação por sujeito/caso e o holdout congelado antecedem o
  ajuste; configuração, proveniência e escopo permanecem auditáveis.
- Dados exibidos: pool elegível de 33 casos/55.471 janelas (23 Figshare + 10
  VitalDB), ativo Figshare com split 13/5/5 e misto fixo de 28 grupos (18+10)
  com split 16/6/6. Somente 13 Figshare ajustam o ativo; 12 Figshare + 4 VitalDB
  ajustam o misto. Avaliação comum: 5 casos Figshare e 15 VitalDB históricos.
- As setas de avaliação não indicam ajuste. Os cinco casos Figshare históricos
  ficam fora dos 28 grupos fixos. Figshare não possui identificação de pessoa:
  separar casos não comprova independência entre pessoas. Para o misto, VitalDB
  é benchmark histórico da mesma fonte já vista, não validação confirmatória.
- Pré-processamento e inventário completo ficam no método, não em cards.
- Fontes: `reports/corpus_manifest.json`, `models/brainsniffer_cnn.json` e
  `models/brainsniffer_corpus_fixed.json`. Snapshots LSL só auditam escopo.

## `comparison.png`

- Pergunta: a melhora agregada de MAE se repete caso a caso no VitalDB?
- Takeaway: o candidato misto reduz MAE em 14 dos 15 casos pareados, mas há
  heterogeneidade e o `vitaldb_case16` piora 0,67 ponto BIS.
- Grão e unidades: pareamento 1:1 por `case_id` e `n_windows`; 15 casos e
  38.730 janelas no VitalDB. MAE em pontos BIS; Pearson sem unidade. Os
  agregados Figshare usam 5 casos/5.523 janelas.
- Fontes: `reports/figshare_holdout_evaluation.json`,
  `reports/mixed_fixed_figshare_holdout.json`,
  `reports/vitaldb_external_validation.json` e
  `reports/mixed_vitaldb_external.json`.

## `offset_sensitivity.png`

- Pergunta: as métricas mudam quando apenas o alinhamento do BIS varia?
- Takeaway: no grid testado de -20 a +20 s, offsets posteriores reduzem MAE e
  elevam Pearson gradualmente; a tendência é pós-hoc e serve apenas para gerar
  hipótese.
- Suporte histórico original: os nove pontos são lidos, não recalculados ou
  restringidos a suporte comum. Segmentos apenas unem pontos testados; N varia
  entre offsets. A linha vertical indica 0 s, não um ótimo selecionado.
- Grão e unidades: 5 casos congelados; 5.503-5.525 janelas de 5 s por offset;
  MAE em pontos BIS, Pearson sem unidade e offset em segundos.
- Fonte: `reports/offset_sensitivity.json`.

## `bootstrap_intervals.png`

- Pergunta: quanto as estimativas variam ao reamostrar cirurgias inteiras?
- Takeaway: a incerteza depende do holdout e deve acompanhar as estimativas;
  os cinco casos Figshare limitam a leitura do intervalo interno.
- Grão e unidades: 1.000 reamostragens por caso, seed 42; Figshare com 5
  casos/5.523 janelas e VitalDB com 15/38.730; MAE em pontos BIS e Pearson sem
  unidade.
- Leitura do intervalo: o marcador é a métrica observada registrada em
  `metrics` (ou `recomputed_test_metrics` no relatório Figshare ativo), e as
  barras são os limites do IC percentil. O bootstrap reamostra casos inteiros
  (`B=1.000`, seed 42), mas concatena suas janelas em cada réplica; por isso, a
  métrica agregada continua ponderada pelo número de janelas/duração do caso.
- Fontes: `reports/figshare_holdout_evaluation.json`,
  `reports/mixed_fixed_figshare_holdout.json`,
  `reports/vitaldb_external_validation.json` e
  `reports/mixed_vitaldb_external.json`.

## Inventário auditado

Além dos seis relatórios quantitativos usados diretamente, os dois snapshots
LSL sintéticos foram auditados apenas para confirmar as barreiras de uso
registradas nas execuções históricas. LSL, seu comando, dependências e publisher
foram removidos do projeto. Os snapshots permanecem inalterados para auditoria
histórica do gerador: não representam capacidade atual, não validam hardware e
não foram reatribuídos ao JSONL. A fixture
`examples/stream_metadata.bench.synthetic.json` mantém a identidade histórica do
publisher removido e relógio liblsl, não uma instrução ativa. Eles não são
tratados como evidência de desempenho científico. A lista completa é:

1. `reports/corpus_manifest.json`
2. `reports/figshare_holdout_evaluation.json`
3. `reports/lsl_synthetic_intake_session.json`
4. `reports/lsl_synthetic_session.json`
5. `reports/mixed_fixed_figshare_holdout.json`
6. `reports/mixed_vitaldb_external.json`
7. `reports/offset_sensitivity.json`
8. `reports/vitaldb_external_validation.json`
