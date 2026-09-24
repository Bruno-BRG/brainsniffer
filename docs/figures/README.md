# Mapa das figuras científicas

As oito figuras são geradas por `docs/generate_figures.py` a partir de listas
explícitas de snapshots JSON rastreados pelo Git. A geração valida escopo,
holdouts, contagens de janelas, pré-processamento e parâmetros de bootstrap
antes de escrever os PNGs e PDFs vetoriais. O fluxograma também lê os dois
JSON de checkpoints explicitamente listados em `MODEL_FILES` e confere hashes,
contagens e separação do holdout Figshare no desenvolvimento fixo. A figura por
de Pk valida ainda os quatro relatórios `pk_*.json` contra os agregados
históricos (mesmo $n$, MAE/RMSE/bias/Pearson dentro de $10^{-4}$, Pk dentro do
próprio intervalo e semente fixa). Arquivos JSON locais fora dessas listas não
entram nas figuras.

## Contrato visual comum

- Renderizador: Matplotlib (backend Agg), sem desenho direto por Pillow,
  seaborn, cartões, flores, cabeçalhos decorativos ou caixas de conclusão.
- Superfície: largura física de 15 cm (bloco de texto SBC: A4 menos 3 cm de
  margem de cada lado), igual a `\\columnwidth`, de modo que o TeX insere as
  figuras sem reescala e os pontos tipográficos não encolhem. PNG a 300 dpi e
  PDF vetorial com fontes TrueType incorporadas.
- Tipografia: DejaVu Sans distribuída com Matplotlib, 9–10 pt na escala final.
  Fundo branco, eixos pretos de 0,7 pt e grid cinza claro de 0,5 pt.
- Paleta: preto `#000000`, cinza `#555555` e neutros; sem gradientes e sem depender de cor para distinguir séries.
- Distinção por formato (leitura em preto e branco): círculo preenchido e linha contínua para o checkpoint ativo, quadrado aberto e linha tracejada para o candidato misto; IC misto tracejado. Segmentos cinza unem os pares na comparação. No offset, métricas usam símbolos/linhas distintos (círculo/linha contínua vs. quadrado/linha tracejada). Na trajetória, BIS de referência em linha contínua e CNN em linha tracejada. Nas barras de corpus, elegíveis de desenvolvimento em preenchimento sólido, quarentena de desenvolvimento em hachurado (`///`) e congelados fora do pool em quadriculado (`xx`), todos com borda preta; o "9 de 15" dentro do bloco congelado é uma anotação sobre a mesma barra, sem dupla contagem.
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

## `pk_prediction.png`

- Pergunta: o indicador ordena corretamente a profundidade observada, e quanto
  disso sobrevive à troca de fonte de dados?
- Takeaway: no Figshare o ativo fica em Pk 0,773 e o misto em 0,774; no VitalDB
  o ativo cai para 0,515, praticamente acaso, e o misto chega a 0,672. O misto
  melhora MAE e Pearson no Figshare sem mover Pk, o que separa ganho de escala
  de ganho de ordenação.
- Métrica: probabilidade de predição Pk de Smith, Dutton e Smith
  (Anesthesiology 1996, 84:38-51), variante reescalada da associação ordinal de
  Kim; 1 é ordem perfeita e 0,5 é chance. Pares empatados na escala observada
  são excluídos e pares empatados no indicador contam 0,5.
- Escala observada: o BIS de referência do monitor, não um desfecho de resposta
  ao estímulo como no artigo original.
- Grão e unidades: Figshare com 5 casos/5.523 janelas e VitalDB com 15/38.730;
  IC 95% por reamostragem de casos (B=1.000, seed 42); Pk sem unidade.
- Fontes: `reports/pk_figshare_active.json`, `reports/pk_figshare_mixed.json`,
  `reports/pk_vitaldb_active.json` e `reports/pk_vitaldb_mixed.json`, auditados
  contra os quatro agregados históricos.

## `corpus_panels.png`

- Pergunta: como o corpus se compõe e por que os dois gates de qualidade não são
  redundantes?
- Takeaway: os 59 arquivos se dividem em categorias mutuamente exclusivas por
  fonte — 33 elegíveis de desenvolvimento, 11 em quarentena de desenvolvimento e
  15 congelados do benchmark histórico, que ficam fora do pool — e um caso pode
  ter quase todo o EEG finito e ainda perder janelas por qualidade, ou ser
  retido antes da contagem por lacuna inválida.
- Painéis: (a) casos por fonte (24 Figshare + 35 VitalDB = 59), empilhando
  elegíveis de desenvolvimento, quarentena de desenvolvimento e congelados fora
  do pool, com a soma rotulada por fonte e a anotação de que 9 dos 15 congelados
  reprovam nos gates internos (sobre a mesma barra, sem dupla contagem); (b)
  fração de amostras de EEG finitas contra fração de janelas aceitas, com a linha
  tracejada no gate de 90% de finitude.
- Fontes: `reports/corpus_manifest.json`.

## `training_panels.png`

- Pergunta: como o treinamento se comporta e o que uma projeção de casos
  permitiria planejar?
- Takeaway: a perda de treino e as métricas de validação medem coisas diferentes
  (adimensional versus pontos BIS) e o critério de leitura é a validação; a
  projeção é um cenário ancorado no único valor medido, não uma curva medida.
- Painéis: (a) histórico de treino e validação por época (perda de treino,
  adimensional, e MAE/RMSE de validação, em pontos BIS); (b) projeção teórica de
  MAE por número de casos de treino em escala logarítmica, ancorada em 13 casos e
  MAE 7,03, explicitamente não medida.
- Fontes: `models/brainsniffer_cnn.json` e
  `reports/figshare_holdout_evaluation.json`.

## `bis_trajectory.png`

- Pergunta: como BIS e CNN evoluem ao longo de uma gravação completa?
- Takeaway: exemplo individual do Figshare case19 (fixado antes da inferência),
  sem alegação de generalização; associação alta convive com viés sistemático.
- Fonte: auditoria `tmp/pdfs/trajectory-audit/case19.json` (inferência
  explícita via `--infer-trajectory`; execução padrão não infere).

## Inventário auditado

Além dos dez relatórios quantitativos usados diretamente, os dois snapshots
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
9. `reports/pk_figshare_active.json`
10. `reports/pk_figshare_mixed.json`
11. `reports/pk_vitaldb_active.json`
12. `reports/pk_vitaldb_mixed.json`
