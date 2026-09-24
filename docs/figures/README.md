# Mapa das figuras científicas

As nove figuras são geradas por `docs/generate_figures.py` a partir de listas
explícitas de snapshots JSON rastreados pelo Git. A geração valida escopo,
holdouts, contagens de janelas, pré-processamento e parâmetros de bootstrap
antes de escrever os PNGs e PDFs vetoriais. O fluxograma também lê os dois
JSON de checkpoints explicitamente listados em `MODEL_FILES` e confere hashes,
contagens e separação do holdout Figshare no desenvolvimento fixo. A figura de
Pk valida ainda os quatro relatórios `pk_*.json` contra os agregados
históricos (mesmo $n$, MAE/RMSE/bias/Pearson dentro de $10^{-4}$, Pk dentro do
próprio intervalo e semente fixa), e o painel de calibração valida
`reports/calibration_analysis.json` contra os mesmos agregados (mesmo $n$ e MAE
dentro de $10^{-4}$). A varredura EWMA lê `reports/ewma_postprocess.json`
(pós-hoc exploratório, sem retreino) e confere que o span 1 reproduz os mesmos
agregados brutos e que os intervalos bootstrap estão ordenados; Pk, inclinação e
ICC de cada span também são auditados. Arquivos JSON locais fora dessas listas
não entram nas figuras.

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
- Painel de calibração: diamante cinza preenchido marca o baseline espectral
  (RF); as linhas de referência em x = 0 (pontilhada) e x = 1 (tracejada)
  distinguem-se dos IC por braço (sólido para o ativo, tracejado para o misto
  e pontilhado para o RF).
- EWMA exploratório (span 10, pós-hoc, sem retreino): a versão suavizada usa o
  mesmo marcador do braço com a metade esquerda preenchida e linhas/intervalos
  pontilhados, sempre ao lado da versão bruta; onde não há bootstrap de $P_K$
  nem IC da inclinação EWMA, o valor aparece sem intervalo, sem estimativa.
- Na trajetória, a curva EWMA span 10 é derivada na geração a partir das
  predições brutas já salvas (mesma regra do relatório: $\alpha = 2/(\mathrm{span}+1)$,
  reinício por caso; com um único caso, começa na primeira janela aceita) e por
  isso não faz parte da auditoria de inferência `case19.json`.
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
  heterogeneidade e o `vitaldb_case16` piora 0,67 ponto BIS. Com EWMA span 10,
  os agregados caem nos dois benchmarks (Figshare misto 6,69 para 5,84 e
  VitalDB misto 8,60 para 7,89), de forma exploratória.
- Painéis: (a) MAE agregado e (b) Pearson agregado trazem, por benchmark, a
  versão bruta e a EWMA span 10 (sub-lane inferior, rótulo "EWMA span 10" no
  eixo, marcador semicircular e ligação pontilhada); (c) mostra o MAE pareado
  por caso apenas na versão bruta, que é o foco do pareamento.
- Grão e unidades: pareamento 1:1 por `case_id` e `n_windows`; 15 casos e
  38.730 janelas no VitalDB. MAE em pontos BIS; Pearson sem unidade. Os
  agregados Figshare usam 5 casos/5.523 janelas.
- Fontes: `reports/figshare_holdout_evaluation.json`,
  `reports/mixed_fixed_figshare_holdout.json`,
  `reports/vitaldb_external_validation.json`,
  `reports/mixed_vitaldb_external.json` e, para os pontos EWMA span 10,
  `reports/ewma_postprocess.json`.

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
  os cinco casos Figshare limitam a leitura do intervalo interno. O EWMA span 10
  desloca os intervalos para baixo no MAE (Figshare misto 5,65--8,53 para
  4,56--7,62) sem alterar a conclusão de incerteza.
- Grão e unidades: 1.000 reamostragens por caso, seed 42; Figshare com 5
  casos/5.523 janelas e VitalDB com 15/38.730; MAE em pontos BIS e Pearson sem
  unidade.
- Leitura do intervalo: o marcador é a métrica observada registrada em
  `metrics` (ou `recomputed_test_metrics` no relatório Figshare ativo), e as
  barras são os limites do IC percentil. O bootstrap reamostra casos inteiros
  (`B=1.000`, seed 42), mas concatena suas janelas em cada réplica; por isso, a
  métrica agregada continua ponderada pelo número de janelas/duração do caso.
- EWMA: para cada braço CNN, a linha de baixo repete o intervalo no span 10 com
  traço pontilhado e marcador semicircular (fonte: `case_bootstrap` de
  `reports/ewma_postprocess.json`). Não há bootstrap de $P_K$ para o EWMA, e
  $P_K$ não aparece neste painel.
- Fontes: `reports/figshare_holdout_evaluation.json`,
  `reports/mixed_fixed_figshare_holdout.json`,
  `reports/vitaldb_external_validation.json`,
  `reports/mixed_vitaldb_external.json` e `reports/ewma_postprocess.json`.

## `pk_prediction.png`

- Pergunta: o indicador ordena corretamente a profundidade observada, e quanto
  disso sobrevive à troca de fonte de dados?
- Takeaway: no Figshare o ativo fica em Pk 0,773 e o misto em 0,774; no VitalDB
  o ativo cai para 0,515, praticamente acaso, e o misto chega a 0,672. O misto
  melhora MAE e Pearson no Figshare sem mover Pk, o que separa ganho de escala
  de ganho de ordenação. Com EWMA span 10, o misto sobe para 0,791 no Figshare e
  0,703 no VitalDB, enquanto o ativo no VitalDB permanece em 0,511.
- Painéis: cada benchmark tem uma linha bruta (com IC 95% por caso) e uma linha
  EWMA span 10 (sem IC, pois não há bootstrap de $P_K$ para o EWMA).
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
  contra os quatro agregados históricos; os pontos EWMA span 10 vêm de
  `reports/ewma_postprocess.json`.

## `ewma_sweep.png`

- Pergunta: como a suavização causal EWMA muda MAE e $P_K$ conforme o span, e o
  ganho se sustenta na troca de fonte?
- Takeaway: o MAE cai com o span nos dois benchmarks, com retorno decrescente;
  no Figshare o misto vai de 6,69 bruto a 5,84 (span 10) e 5,68 (span 15), e o
  RF espectral também melhora (8,86 para 7,56/7,38); no VitalDB o misto vai de
  8,60 a 7,89 (span 10), enquanto o ativo quase não muda (12,43 para 12,06).
  $P_K$ do ativo no VitalDB continua no acaso (0,515 para 0,511), o que separa
  perda de ordenação de ruído.
- Painéis: linha superior Figshare, inferior VitalDB; coluna esquerda MAE
  (pontos BIS, eixo de 0 a 14,5) e direita $P_K$ (0,4 a 1,0, com linha
  pontilhada de chance em 0,5); x é o span em janelas de 5 s (1, 2, 3, 5, 10,
  15). A linha guia vertical marca o span 10 e a anotação traz o valor do misto
  nesse span. Círculo cheio: CNN ativo; quadrado aberto: CNN misto; diamante
  cinza: RF espectral; linhas contínua, tracejada e pontilhada seguem a
  convenção do artigo.
- Escopo: a varredura inteira é pós-hoc e exploratória, sem retreino, sobre
  predições congeladas; span 1 reproduz o bruto auditado, e span 10 é a
  referência das demais figuras por equilibrar memória (50 s) e ganho. Não há
  bootstrap de $P_K$ por span.
- Fonte: `reports/ewma_postprocess.json` (auditado; o span 1 confere com os
  agregados históricos).

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

- Pergunta: como o treinamento se comporta e como cada braço calibra fora do
  domínio em que foi ajustado?
- Takeaway: a perda de treino e as métricas de validação medem coisas diferentes
  (adimensional versus pontos BIS) e o critério de leitura é a validação; na
  calibração, o ativo perde a relação no VitalDB (inclinação 0,02, ICC 0,02) e o
  misto recupera parcialmente (0,55, ICC 0,67) sem igualar o Figshare. O EWMA
  span 10 eleva o ICC do misto (Figshare 0,81 para 0,86; VitalDB 0,67 para 0,71)
  e mantém a inclinação (0,82 para 0,84; 0,55 para 0,54), mas não recupera a
  ordenação do ativo no VitalDB (ICC 0,03).
- Painéis: (a) histórico de treino e validação por época (perda de treino,
  adimensional, e MAE/RMSE de validação, em pontos BIS); (b) inclinação da reta
  de calibração predito~referência por braço, com IC 95% por bootstrap de casos
  apenas no bruto (linha de cima); o EWMA span 10 aparece na linha de baixo,
  com marcador semicircular e sem IC (não disponível, não estimado); as linhas
  de referência marcam x = 0 (sem calibração) e x = 1 (identidade), e as
  anotações à direita trazem o ICC(2,1) absoluto bruto e EWMA e a fração bruta
  de janelas com |erro| ≤ 10 pontos BIS. Círculo preenchido: CNN ativo;
  quadrado aberto: CNN misto; diamante cinza: baseline espectral (RF); eixo y
  agrupado por benchmark, Figshare acima e VitalDB abaixo.
- Escopo: a projeção teórica de MAE por número de casos permanece apenas nos
  relatórios e no painel do site, fora do artigo; aqui a calibração é medida nos
  mesmos holdouts auditados (Figshare com 5 casos/5.523 janelas e VitalDB com
  15/38.730; B=1.000, seed 42) e confere n e MAE com os agregados históricos.
- Fontes: `models/brainsniffer_cnn.json`, `reports/calibration_analysis.json` e
  `reports/ewma_postprocess.json` (inclinações e ICC do span 10).

## `bis_trajectory.png`

- Pergunta: como BIS e CNN evoluem ao longo de uma gravação completa?
- Takeaway: exemplo individual do Figshare case19 (fixado antes da inferência),
  sem alegação de generalização; associação alta convive com viés sistemático.
  No painel superior, a curva pontilhada é o EWMA span 10 da própria CNN ativa,
  derivado na geração a partir das predições salvas ($\alpha = 2/(\mathrm{span}+1)$,
  reinício por caso; um caso só, sem retreino); o painel de erro permanece
  bruto.
- Fonte: auditoria `tmp/pdfs/trajectory-audit/case19.json` (inferência
  explícita via `--infer-trajectory`; execução padrão não infere) e o array
  `cnn_raw_bis` de `case19.npz`, do qual o EWMA é derivado apenas no desenho.
  A auditoria descreve o alinhamento da inferência e registra que ela não
  aplicou EWMA; por isso a curva suavizada não entra em `case19.json`.

## Inventário auditado

Além dos doze relatórios quantitativos usados diretamente, os dois snapshots
LSL sintéticos foram auditados apenas para confirmar as barreiras de uso
registradas nas execuções históricas. LSL, seu comando, dependências e publisher
foram removidos do projeto. Os snapshots permanecem inalterados para auditoria
histórica do gerador: não representam capacidade atual, não validam hardware e
não foram reatribuídos ao JSONL. A fixture
`examples/stream_metadata.bench.synthetic.json` mantém a identidade histórica do
publisher removido e relógio liblsl, não uma instrução ativa. Eles não são
tratados como evidência de desempenho científico. A lista completa é:

1. `reports/calibration_analysis.json`
2. `reports/corpus_manifest.json`
3. `reports/ewma_postprocess.json`
4. `reports/figshare_holdout_evaluation.json`
5. `reports/lsl_synthetic_intake_session.json`
6. `reports/lsl_synthetic_session.json`
7. `reports/mixed_fixed_figshare_holdout.json`
8. `reports/mixed_vitaldb_external.json`
9. `reports/offset_sensitivity.json`
10. `reports/vitaldb_external_validation.json`
11. `reports/pk_figshare_active.json`
12. `reports/pk_figshare_mixed.json`
13. `reports/pk_vitaldb_active.json`
14. `reports/pk_vitaldb_mixed.json`
