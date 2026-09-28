# Mapa das figuras científicas

As nove figuras são geradas por `docs/artigo/generate_figures.py` a partir de listas
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

A `eeg_windows.png` é a exceção deliberada: ela é a única figura que não lê
relatório nenhum, e sim o sinal bruto de `data/raw/case19.mat`, porque existe
para explicar o pré-processamento e não para medir desempenho. Ela passa pelo
mesmo `StreamingPreprocessor` que a inferência usa, de modo que o que a figura
mostra é o que o modelo recebe. As nove estão na ordem em que aparecem no
artigo, e as capturas do LaTeX carregam a mesma leitura que os "Takeaway" abaixo.

## Contrato visual comum

- Renderizador: Matplotlib (backend Agg), sem desenho direto por Pillow,
  seaborn, cartões, flores, cabeçalhos decorativos ou caixas de conclusão.
- Superfície: largura física de 15 cm (bloco de texto SBC: A4 menos 3 cm de
  margem de cada lado), igual a `\\columnwidth`, de modo que o TeX insere as
  figuras sem reescala e os pontos tipográficos não encolhem. PNG a 300 dpi e
  PDF vetorial com fontes TrueType incorporadas.
- Tipografia: DejaVu Sans distribuída com Matplotlib, 9–10 pt na escala final.
  Fundo branco, eixos pretos de 0,7 pt e grid cinza claro de 0,5 pt.
- Cor: a cor é reforço, nunca o único canal. A paleta é a de Okabe–Ito, segura
  para daltonismo, com um matiz por braço: azul `#0072B2` para o checkpoint
  ativo, vermelho `#D55E00` para o candidato misto fixo, verde `#009E73` para o
  baseline espectral (RF), cinza escuro `#333333` para a referência BIS sem
  arm, âmbar `#E69F00` para a quarentena de desenvolvimento e cinza `#9E9E9E`
  para os congelados fora do pool. Sem gradientes decorativos; a única escala
  contínua do conjunto é a do espectrograma. A versão em PNG e a vetorial em
  PDF saem do mesmo desenho, então a figura não muda de cor entre as duas.
- Distinção por formato (leitura em preto e branco): círculo preenchido e linha
  contínua para o checkpoint ativo, quadrado aberto e linha tracejada para o
  candidato misto; IC misto tracejado. As barras de erro mantêm a cor do braço,
  mas a leitura não depende dela. Segmentos cinza unem os pares na comparação.
  No offset, métricas usam símbolos/linhas distintos (círculo/linha contínua vs.
  quadrado/linha tracejada). Na trajetória, BIS de referência em linha contínua
  e CNN em linha tracejada. Nas barras de corpus, elegíveis de desenvolvimento
  em preenchimento sólido, quarentena de desenvolvimento em hachurado (`///`) e
  congelados fora do pool em quadriculado (`xx`), todos com borda preta; o
  "9 de 15" dentro do bloco congelado é uma anotação sobre a mesma barra, sem
  dupla contagem. No fluxograma, as duas matizes de braço têm luminância quase
  igual (231,7 contra 231,0, ou seja, indistinguíveis em escala de cinza), então
  cada caixa de braço carrega o marcador que o identifica nas demais figuras:
  círculo cheio na coluna do ativo, quadrado aberto na coluna do misto.
- Trajetória: o BIS de referência é cinza escuro contínuo, a predição bruta é
  azul-claro `#56B4E9` tracejada e fina, e o EWMA é o azul cheio pontilhado e
  grosso. Bruto e EWMA saem do mesmo checkpoint, então os dois são azuis: com o
  mesmo tom, as duas curvas se confundiam onde andavam juntas. O azul-claro
  também abre 70 pontos de luminância em relação ao azul cheio (164 contra 94),
  que é o que faz a separação sobreviver ao preto e branco.
- Espectrograma: a potência é desenhada com `rasterized=True`, senão o PDF
  vetorial embute centenas de milhares de quadriláteros e o arquivo deixa de ser
  utilizável. O mapa de cor é o `turbo`, que vai do azul ao vermelho — a leitura
  que a maioria espera num espectrograma. **É o único ponto do conjunto em que a
  cor carrega o valor sozinha, e o `turbo` não é monótono em luminância**, então
  em preto e branco o espectrograma perde a gradação (a banda baixa e a alta
  caem as duas no escuro). Foi uma troca aceita: as outras oito continuam
  legíveis em preto e branco, e a escala absoluta do espectrograma nunca foi
  lida por cor em texto nenhum. A escala é limitada por `nanpercentile` (5% a
  99,5%) sobre a própria gravação — testei cortes mais apertados no topo (97,
  95, 90) e todos saturavam mais a banda baixa e apagavam a estrutura de
  beta/gama, então o corte largo ficou. As colunas sem dado viram `NaN`,
  desenhadas como lacunas brancas em vez de manchas escuras que se leriam como
  silêncio do EEG.
- Painel de calibração: diamante preenchido marca o baseline espectral (RF); as
  linhas de referência em x = 0 (pontilhada) e x = 1 (tracejada) distinguem-se
  dos IC por braço (sólido para o ativo, tracejado para o misto e pontilhado
  para o RF). Os rótulos da legenda são curtos ("Ativo", "Misto", "RF") porque
  o nome completo de cada braço já está no rótulo do eixo y; com os nomes
  longos a legenda ficava mais larga que o painel e vazava pela borda. As
  anotações da direita ficam em duas linhas (`ICC bruto→EWMA`, `% janelas ≤10`),
  com o cabeçalho na faixa livre de cima: em três linhas a anotação ficava mais
  alta que o passo entre as linhas do painel e os textos vizinhos se
  sobrepunham.
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

## `eeg_windows.png`

- Pergunta: como é uma janela de entrada, o que o pré-processamento causal faz
  com ela, e o que o modelo vê ao longo da gravação?
- Takeaway: o clip, o filtro causal e a escala quase não tocam um trecho limpo
  (pico de 38 para 36 $\mu V$, BIS 30) e contêm um trecho com pico de
  $125\,\mu V$ (BIS 75) para $88\,\mu V$; o espectrograma da gravação inteira na
  base de tempo do BIS mostra a relação que o modelo explora.
- Painéis: (a,b) duas janelas reais do caso 19 em quatro eixos que compartilham
  x (tempo na janela, em segundos) e y (amplitude, em $\mu V$) — linha de cima o
  trecho limpo, linha de baixo o trecho com pico, coluna da esquerda o sinal
  bruto e da direita o sinal após o pipeline; cada linha traz o BIS do alvo no
  rótulo. (c) espectrograma causal dos 72 min e a referência BIS logo abaixo, no
  mesmo eixo de tempo.
- Espectrograma: STFT de janela Hann com `nperseg` igual à janela do modelo (640
  amostras, 5 s, a mesma unidade que o modelo consome) e avanço de 2,5 s,
  potência em escala de densidade; a banda desenhada é a mesma do
  pré-processamento (0,5--45 Hz). O mapa de cor é o `turbo` (azul a vermelho),
  com a ressalva de preto e branco registrada no contrato visual.
- Leitura de banda: a correlação entre a potência de banda e o BIS
  no caso 19 sustenta o texto do artigo — teta (4--8 Hz) $r=-0,417$, beta
  (13--30 Hz) $r=+0,436$ e gama (30--45 Hz) $r=+0,359$, enquanto delta (0,5--4
  Hz) $r=-0,047$ e alfa (8--13 Hz) $r=-0,157$ ficam fracas. A figura mostra
  teta e beta/gama de propósito: não há alegação sobre a banda baixa.
- Quedas de eletrodo: a gravação tem uma a cada $\approx$10 min (min 8,50,
  18,46, 28,42, 38,42, 48,35, 58,27 e 68,25), em que o EEG cru é exatamente
  constante. A geração detecta essas colunas e as marca como `NaN`, de modo que
  aparecem como lacunas brancas; sem isso virariam listras escuras e se leriam
  como silêncio do EEG. O eixo de cor é fixado por `nanpercentile` sobre os
  valores finitos.
- Grão e unidades: um caso; frequência em Hz, tempo em minutos desde o início da
  gravação, potência em dB. A referência BIS é a do monitor, com valores fora de
  $[0,100]$ descartados.
- Fonte: `data/raw/case19.mat` (rastreado pelo Git), lido pelo mesmo
  `StreamingPreprocessor` da inferência e processado de uma vez sobre a
  gravação inteira. Nada é recalculado a partir de relatórios: esta é a única
  figura que lê sinal bruto, e ela existe para explicar o pré-processamento, não
  para medir desempenho.

## `bis_trajectory.png`

- Pergunta: como BIS e CNN evoluem ao longo de uma gravação completa?
- Takeaway: exemplo individual do Figshare case19 (fixado antes da inferência),
  sem alegação de generalização; associação alta convive com viés sistemático.
  No painel superior, a curva pontilhada é o EWMA span 10 da própria CNN ativa,
  derivado na geração a partir das predições salvas ($\alpha = 2/(\mathrm{span}+1)$,
  reinício por caso; um caso só, sem retreino); o painel de erro permanece
  bruto.
- Cores: as duas curvas da CNN saem do mesmo checkpoint, então as duas são
  azuis — mas em tons separados, com o bruto em azul-claro `#56B4E9` tracejado
  e fino, e o EWMA no azul cheio `#0072B2` pontilhado e grosso. Com o mesmo
  tom, as duas se confundiam onde andavam juntas, que é a maior parte da
  gravação. Ver a nota de cor no contrato visual.
- Fonte: auditoria `tmp/pdfs/trajectory-audit/case19.json` (inferência
  explícita via `--infer-trajectory`; execução padrão não infere) e o array
  `cnn_raw_bis` de `case19.npz`, do qual o EWMA é derivado apenas no desenho.
  A auditoria descreve o alinhamento da inferência e registra que ela não
  aplicou EWMA; por isso a curva suavizada não entra em `case19.json`.

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
- Legenda: a entrada do EWMA mostra **os dois** marcadores, o semicírculo do
  ativo e o semiquadrado do misto. Com só o semicírculo, a legenda dizia que o
  EWMA existia apenas para o ativo, embora os dois braços tenham a versão
  suavizada no painel.
- Grão e unidades: pareamento 1:1 por `case_id` e `n_windows`; 15 casos e
  38.730 janelas no VitalDB. MAE em pontos BIS; Pearson sem unidade. Os
  agregados Figshare usam 5 casos/5.523 janelas.
- Fontes: `reports/figshare_holdout_evaluation.json`,
  `reports/mixed_fixed_figshare_holdout.json`,
  `reports/vitaldb_external_validation.json`,
  `reports/mixed_vitaldb_external.json` e, para os pontos EWMA span 10,
  `reports/ewma_postprocess.json`.

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

## `bootstrap_offset.png`

Figura única com quatro painéis: os dois de cima são a incerteza por bootstrap
(o antigo `bootstrap_intervals.png`) e os dois de baixo são a sensibilidade ao
offset do rótulo (o antigo `offset_sensitivity.png`). Foram fundidos porque cada
um sozinho ocupava menos de meia página e passava a ideia de dois resultados
independentes, quando os dois medem a mesma coisa: o quanto a métrica se move
quando se mexe no referencial e não no modelo.

- Pergunta (a,b): quanto as estimativas variam ao reamostrar cirurgias inteiras?
- Pergunta (c,d): as métricas mudam quando apenas o alinhamento do BIS varia?
- Takeaway: a incerteza depende do holdout e deve acompanhar as estimativas;
  os cinco casos Figshare limitam a leitura do intervalo interno. O EWMA span 10
  desloca os intervalos para baixo no MAE (Figshare misto 5,65--8,53 para
  4,56--7,62) sem alterar a conclusão de incerteza. No grid de -20 a +20 s,
  offsets posteriores reduzem MAE e elevam Pearson gradualmente; a tendência é
  pós-hoc e serve apenas para gerar hipótese.
- Painéis: (a) Pearson por caso e (b) MAE por caso, cada um com o intervalo
  bruto e o do span 10; (c) MAE e (d) Pearson contra o offset, com eixo y
  ampliado. Os painéis de offset têm um único checkpoint, então usam só o
  círculo cheio do ativo — não há par a comparar.
- Suporte histórico original (c,d): os nove pontos são lidos, não recalculados
  ou restringidos a suporte comum. Segmentos apenas unem pontos testados; N
  varia entre offsets. A linha vertical indica 0 s, não um ótimo selecionado.
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

## `ewma_tradeoff.png`

- Pergunta: quanto do ganho do EWMA é suavização emprestada, e a que preço de
  atraso a cadeia alcança o melhor resultado publicado no corpus?
- Takeaway: o resultado comparável do TCN-BiLSTM (6,03 ± 0,38) vem do protocolo
  por sujeito, onde o α é escolhido por fold e recai sobretudo em 0,05 ou 0,10 —
  isto é, de 45 a 95 s de idade média. Do span 3 ao 15, de 5 s a 35 s, o misto
  cai dentro ou abaixo desse intervalo de erro, com toda a varredura à esquerda
  da faixa em que eles operam. O span 5 é a comparação principal: 6,11 com 10 s
  de atraso. O span 10 dá o melhor MAE (5,84), mas dobra o atraso para 22,5 s, e
  o ganho além do span 5 é em parte emprestado da suavização. O α = 0,35, que
  casaria com o span 5, vale só para as curvas de sensibilidade dos splits
  aleatórios, outro regime.
- Painel: um só, MAE no Figshare contra a idade média das amostras do filtro
  (0, 2,5, 5, 10, 22,5 e 35 s, correspondendo aos spans 1, 2, 3, 5, 10 e 15;
  cada ponto é rotulado com o span, e o eixo vai até 100 s para caber a faixa
  deles). Faixa cinza: intervalo de erro reportado pelo TCN-BiLSTM; linha
  tracejada horizontal, o ponto estimado dele; região clara à direita, a faixa
  de atraso em que ele opera sob avaliação por sujeito (45 a 95 s). O anel no
  span 5 marca a comparação principal. Círculo cheio: CNN ativo; quadrado
  aberto: CNN misto; diamante cheio: RF espectral.
- Escopo: a varredura é pós-hoc e exploratória, sem retreino, sobre predições
  congeladas; span 1 reproduz o bruto auditado. Os IC 95% por bootstrap
  agrupado por caso existem para os spans 1, 5 e 10 (span 5: 4,84--7,96); não há
  bootstrap de $P_K$ por span.
- Fonte: `reports/ewma_postprocess.json` (auditado; o span 1 confere com os
  agregados históricos); α e faixa de atraso do TCN-BiLSTM conforme
  \cite{sukriti2026}, `SUKRITI_ALPHA_MIN`/`SUKRITI_ALPHA_MAX` em
  `docs/artigo/generate_figures.py`.

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
  anotações à direita trazem o ICC(2,1) absoluto bruto e EWMA na forma
  `ICC bruto→EWMA` e a fração bruta de janelas com |erro| ≤ 10 pontos BIS.
  Círculo preenchido: CNN ativo; quadrado aberto: CNN misto; diamante
  preenchido: baseline espectral (RF); meio preenchido à esquerda: EWMA span 10.
  Eixo y agrupado por benchmark, Figshare acima e VitalDB abaixo.
  A legenda do painel (b) usa "Ativo"/"Misto"/"RF" (o nome completo está no
  rótulo do eixo y) e fica em uma linha na faixa livre de baixo; a do painel
  (a) fica embaixo à esquerda, porque no canto superior caía sobre a curva de
  RMSE, a mais alta das três.
- Escopo: a projeção teórica de MAE por número de casos permanece apenas nos
  relatórios e no painel do site, fora do artigo; aqui a calibração é medida nos
  mesmos holdouts auditados (Figshare com 5 casos/5.523 janelas e VitalDB com
  15/38.730; B=1.000, seed 42) e confere n e MAE com os agregados históricos.
- Fontes: `models/brainsniffer_cnn.json`, `reports/calibration_analysis.json` e
  `reports/ewma_postprocess.json` (inclinações e ICC do span 10).

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
