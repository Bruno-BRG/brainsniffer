# Paridade site (Dash) x artigo (LaTeX/SBC)

Regra: toda figura do artigo aparece no site e todo painel analítico do site
aparece no artigo. Pendência, não decisão.

## Correspondência verificada (2026-09-18; revisada em 2026-09-24)

| Artigo | Site | Status |
|---|---|---|
| Fig. 1 `pipeline.pdf` | galeria "Figuras do artigo" e aba Corpus | OK |
| Fig. 2 `bis_trajectory.pdf` | aba Trajetória completa e galeria | OK |
| Fig. 3 `comparison.pdf` | aba Resultados (erros, associação, por caso) e galeria | OK |
| Fig. 4 `pk_prediction.pdf` | Resultados: gráfico Pk, tabela e galeria | OK |
| Fig. 5 `bootstrap_intervals.pdf` | Resultados: incerteza por caso e galeria | OK |
| Fig. 6 `offset_sensitivity.pdf` | Resultados: sensibilidade ao offset e galeria | OK |
| Fig. 7 `corpus_panels.pdf` | Corpus (a, b) e galeria | OK |
| Fig. 8 `training_panels.pdf` | Modelo: histórico de treino (a) e calibração por braço (b) e galeria | OK |
| Tabela de benchmarks/Pk | Resultados: tabelas de MAE/Pk no site | OK |

A galeria fica na aba Método e limites, na rota `/figures/<nome>`, com lista
explícita em `ARTICLE_FIGURES`. A análise por zona (tabela e figura) foi
removida dos dois lados quando o Pk entrou.

## Painéis interativos sem figura estática própria

Estes painéis leem os mesmos JSON do artigo e renderizam números que já estão
no texto; a forma é interativa, o conteúdo é o mesmo.

- Replay causal (EEG/BIS, qualidade, gate de emissão): protocolo descrito no
  artigo; a figura estática correspondente é a Fig. 2.
- Erros contínuos e associação/classificação por conjunto: mesmos números das
  tabelas de benchmark.
- Comparação ativa x misto e MAE por caso no VitalDB: mesmos números da Fig. 3.
- Curva de aprendizagem: projeção de planejamento do site, calculada em
  `brainsniffer.pipeline.planning`; não está mais no artigo, cuja Fig. 8b passou a
  mostrar a calibração por braço de `reports/calibration_analysis.json`.

## Citação

O método Pk é citado no artigo como `smith1996`: Smith, W. D., Dutton, R. C. e
Smith, N. T. (1996), Anesthesiology 84(1):38-51, DOI
10.1097/00000542-199601000-00005, PMID 8572353. A escala observada usada aqui é
o BIS de referência do monitor, não o desfecho de resposta ao estímulo do
artigo original, e isso está declarado no texto e nas legendas.

## Como manter

1. Novo gráfico no site: gerar também via `docs/generate_figures.py`, incluir em
   `docs/figures/`, citar no `.tex` e registrar aqui.
2. Nova figura no artigo: entra na galeria do site adicionando o nome em
   `ARTICLE_FIGURES`, sem recálculo à mão.
3. Auditoria: `generate_figures.py` confere n, métricas, Pk dentro do IC e
   semente; o Dash lê os mesmos JSON no startup.
