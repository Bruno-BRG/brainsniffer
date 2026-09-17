# Paridade site (Dash) x artigo (LaTeX/SBC)

Regra do projeto: tudo que esta no site tem que estar no artigo e vice-versa.
Grafico no site sem correspondente no artigo (ou o inverso) e pendencia, nao decisao.

## Correspondencia verificada (2026-09-17)

| Site (Dash) | Artigo (.tex) | Status |
|---|---|---|
| Visao geral: cards MAE/RMSE/Pearson | Resumo + Tab. benchmarks | OK |
| Trajetoria completa case19 | Fig. bis_trajectory.pdf | OK |
| Replay causal | Secao Replay retrospectivo | OK |
| Modelo: arquitetura + historico | Tab. config + Secao Modelos | OK |
| Corpus: composicao, qualidade, ativo x misto | Fig. pipeline.pdf + Tab. modelos | OK |
| Resultados: erros, associacao, bootstrap, offset | Fig. comparison/bootstrap/offset | OK |
| Resultados: MAE por zona + tabela (NOVO) | Fig. zone_accuracy + Tab. zonas (NOVO) | OK |
| Metodo e limitacoes | Materiais e metodos + Discussao | OK |

## Pendencias conhecidas

- Site-only: curva de aprendizagem teorica, historico por epoca detalhado,
  mapa finitude x janelas caso a caso, MAE por caso VitalDB em barras.
  Plano: compilar no artigo ou marcar como apoio interativo sem numeros divergentes.
- Artigo-only: fluxograma pipeline.pdf nao exibido no site.
  Plano: exibir os PDFs do artigo no site.
- Slides: ainda nao versionados. Cada grafico deve apontar para a mesma figura/tabela.

## Como manter

1. Novo grafico no site -> gerar via generate_figures.py, incluir em figures/,
   citar no .tex e registrar aqui.
2. Nova tabela no artigo -> exibir os mesmos numeros no Dash dos mesmos JSON.
3. Auditoria: generate_figures.py confere n, metricas, splits e hashes.
