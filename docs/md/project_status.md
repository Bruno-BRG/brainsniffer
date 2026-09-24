# Status e rastreabilidade do BrainSniffer

Este documento liga o pedido original aos artefatos e aos registros históricos
do workspace. “Implementado” na tabela é um status histórico, não uma afirmação
de teste da versão atual. A rodada retrospectiva atual compilou e inspecionou
o PDF e executou inferência ilustrativa em um caso, sem treino ou testes do sistema.
Código em alteração paralela não é aprovado por esse build documental.
Relatórios históricos não provam correção atual nem validação clínica.

## Escopo definitivo e rodada temporal — evidência atual

- O TCC usa exclusivamente arquivos existentes e replay retrospectivo; não haverá
  conexão com EEG físico. LSL, seu adaptador, comando `stream-lsl`, publisher e
  dependências foram removidos. JSONL local opcional, auditoria e replay permanecem;
  não são promessa de integração com equipamento. Os registros históricos abaixo
  não são requisitos, entregas ou plano futuro deste TCC.
- Artigo e roteiro retiram destaque LSL/streaming laboratorial e distinguem replay
  de aquisição. Resultados negativos VitalDB, limites BIS, reuso de holdout e
  imputação offline foram preservados. Pipeline agora diz arquivos retrospectivos.
- Inferência CPU realmente executada no case19 inteiro, escolhido antes dos
  resultados por consistência com a demonstração indicada pelo coordenador, não
  por desempenho; nenhum outro caso ou trecho foi comparado para seleção.
  Loader restrito, checkpoint ativo existente, hashes conferidos, configuração
  original: 128 Hz, janela/passo 5 s, offset zero, qualidade mínima 0,20,
  dois threads e batch 64. Nenhum treino, download ou relatório histórico escrito.
- 4.355 s EEG, 871 janelas completas, 850 aceitas; 21 rejeitadas por qualidade.
  BIS tem 906 pontos: 35 finais sem EEG correspondente. As 56 posições sem
  previsão ficam NaN. EEG integralmente finito; referência não suavizada.
  MAE 8,7152, RMSE 10,2100, viés +4,7852 e Pearson 0,9100 nesse caso.
  A curva mostra superestimação intermediária apesar da associação alta.
- Figura vetorial `docs/figures/bis_trajectory.pdf` e PNG, dados
  `tmp/pdfs/trajectory-audit/case19.{npz,csv,json}`. Tempo do alvo offline
  (início+offset), não emissão do replay (fim+processamento); CNN bruta sem EWMA.
  Figura individual ilustrativa, sem alegação de generalização ou ausência de atraso.
- Reprodução, sem instalar dependências: `PYTHONDONTWRITEBYTECODE=1
  MPLCONFIGDIR=$PWD/tmp/pdfs/trajectory-audit/mpl .venv/bin/python
  docs/generate_figures.py --infer-trajectory`. Executado com saída 0; repetido
  após enriquecer a auditoria. `--trajectory` apenas redesenha os dados salvos;
  execução padrão gera figuras históricas sem inferir/baixar/treinar.
- Build solicitado Tectonic passou: **15 páginas**, sem overflow, referências
  indefinidas ou `??`. Auditoria: **37 parágrafos e 8 captions**, todos citados,
  nenhuma chave inexistente, sem agradecimentos. Bibliografia não editada.
- PDF renderizado com Poppler; páginas alteradas 1–3, 5, 7–9, 12–13 abertas e
  inspecionadas, inclusive curva na página 9. Não observados cortes/sobreposições;
  páginas de floats têm espaço branco amplo. Não se afirma nova inspeção das
  demais páginas. Avisos Underfull/Fontconfig/template permanecem no log.
- Evidências: `tmp/pdfs/trajectory-audit/` contém `generator.log`, `lint.log`,
  `build-console.log`, `pdfinfo.txt`, `layout.txt`, `pdf.sha256`, inventário de
  citações e `pages/`. Lint inicialmente apontou linha longa; enriquecimento
  intermediário da auditoria teve erro de sintaxe, ambos corrigidos antes do
  último gerador/lint. Não executados testes integrados ou benchmarks completos.
- Metadados institucionais, termos/ética, revisão e aceite final do autor
  continuam pendentes; ensaio cronometrado do roteiro não realizado.

## Rodada de citações — evidência histórica (PDF substituído acima)

- Agradecimentos (seção e texto) removidos integralmente. Auditoria do TeX:
  **36 parágrafos de prosa e 7 captions**, todos com citação; títulos,
  palavras-chave, equações, células e entradas bibliográficas são estruturais.
  Inventário e revisão semântica manual: `tmp/pdfs/citation-audit/paragraphs.md`,
  `paragraphs.json` e `review.md`. O parser não prova suporte semântico.
- Uma referência nova, `brainsnifferArtefatos`: autoria de `CITATION.cff`,
  artefatos locais não publicados em `src/brainsniffer/`, dois JSON de
  checkpoints e corpus de `reports/`; `s.d.` por ausência de data única
  comprovada. Nenhum DOI/URL público/commit/release foi atribuído ao conjunto.
  Resultados próprios não foram atribuídos a publicações externas. Ética cita
  origem dos datasets, sem presumir aprovação ou dispensa institucional.
- `TECTONIC_BIN=/home/fryits/.local/bin/tectonic scripts/build_tcc_article.sh`:
  saída 0, **14 páginas A4**, 25 referências resolvidas, sem overflow ou `??`;
  gates intactos. PDF atual: 152.331 bytes, SHA-256
  `88ae57c5bf806150f17db970c8527d590a5a161bdd5785c834d7e92179df3b9c`.
- `pdfinfo`, `pdftotext -layout` e `pdftoppm -png -r 85` executados;
  **todas as 14 páginas abertas e inspecionadas**. Sem cortes/sobreposições
  observados; espaço branco amplo em páginas de floats preservado. Figuras
  Matplotlib não regeneradas ou alteradas. Evidências em
  `tmp/pdfs/citation-audit/`: `build-console.log`, `final-tex.log`,
  `final-bibtex.log`, `pdfinfo.txt`, `layout.txt` e `pages/`.
- Avisos não ocultados: Fontconfig, UTF-8 legado do template, inputenc ignorado,
  seis Underfull hbox e um Underfull vbox finais. `git diff --check` dos três
  textos próprios passou. Hashes de relatórios, JSON de modelos, figuras e
  código inspecionado permaneceram iguais (`sources-check.log`). Sequência
  numérica do TeX fora das chaves de citação preservada (`checks.txt`).
- Nenhum treino, teste de sistema ou reavaliação de modelos executado.
  `docs/article.md` não editado nesta rodada. Curso, instituição, cidade/estado,
  revisão institucional/ética e aceite final do autor continuam pendentes.

## Rodada Matplotlib — evidência histórica das figuras (PDF substituído acima)

- Quatro figuras regeneradas realmente com Matplotlib 3.11.1, backend Agg,
  DejaVu Sans 9–10 pt, branco/eixos discretos; sem cards, flores ou takeaways.
  PNGs 300 dpi (chunk pHYs: 299,9994 dpi por arredondamento) e quatro PDFs
  vetoriais em `docs/figures/`, usados no TeX a 16 cm de largura.
- Matplotlib ausente na verificação inicial; adicionado ao extra `dev` com
  `uv add --optional dev 'matplotlib>=3.10' --no-sync`. Lock mínimo relativo
  ao início desta rodada: sete pacotes novos, nenhuma versão existente mudou.
  Pillow é agora dependência transitiva declarada de Matplotlib, não sobra do
  Streamlit; o gerador não importa Pillow. Dash/Plotly não foram alterados.
- `MPLCONFIGDIR=$PWD/tmp/pdfs/matplotlib/cache uv run --locked --extra dev python
  docs/generate_figures.py`: saída 0, oito snapshots auditados e oito saídas.
  Funções originais `report_value`, `ensure`, `load_reports`, `audit_reports`
  preservadas por comparação de AST. Dois controles negativos em memória
  (IDs/contagens VitalDB divergentes) foram rejeitados. Conferências adicionais
  do fluxograma verificam hashes, splits, fontes e exclusão do holdout Figshare.
- `TECTONIC_BIN=/home/fryits/.local/bin/tectonic scripts/build_tcc_article.sh`:
  saída 0, **14 páginas A4**, sem overflow/referências indefinidas/`??`, gates
  intactos. PDF atual: 150.726 bytes, SHA-256
  `642a17d7190fa312426cadc0d1400e8e90aad9f2fb79999b84c36410f3140704`.
- Evidências atuais: `tmp/pdfs/matplotlib/build-console.log`, `generator.log`,
  `pdfinfo.txt`, `layout.txt`, `sources-check.log`; intermediários finais em
  `tmp/pdfs/tcc-build/`. `pdftoppm -png -r 100` renderizou as 14 páginas em
  `tmp/pdfs/matplotlib/pages/`. Foram abertas as páginas 3 e 8–14 (figuras nas
  páginas 3, 9 e 10), além das quatro renderizações vetoriais `*-vector.png`
  a 150 dpi. Sem cortes/sobreposições observados; páginas de float 3 e 9 têm
  espaço branco amplo. Permanecem avisos Fontconfig, UTF-8 legado do template
  e Underfull, não escondidos. Não se afirma nova inspeção das páginas 1–2/4–7.
- `uv run --locked --extra dev ruff check docs/generate_figures.py` e
  `uv lock --check` passaram; lint inicial tinha formatação/linhas longas,
  corrigidas somente no gerador. SHA-256 de todos os JSON `reports/*.json` e
  `models/*.json` permaneceu idêntico ao início (`sources-before.sha256`).
  Métricas/IC/grade de offset apenas lidos, sem novos experimentos ou treino.
- O método não foi reduzido nesta rodada; somente inclusão e captions das
  figuras mudaram no TeX. Aceite estético final cabe ao autor. Metadados e
  revisão institucional/ética continuam pendentes; build não é certificação
  clínica nem aprovação do sistema.

## Rodada anterior do artigo — evidência histórica (PDF substituído acima)

- Tectonic 0.17.0 instalado em `/home/fryits/.local/bin/tectonic`, release oficial
  x86_64 Linux musl. Download direto falhou por conexão; download pela API oficial
  HTTPS funcionou, sem credenciais/sudo. SHA-256 do tarball conferido contra
  `digest` publicado: `8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7`.
  Procedência completa em `docs/latex/README.md`; metadados em
  `tmp/pdfs/tectonic-install/release.json`.
- `TECTONIC_BIN=/home/fryits/.local/bin/tectonic scripts/build_tcc_article.sh`:
  saída 0, **15 páginas A4**, bibliografia resolvida, sem `Overfull` e sem `??`
  no texto extraído. Script e seus gates foram preservados. PDF anterior só foi
  substituído após aprovação dos gates, conforme o script.
- PDF: `docs/tcc_brainsniffer.pdf`, 724.714 bytes, SHA-256
  `e7ea37962db8c185d067d5354dbb54c2c34683fe678c58fef5d3d0eb86662ccb`.
  Poppler 26.08.0. `pdfinfo`, `pdftotext -layout` e
  `pdftoppm -png -r 100` executados. Evidências: `tmp/pdfs/pdfinfo.txt`,
  `tmp/pdfs/tcc-layout.txt`, `tmp/pdfs/build-console.log`,
  `tmp/pdfs/tcc-build/` e `tmp/pdfs/tcc-render/page-01.png` a `page-15.png`.
- Todas as 15 renderizações foram abertas e inspecionadas: tabelas nas páginas
  6–8, figuras nas páginas 3 e 9–11, referências nas páginas 13–15. Não foram
  observados cortes/sobreposições. Há espaço branco em páginas de figuras e
  espaçamento vertical amplo na página 5. Avisos remanescentes: Fontconfig do
  host, `inputenc` ignorado pelo motor UTF-8, bytes legados em comentários das
  linhas 9–10 de `sbc-template.sty`, `Underfull` horizontais/vertical. Não foram
  ocultados nem corrigidos em arquivos fora da propriedade desta tarefa.
- Texto reduzido ao protótipo e comparação existente; `article.md` tornou-se
  nota histórica curta. Roteiro: **804 palavras faladas**, estimativa aritmética
  **5,36–6,70 min** a 150–120 palavras/min; ensaio cronometrado não realizado.
- Relato BIS completado por inspeção de `src/brainsniffer/data/vitaldb.py:173–207`:
  interpolação em grade de 1 s entre finitos, NaN fora do intervalo, sem limite
  de lacuna e sem máscara persistida no NPZ. Nenhuma política/métrica foi mudada.
- `bash -n scripts/build_tcc_article.sh` e `git diff --check` dos textos
  modificados passaram. Nenhum teste, treino, novo gráfico ou alteração de
  bibliografia realizado nesta rodada; bundle/cache TeX não fixado.
- Pendências humanas: curso, instituição, cidade/estado, revisão institucional,
  termos de uso e enquadramento ético. Não há parecer/dispensa inventado.
  O build não constitui aprovação científica, clínica ou do sistema.


| Requisito | Artefato principal | Evidência histórica ou caminho documentado | Status histórico / pendência |
|---|---|---|---|
| Python 3.12 | `pyproject.toml`, `.python-version` | `requires-python >=3.12,<3.13`; compileall | Implementado |
| CNN para EEG | `src/brainsniffer/models/cnn.py` | checkpoint `models/brainsniffer_cnn.pt` carregado e avaliado | Implementado |
| Estimar referência BIS/estágio | `data/preprocess.py`, `pipeline/metrics.py` | regressão 0–100 e quatro faixas de pesquisa | Implementado como pesquisa |
| Inferência em fluxo | `pipeline/realtime.py` | replay causal e emissão a cada stride | Implementado |
| Entrada local opcional | `pipeline/streaming.py`, CLI `stream-json` | Contrato JSONL; smoke histórico 256→128 Hz não prova equipamento | Preservado; LSL removido |
| Publisher LSL histórico | Antigo `examples/lsl_synthetic_publisher.py` (removido) | Relatos sintéticos preservados abaixo, não instrução de reprodução atual | Removido |
| Preflight do bridge | `pipeline/stream_audit.py`, CLI `audit-json` | stream bom/ruim testado no executável | Implementado |
| Manifesto do sinal | `pipeline/stream_audit.py`, CLI `stream-json`, `examples/stream_metadata.template.json` | unidade, posição, referência e montagem registrados; gate `--require-metadata` e manifesto JSON versionável testados | Implementado como gate de pesquisa |
| Ficha do equipamento | `pipeline/intake.py`, CLI `validate-intake`, `examples/stream_metadata.bench.synthetic.json` | fabricante, modelo, firmware, bridge, taxa, unidade, canal, referência, montagem, faixa nominal e processamento; gate `--require-intake` testado | Implementado como gate de bancada |
| Auditoria de execução | `stream-json --report` | versão do ambiente, pré-processamento, stride, hash e diagnóstico sem EEG bruto | Preservado para JSONL |
| Expiração de saída stale | `RealtimeEstimator.mark_stale`; antigo `stream-lsl --stale-timeout` | Relato histórico de silêncio LSL; não atribuído ao consumidor JSONL | Núcleo preservado; consumidor LSL removido |
| Escopo seguro do relatório | `cli.py` | sessão marcada como `research_only`, sem decisão clínica ou controle de anestésico | Implementado |
| Dataset para desenvolvimento | `data/figshare.py`, CLI `download-data` | 24 casos Figshare, download com MD5 | Implementado |
| Benchmark VitalDB | `data/vitaldb.py` | quinze casos VitalDB compatíveis normalizados e avaliados sem retreino pelo checkpoint Figshare-only | Avaliação cruzada de dataset; retrospectiva e exploratória |
| Corpus misto | `data/corpus.py`, `build-corpus`, `train-corpus`, `reports/corpus_manifest.json` | pool 23 Figshare + 10 VitalDB; fixed: 18+10=28, split 16/6/6, ajuste 12 Figshare + 4 VitalDB; Figshare por caso, VitalDB por subjectid quando disponível | Candidato experimental |
| Catálogo de datasets | `docs/data_catalog.md`, `docs/mixed_corpus.md` | Figshare/VitalDB integrados em pool auditável; DOSE-I, PhysioNet GABA e Dryad continuam separados por alvo, acesso e compatibilidade | Implementado como pesquisa |
| Proveniência de dados | `data/vitaldb.py`, `data/mat_reader.py`, `signal_diagnostics` | novos NPZ registram origem, unidade, nomes/IDs dos tracks; avaliações registram finitude e imputação offline | Implementado |
| Relatório holdout | `evaluate --report`, `reports/figshare_holdout_evaluation.json` | métricas recalculadas, bootstrap agrupado por caso, cinco casos, configuração, manifesto SHA-256 e ausência de EEG bruto | Reproduzível |
| Relatório VitalDB | `evaluate-external --report`, `reports/vitaldb_external_validation.json` | métricas, bootstrap, diagnósticos, política offline/online, configuração e SHA-256 salvos pela CLI | Benchmark histórico reproduzível; não confirmatório |
| Interface web | `dash_app.py` | Dash + Plotly é a única UI web; inspeção de dados/relatórios locais e replay retrospectivo | Aceite integrado atual pendente |
| Operações locais de pesquisa | CLI `download-data`, `download-vitaldb`, `train`, `train-corpus`, `validate-intake` | Download, treino e validação da ficha permanecem na CLI; LSL removido, não migrado para Dash | Fora das ações da demo; aceite atual pendente |
| Artigo técnico | `docs/tcc_brainsniffer.tex` e `docs/tcc_brainsniffer.pdf` | build Tectonic e inspeção das 14 páginas na rodada de citações acima | Gates documentais passaram; metadados e revisão institucional pendentes |
| Artigo falado | `docs/talk_script.md` | sincronizado ao relato; 804 palavras faladas | 5,36–6,70 min estimados; ensaio humano pendente |
| Decisões fundamentadas | `docs/decisions.md`, `docs/source_ledger.md` | matriz BIS/PSI/Openibis/AnesNET/LSL, fontes, hipóteses e limites de extrapolação | Implementado |
| Registro de protocolo clínico fora do TCC | `docs/_archive/prospective_protocol.md` | documento histórico, não roteiro de execução desta entrega retrospectiva | Fora do escopo definitivo |
| Reprodutibilidade | `docs/model_card.md`, `evaluate`, `evaluate-external`, testes | métricas recalculadas coincidem com checkpoint; ambiente, SHA-256 e seed do bootstrap salvos/verificados | Implementado |
| Sensibilidade EEG-BIS | `evaluate-offset`, `reports/offset_sensitivity.json` | mesma divisão por caso e pesos congelados; grade −20…+20 s executada sem retreino | Exploratório |

## Evidência numérica histórica preservada

No Figshare, com divisão por caso, o checkpoint tem MAE 7,03, RMSE 11,08,
Pearson 0,784 e macro-F1 0,548 em 5.523 janelas de teste. Na avaliação cruzada
de dataset no VitalDB, sem retreino, quinze casos compatíveis produziram MAE
12,43, Pearson 0,024 e macro-F1 0,398 em 38.730 janelas; o bootstrap por caso
teve Pearson 95% entre −0,126 e 0,193. Esse benchmark é retrospectivo e não
equivale a validação externa clínica confirmatória.

O pool misto elegível tem 23 Figshare + 10 VitalDB. O candidato `fixed`
exclui os cinco casos do holdout Figshare histórico e tem 18 Figshare + 10
VitalDB = 28 casos/grupos de desenvolvimento, divididos em 16/6/6 para
treino/validação/teste interno. Somente 12 Figshare + 4 VitalDB ajustam os pesos;
os demais são validação (2+4) e teste interno (4+2), conforme
`models/brainsniffer_corpus_fixed.json`. Figshare é separado por caso, sem
independência por pessoa comprovada; VitalDB por `subject_id` quando disponível,
com fallback por caso prefixado pela fonte. No holdout Figshare fixo, o candidato teve
MAE 6,69 e Pearson 0,818; nos 15 casos VitalDB congelados, teve MAE 8,60 e
Pearson 0,688, contra MAE 12,43 e Pearson 0,024 do checkpoint atual. Esses
números são uma evidência exploratória de adaptação após exposição ao domínio,
registrada em
`reports/mixed_fixed_figshare_holdout.json` e
`reports/mixed_vitaldb_external.json`; como o candidato misto viu outros
participantes VitalDB durante o desenvolvimento, esse resultado é um holdout
histórico da mesma fonte após exposição ao domínio e reutilização para comparação,
não validação externa confirmatória. O checkpoint
ativo continua sendo o original até uma revisão de seeds, partições e outliers.
No holdout Figshare, o bootstrap exploratório por cirurgia (1.000 reamostragens)
teve Pearson médio 0,789 (95%: 0,703–0,881) e MAE médio 7,11 (95%: 6,38–8,24);
esses intervalos refletem apenas cinco casos.
Esses números mostram um protótipo e uma mudança de domínio, não desempenho
clínico.

## Registro de auditoria operacional histórica — não reexecutada

O texto abaixo preserva relatos anteriores, originalmente atribuídos a
2026-09-02 / Python 3.12.14, e acréscimos posteriores sem data individual.
Não se atribui essa data a todos os eventos: por exemplo, o manifesto misto
registra `created_utc=2026-09-04T02:14:50.351695+00:00`. Palavras como “agora”,
“atual” e “confirmou” nos itens abaixo referem-se à execução relatada, não à
revisão presente. Comandos sem log anexado permanecem relatos não revalidados;
os JSON vinculados só sustentam os resultados e configurações neles salvos.
Nenhuma aprovação de build/testes atuais é inferida deste histórico.
Os relatos de abas Explorar/Treinar/EEG ao vivo, controles LSL, download e
avaliação VitalDB pela interface pertencem ao Streamlit **removido**, não ao
Dash. São preservados como auditoria histórica, sem atribuir seus testes à UI
atual e sem reexecução nesta remoção. Também foram removidos o adaptador LSL,
`stream-lsl`, o extra `live` e o publisher. Toda menção a eles neste registro é
histórica, não uma instrução atual; seus resultados não foram reatribuídos ao
JSONL. A fixture `examples/stream_metadata.bench.synthetic.json` mantém a
identidade desse publisher e relógio liblsl, conforme
[live_acquisition.md](_archive/live_acquisition.md), sem promessa de hardware.

- `uv run pytest -q`: 91 testes passaram; `uv run ruff check .`, compilação e
  `uv lock --check` também passaram.
- Um treino CLI de smoke com `--label-offset-seconds 0` concluiu e persistiu o
  parâmetro no manifesto do checkpoint, confirmando a rota de configuração sem
  alterar o alinhamento do modelo oficial.
- O checkpoint oficial foi carregado com verificação de SHA-256:
  `fde8e45fff2fa5414944686fd086fc3dd42248f247c7fb4ea1e31aa00618ee8c`.
- `evaluate` reproduziu exatamente MAE 7,0254, RMSE 11,0849 e Pearson
  0,7837 do checkpoint em 5.523 janelas de teste.
- A mesma execução foi salva em `reports/figshare_holdout_evaluation.json`;
  o relatório registra os cinco casos de teste, seus hashes e a configuração
  efetiva, bootstrap por cirurgia e declara `raw_eeg_in_report=false` sem
  carregar arrays EEG.
- O bootstrap do holdout, com seed 42 e 1.000 reamostragens de casos inteiros,
  estimou Pearson 95% de 0,703–0,881 e MAE 95% de 6,38–8,24; por haver somente
  cinco cirurgias, a incerteza deve ser tratada como exploratória.
- `benchmark-baseline --folds 5` reproduziu a baseline agrupada: MAE 7,91 ±
  0,79, RMSE 11,37 ± 0,94, Pearson 0,745 ± 0,034 e macro-F1 0,518 ± 0,030;
  são métricas da baseline espectral, não da CNN e não são evidência clínica.
- O checkpoint oficial registra o manifesto SHA-256/tamanho dos 24 arquivos de
  entrada; `evaluate` verificou a integridade e confirmou `dataset_summary_match=true`.
  Checkpoints de smoke test também são sinalizados quando têm escopo de janelas
  diferente da avaliação completa.
- `download-vitaldb --overwrite` regenerou os dez arquivos externos iniciais e confirmou
  em todos os casos a proveniência `VitalDB Open Dataset`, unidade `uV`,
  `BIS/EEG1_WAV`, `BIS/BIS` e 128 Hz; casos 12–14, 16 e 17 foram depois
  adicionados com a mesma proveniência, totalizando 15 casos e 38.730 janelas.
- `evaluate-external` nos 15 casos produziu MAE 12,43, Pearson 0,024 e macro-F1
  0,398; o bootstrap agrupado por caso produziu Pearson 95% entre −0,126 e
  0,193 com 1.000 reamostragens e seed 42. O caso 11 foi rejeitado por ausência
  dos dois tracks necessários.
- `evaluate-external --data-dir data/vitaldb` agora descobre exatamente os 15
  arquivos normalizados e registra o manifesto de cada entrada, reduzindo risco
  de uma avaliação externa omitir casos por erro manual; o JSON também registra
  `bootstrap_samples`, `bootstrap_seed` e diagnósticos agregados do sinal bruto.
- A mesma execução foi salva pela CLI em `reports/vitaldb_external_validation.json`;
  o arquivo contém 15 casos, 38.730 janelas, métricas, bootstrap, SHA-256,
  configuração e política offline/online, sem arrays EEG brutos.
- A avaliação externa curta confirmou pontos não finitos nos 15 arquivos VitalDB;
  essa informação agora fica explícita em `input_diagnostics` e não é confundida
  com uma captura online válida.
- O relatório externo também declara `data_handling` em formato machine-readable:
  imputação linear somente offline, rejeição online antes do filtro/resampler e
  ausência de EEG bruto no relatório.
- `evaluate-offset` executou nove offsets entre −20 e +20 s nos mesmos cinco casos
  do holdout, sem retreinar: o MAE variou de 7,22 a 6,76 e Pearson de 0,766 a
  0,799. O relatório salva a grade, o manifesto dos arquivos e os pesos congelados;
  a análise continua pós-hoc e não fixa um offset clínico.
- Duas execuções externas com 20 reamostragens e seed 42 produziram o mesmo
  SHA-256 de saída (`2226a123df096e00ebaebdea0182b5e918443ed0520a0dff389e943bedfa924a`),
  confirmando a reprodutibilidade do bootstrap sob o ambiente atual.
- Um stream sintético de 1.408 amostras a 256 Hz foi reamostrado para 128 Hz e
  produziu uma predição JSON com qualidade 1,0 e timestamp de origem; o
  `audit-json` do mesmo stream retornou `ok=true`.
- Um publisher LSL sintético real a 128 Hz foi conectado pelo comando
  `stream-lsl --require-metadata`; o processo recebeu 712 amostras, emitiu 1
  predição causal, preservou timestamps monotônicos, importou do XML a unidade,
  canal, referência e montagem, confirmou a taxa 128 Hz e fechou com relatório
  `ok=true`, qualidade 1,0 e o SHA-256 atual do checkpoint.
- O publisher reproduzível `examples/lsl_synthetic_publisher.py` também foi
  executado isoladamente a 64 Hz por 1 s e encerrou com sucesso, validando a
  ferramenta de bancada sem envolver dados de paciente.
- Um preflight com timestamp não finito retornou `ok=false` e registrou o
  contador correspondente, sem permitir que o stream fosse aceito.
- O mesmo preflight agora retorna código de saída 1 quando `ok=false`, permitindo
  interromper um gate automatizado antes da inferência ou do estudo.
- `stream-json --fail-on-audit` foi testado com uma janela em linha plana: salvou
  o relatório parcial como erro, interrompeu antes da inferência e não tratou a
  sessão reprovada como concluída.
- O fluxo ponta a ponta foi repetido com publisher LSL sintético por 5 s e
  `--metadata-file --require-metadata --fail-on-audit`: 641 amostras, uma
  predição, metadata completo, `audit.ok=true` e relatório `completed`.
- A execução atualizada do publisher a 256 Hz durante oito segundos foi
  reamostrada para 128 Hz pelo consumidor LSL: 2.054 amostras, 40 chunks e
  três predições causais, com timestamps presentes, qualidade mínima 1,0 e
  `audit.ok=true` e zero `stale_abstentions`. O relatório está em
  `reports/lsl_synthetic_session.json`;
  continua sendo um teste de integração sintético, não um teste em paciente.
- O mesmo fluxo foi repetido com `--require-intake` e o manifesto
  `examples/stream_metadata.bench.synthetic.json`: 2.054 amostras, 40 chunks,
  três predições, `intake.status=ready_for_bench`, `audit.ok=true` e nenhum
  conflito de taxa 256→128 Hz. O relatório está em
  `reports/lsl_synthetic_intake_session.json`; `ready_for_bench` aqui só prova
  o contrato do gate com uma fixture sintética.
- Um stream JSON sem unidade, canal, referência e montagem foi bloqueado antes da
  inferência com `--require-metadata`; o relatório registrou os campos ausentes.
- Um manifesto JSON versionado foi carregado por `stream-json`; flags e valores do
  manifesto foram combinados sem sobrescrita silenciosa e conflitos foram rejeitados.
- O adaptador LSL foi testado com um descritor XML sintético contendo rótulo,
  unidade, referência e montagem; esses campos passam automaticamente ao
  manifesto sem depender de redigitação.
- A auditoria rejeita metadata com taxa inválida ou divergente da taxa efetiva
  dos chunks, evitando que o filtro/resampler opere sob uma declaração falsa.
- O relatório JSON do stream inclui ambiente, configuração efetiva do
  pré-processamento, stride e `report_version=2`, além do manifesto, sem carregar
  amostras EEG; também registra as flags `require_metadata`, `require_timestamps`
  e `fail_on_audit` efetivamente usadas.
- O relatório também fixa o escopo `research_only` e dois flags falsos de segurança
  (`clinical_decision_support` e `controls_anesthetic_delivery`).
- Uma falha de descoberta LSL foi simulada e o processo preservou o relatório
  parcial com o erro e o manifesto informado, sem declarar a sessão concluída.
- `stream-json --report` gerou um relatório de sessão sem campo de amostras EEG,
  registrando 1.408 amostras auditadas, uma predição, qualidade 1,0 e o mesmo
  SHA-256 do checkpoint.
- O servidor Dash/Gunicorn local respondeu `status=ok` em `/healthz`, e o layout
  carregou a aba Corpus; na execução histórica, o Streamlit (hoje removido)
  respondeu em `/_stcore/health`. Esse endpoint não integra a interface atual.
- A interface foi aberta no navegador integrado e as abas Dados, Explorar, Treinar,
  Replay em fluxo e EEG ao vivo (LSL) exibiram seus controles e painéis sem
  erro de renderização.
- A aba LSL da interface foi configurada para exigir timestamps válidos por amostra
  e registrar essa exigência no relatório da sessão.
- A interface também oferece `fail_on_audit` ativado por padrão, registrando a
  escolha no relatório e evitando marcar uma captura reprovada como concluída.
- A aba LSL valida a ficha técnica antes da conexão e oferece download do mesmo
  manifesto JSON que será aplicado ao stream.
- `--max-gap-factor` agora é compartilhado por preflight e streams JSON/LSL e fica
  registrado no relatório, evitando critérios de temporização divergentes.
- A interface foi exercitada com quinze arquivos VitalDB locais: mostrou 38.730
  janelas, MAE 12,43, Pearson 0,024, a tabela de bootstrap e o botão de
  download do relatório JSON.
- O relatório VitalDB baixado pela interface agora declara o mesmo escopo
  `research_only`, configuração de pré-processamento e ausência de EEG bruto
  usados no relatório do CLI.
- `replay --case 1 --data-dir data/raw` percorreu a gravação Figshare inteira
  pelo estimador causal e emitiu a última saída com o hash do checkpoint. Uma
  tentativa equivalente no VitalDB encontrou 164 amostras não finitas e foi
  rejeitada antes do modelo, confirmando o comportamento fail-closed do caminho
  online; o VitalDB continua avaliado pelo caminho offline documentado.
- O benchmark de 30 iterações mediu p50 0,57 ms e p95 0,79 ms na execução
  histórica; repetições recentes mediram p50 entre 0,66 e 0,74 ms e p95 entre
  1,26 e 2,96 ms. A variação é do host e tudo isso é custo de software, não
  latência clínica.
- A rodada de aceite atual passou um JSONL de 1.408 amostras com sete predições
  e um LSL sintético de 640 amostras a 128 Hz com uma predição; ambos exigiram
  metadata/timestamps e terminaram com `audit.ok=true` e escopo `research_only`.
- `validate-intake` foi exercitado com manifesto incompleto e completo; o primeiro
  retornou código 1 com campos ausentes e o segundo retornou
  `ready_for_bench=true`. O gate `stream-json --require-intake` aceitou somente
  o manifesto completo e registrou a decisão no relatório.
- `validate-intake` também rejeita unidade `mV` como incompatível: o bridge deve
  converter para microvolt antes do filtro, sem conversão silenciosa no modelo.
- Com `--fail-on-audit`, o stream JSON agora rejeita uma lacuna de timestamp antes
  de chamar o modelo; o relatório parcial registra a lacuna e zero predições.
- O LSL agora trata silêncio explicitamente com `--stale-timeout`: em modo
  fail-closed, a sessão é rejeitada; em modo exploratório, a última estimativa
  vira `ABSTAIN` e o buffer/filtro causal é reiniciado. O caso de silêncio após
  dados foi coberto por teste unitário e de CLI.
- A inspeção do ambiente em 2026-09-02 não encontrou um EEG físico identificável
  nem portas `/dev/ttyUSB*` ou `/dev/ttyACM*`; portanto, a integração atual
  permanece comprovada somente com LSL sintético e bridge JSON de bancada.

## Limites metodológicos e reprodução

- Reavaliar os pesos congelados com configuração, partições e hashes salvos é
  diferente de reproduzir seu treino. A receita histórica nos JSON não comprova
  uso de recursos posteriormente adicionados ao loop; treinar no código atual
  é novo experimento e não deve sobrescrever os artefatos históricos.
- `RobustConv1DDepthEstimator` é API experimental não integrada à CLI e sem
  incerteza validada; não é a arquitetura dos checkpoints comparados.
- SQI é heurístico, não confiança clínica, probabilidade de acerto ou incerteza
  calibrada. Interpolação linear de não finitos na avaliação offline pode usar
  amostras futuras; não equivale à rejeição online antes do filtro/resampler.
  Gates do pool misto não certificam os 15 casos do holdout histórico.
- Com offset zero, o alvo refere-se ao início da janela de 5 s e a emissão online
  ocorre ao final. Filtro causal não remove essa diferença ou o atraso do BIS.
- `reports/mixed_internal_holdout.json` referencia `brainsniffer_corpus.pt`,
  não o candidato `fixed`; não transferir suas métricas entre artefatos.

## Limites gerais do software — não requisitos de aquisição do TCC

- Testes/build do sistema atual permanecem pendentes de verificações separadas.
  O build documental e a sincronização do roteiro foram conferidos nesta rodada;
  aceite institucional e ensaio cronometrado continuam pendentes.
- Reprodução do treino histórico e equivalência offline/online ponta a ponta.
- Compatibilidade com um EEG físico específico: fabricante, SDK/protocolo,
  unidade, referência, montagem, canal e timestamps.
- Fidelidade do bridge, latência ponta a ponta, perdas e reconexão no centro
  cirúrgico.
- Generalização definitiva para outro aparelho, centro, fármaco e população.
- SQI clínico calibrado, referência complementar ao BIS e política de abstenção
  prospectiva.
- Segurança, ética, proteção de dados e avaliação regulatória para qualquer uso
  com pacientes.

## Rodada de revisão editorial — 24/09/2026

- **Dados:** os 15 derivados `data/vitaldb/*.npz` saíram do índice e do histórico Git
  (reescrita + push forçado); a cópia local permanece ignorada. O repositório público
  não redistribui VitalDB; o README documenta a política. A ref local `refs/codex/*`,
  que retinha os objetos antigos, foi removida.
- **Artigo:** fechado em 17 páginas (sem Overfull) após ajustes de revisão: seção de
  Disponibilidade, ética assertiva, tabelas de benchmark fundidas com ICs, Fig. 7a com
  categorias mutuamente exclusivas (59 casos) e Fig. 8b trocada por calibração; saíram
  a projeção teórica do corpo e o duplicado de PhysioNet da bibliografia (24 entradas,
  todas citadas; entrou Johansen \& Sebel 2000).
- **Reanálises (sem retreino):** `scripts/reviewer_reanalysis.py` gera
  `reports/paired_bootstrap.json`, `reports/calibration_analysis.json`,
  `reports/pk_audit.json` e `reports/spectral_baseline_vitaldb.json` a partir de
  predições por janela em `tmp/reanalysis/` (não versionadas). Resultados centrais:
  bootstrap pareado por caso do $\Delta$MAE/$\Delta P_K$, baseline espectral também
  sem transporte no VitalDB (MAE 13,62; $P_K$ 0,502) e calibração por
  inclinação/ICC/$|erro|\leq10$.
- **Pendências:** determinismo da segunda execução em verificação; pareceres
  simulados arquivados fora do repositório (temp); hash do commit final a fixar na
  seção de Disponibilidade; paridade site$\leftrightarrow$artigo revisada em seguida.

## Critério de conclusão do TCC retrospectivo

A entrega é o artigo documentado, as comparações existentes e a trajetória
ilustrativa reproduzível de uma gravação, com limites explícitos. Não há etapa
para avançar ao hardware. Fichas de equipamento e relatos LSL preservados neste
histórico não condicionam a conclusão nem autorizam aquisição. Cabe ao autor e
à instituição resolver metadados, termos de uso e enquadramento ético dos dados
secundários. Nenhuma saída deve recomendar dose ou conduta clínica.
