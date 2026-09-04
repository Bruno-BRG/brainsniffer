# Status e rastreabilidade do BrainSniffer

Este documento liga o pedido original aos artefatos e às evidências verificadas
no workspace. O status “implementado” significa que o caminho de pesquisa foi
executado; não significa autorização ou validação clínica.

| Requisito | Artefato principal | Evidência atual | Status |
|---|---|---|---|
| Python 3.12 | `pyproject.toml`, `.python-version` | `requires-python >=3.12,<3.13`; compileall | Implementado |
| CNN para EEG | `src/brainsniffer/models/cnn.py` | checkpoint `models/brainsniffer_cnn.pt` carregado e avaliado | Implementado |
| Estimar referência BIS/estágio | `data/preprocess.py`, `pipeline/metrics.py` | regressão 0–100 e quatro faixas de pesquisa | Implementado como pesquisa |
| Inferência em fluxo | `pipeline/realtime.py` | replay causal e emissão a cada stride | Implementado |
| Entrada de aquisição | `pipeline/streaming.py`, CLI `stream-lsl`/`stream-json` | smoke 256→128 Hz com timestamps; ficha de hardware em `docs/real_eeg_intake.md` | Bridge vendor-neutral |
| Publisher de bancada | `examples/lsl_synthetic_publisher.py` | stream LSL sintético com metadata nativo para reproduzir o smoke test | Pesquisa |
| Preflight do bridge | `pipeline/stream_audit.py`, CLI `audit-json` | stream bom/ruim testado no executável | Implementado |
| Manifesto do sinal | `pipeline/stream_audit.py`, `stream-json`, `stream-lsl`, `app.py`, `examples/stream_metadata.template.json` | unidade, posição, referência e montagem registrados; gate `--require-metadata` e manifesto JSON versionável testados | Implementado como gate de pesquisa |
| Ficha do equipamento | `pipeline/intake.py`, `validate-intake`, `app.py`, `examples/stream_metadata.bench.synthetic.json` | fabricante, modelo, firmware, bridge, taxa, unidade, canal, referência, montagem, faixa nominal e processamento; gate `--require-intake` testado | Implementado como gate de bancada |
| Auditoria de execução | `stream-json --report`, `stream-lsl --report`, interface LSL | versão do ambiente, pré-processamento, stride, hash e diagnóstico sem EEG bruto | Implementado |
| Expiração de saída stale | `RealtimeEstimator.mark_stale`, `stream-lsl --stale-timeout`, interface LSL | silêncio invalida a última saída; modo fail-closed preserva relatório parcial; buffer causal é reiniciado | Implementado como gate de engenharia |
| Escopo seguro do relatório | `cli.py`, `app.py` | sessão marcada como `research_only`, sem decisão clínica ou controle de anestésico | Implementado |
| Dataset para desenvolvimento | `data/figshare.py`, `app.py` | 24 casos Figshare, download com MD5 | Implementado |
| Dataset externo | `data/vitaldb.py` | quinze casos VitalDB compatíveis normalizados e avaliados sem retreino | Holdout externo congelado |
| Corpus misto | `data/corpus.py`, `build-corpus`, `train-corpus`, `reports/corpus_manifest.json` | 23 Figshare + 10 VitalDB elegíveis; gates de finitude/lacuna/BIS/qualidade; split por subjectid quando disponível | Candidato experimental |
| Catálogo de datasets | `docs/data_catalog.md`, `docs/mixed_corpus.md` | Figshare/VitalDB integrados em pool auditável; DOSE-I, PhysioNet GABA e Dryad continuam separados por alvo, acesso e compatibilidade | Implementado como pesquisa |
| Proveniência de dados | `data/vitaldb.py`, `data/mat_reader.py`, `signal_diagnostics` | novos NPZ registram origem, unidade, nomes/IDs dos tracks; avaliações registram finitude e imputação offline | Implementado |
| Relatório holdout | `evaluate --report`, `reports/figshare_holdout_evaluation.json` | métricas recalculadas, bootstrap agrupado por caso, cinco casos, configuração, manifesto SHA-256 e ausência de EEG bruto | Reproduzível |
| Relatório externo | `evaluate-external --report`, `reports/vitaldb_external_validation.json` | métricas, bootstrap, diagnósticos, política offline/online, configuração e SHA-256 salvos pela CLI | Exploratório reproduzível |
| Interface | `dash_app.py`, `app.py` | Dash + Plotly publicado com abas Dados, Trajetória, Replay, Corpus, Estatística e Método; Streamlit legado mantém os fluxos de pesquisa | Implementado |
| Artigo técnico | `docs/tcc_brainsniffer.tex` e `docs/tcc_brainsniffer.pdf` | método, resultados, limitações, figuras e referências | Revisado; PDF final com 11 páginas, dentro do limite de 15, e citações nos parágrafos |
| Artigo falado | `docs/talk_script.md` | roteiro de 5–7 minutos sincronizado com os resultados | Implementado |
| Decisões fundamentadas | `docs/decisions.md`, `docs/source_ledger.md` | matriz BIS/PSI/Openibis/AnesNET/LSL, fontes, hipóteses e limites de extrapolação | Implementado |
| Plano de avanço clínico | `docs/prospective_protocol.md` | gates de bancada, validação externa travada e modo sombra | Preparatório |
| Reprodutibilidade | `docs/model_card.md`, `evaluate`, `evaluate-external`, testes | métricas recalculadas coincidem com checkpoint; ambiente, SHA-256 e seed do bootstrap salvos/verificados | Implementado |
| Sensibilidade EEG-BIS | `evaluate-offset`, `reports/offset_sensitivity.json` | mesma divisão por caso e pesos congelados; grade −20…+20 s executada sem retreino | Exploratório |

## Evidência numérica atual

No Figshare, com divisão por caso, o checkpoint tem MAE 7,03, RMSE 11,08,
Pearson 0,784 e macro-F1 0,548 em 5.523 janelas de teste. No VitalDB, sem
retreino, quinze casos compatíveis produziram MAE 12,43, Pearson 0,024 e
macro-F1 0,398 em 38.730 janelas; o bootstrap por caso teve Pearson 95% entre
−0,126 e 0,193.

O candidato do corpus misto foi ajustado com 28 casos de desenvolvimento,
excluindo exatamente os cinco casos do holdout Figshare histórico e usando 10
casos VitalDB aprovados pelo gate. No holdout Figshare fixo, o candidato teve
MAE 6,69 e Pearson 0,818; nos 15 casos VitalDB congelados, teve MAE 8,60 e
Pearson 0,688, contra MAE 12,43 e Pearson 0,024 do checkpoint atual. Esses
números são uma evidência exploratória de melhora de domínio, registrada em
`reports/mixed_fixed_figshare_holdout.json` e
`reports/mixed_vitaldb_external.json`; o checkpoint ativo continua sendo o
original até uma revisão de seeds, partições e outliers.
No holdout Figshare, o bootstrap exploratório por cirurgia (1.000 reamostragens)
teve Pearson médio 0,789 (95%: 0,703–0,881) e MAE médio 7,11 (95%: 6,38–8,24);
esses intervalos refletem apenas cinco casos.
Esses números mostram um protótipo e uma mudança de domínio, não desempenho
clínico.

## Auditoria operacional atual

Executada em 2026-09-02 no ambiente Python 3.12.14:

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
  carregou a aba Corpus; o Streamlit legado continua respondendo em `/_stcore/health`.
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

## O que ainda não está comprovado

- Compatibilidade com um EEG físico específico: fabricante, SDK/protocolo,
  unidade, referência, montagem, canal e timestamps.
- Fidelidade do bridge, latência ponta a ponta, perdas e reconexão no centro
  cirúrgico.
- Generalização definitiva para outro aparelho, centro, fármaco e população.
- SQI clínico calibrado, referência complementar ao BIS e política de abstenção
  prospectiva.
- Segurança, ética, proteção de dados e avaliação regulatória para qualquer uso
  com pacientes.

## Critério para avançar ao hardware

O laboratório deve preencher [`docs/real_eeg_intake.md`](real_eeg_intake.md) e
fornecer uma gravação anonimizada ou stream de bancada do equipamento, junto com
sua documentação de taxa/unidade/montagem. Primeiro roda-se `audit-json` ou LSL
em bancada; depois compara-se a distribuição do sinal com o treino; somente então
se mede inferência e latência. Nenhuma saída do protótipo deve controlar ou
recomendar dose anestésica.
