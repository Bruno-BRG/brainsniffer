# BrainSniffer — Visão Geral

**Título do artigo (TCC):** BrainSniffer: protótipo reproduzível para estimar uma referência BIS a partir de EEG frontal

Fonte do artigo: `docs/tcc_brainsniffer.tex` (formato SBC) + `docs/referencias.bib` + PDF em `docs/tcc_brainsniffer.pdf`.

## O que estamos fazendo

Protótipo de engenharia reproduzível que estima, a partir de EEG frontal, um **rótulo de referência BIS** (Índice Bispectral, saída de monitor, 0–100). Não mede consciência, não recomenda dose, não aciona alarme/conduta, não é dispositivo médico.

Pergunta de pesquisa: quanto uma CNN compacta aproxima o rótulo BIS em casos não usados no ajuste, comparada a um baseline espectral, e como isso muda entre fontes (Figshare → VitalDB)?

## Como funciona

1. **Entrada:** janela de 5 s a 128 Hz = 640 amostras, 1 canal frontal, passo 5 s offline. Alvo = BIS do início da janela (`offset` padrão 0 s, quantizado a 5 s no Figshare e 1 s no VitalDB normalizado).
2. **Pré-processamento causal:** banda 0,5–45 Hz (4ª ordem), notch 50 Hz (Q=30), clip ±100 µV, escala 50 µV, saída limitada a [-5,5]. Estado preservado por caso/chunk. Offline existe imputação linear de não-finitos (documentada como limitação); no replay/stream, não-finitos são rejeitados.
3. **Gates de qualidade:** heurística própria (EEG finito ≥90%, lacuna ≤2 s, qualidade global ≥0,35, BIS válido ≥80%, qualidade/janela ≥0,20). Caso Figshare 24 em quarentena por escala incompatível. Abaixo de 0,20 → abstenção (`abstain`).
4. **Modelos:**
   - Ativo (Modelo A): `Conv1DDepthEstimator` treinado só no Figshare (13/5/5 casos, seed 42).
   - Misto (Modelo B): mesmo código, treinado com Figshare + parte do VitalDB (28 grupos, 16/6/6; ajuste usa 12 Figshare + 4 VitalDB).
   - Baseline: potência absoluta/relativa delta/teta/alfa/beta/gama + SEF90 + entropia espectral + RMS + line length → Random Forest (`n_estimators=100, max_depth=12, min_samples_leaf=2`).
5. **Arquitetura CNN (`Conv1DDepthEstimator`):** `Conv1d` channels 1-32-64-128-128, `kernel_size` 7,7,5,5, `padding` 3,3,2,2, `stride` 1, `BatchNorm1d` + GELU por bloco, `MaxPool1d(2)` nos 3 primeiros blocos, `AdaptiveAvgPool1d(1)`, `Linear` 128-64-1 + GELU + Dropout 0.20, saída `100 * sigmoid`. Treino: 10 epochs, `batch_size` 128, `learning_rate` 1e-3, `weight_decay` 1e-4. Código em `src/brainsniffer/models/cnn.py`.
6. **Avaliação:** split por grupo (Figshare = caso, VitalDB = `subjectid`), métricas pooled por janela (MAE/RMSE/viés/Pearson + acurácia/macro-F1 em 4 faixas), bootstrap por caso (B=1000, seed 42), sensibilidade a offset (-20…+20 s, sem retreino). Resultados são retrospectivos/exploratórios.
7. **Replay:** percorre gravação existente (buffer 5 s, emite a cada 1 s, separa bruto/EWMA/qualidade/timestamp). Sem EEG físico neste TCC.

Resultados históricos (5.523 janelas Figshare, 38.730 VitalDB): Figshare CNN ativa 7,03/0,784 vs baseline 8,86/0,667; misto 6,69/0,818. VitalDB: ativa 12,43/0,024, misto 8,60/0,688 (14/15 casos melhoram, caso 16 piora ~0,67). Perda de associação da ativa no VitalDB é o resultado negativo central.

## Datasets e créditos (conferido na web em 14/09/2026)

| Dado | Licença / termos | O que fazer |
|---|---|---|
| Figshare `EEG and BIS raw data` (Ma, 2017, DOI `10.6084/m9.figshare.5589841.v1`, v1, 24 `.mat`) | CC BY 4.0 | Citar Ma 2017 + DOI + indicar alterações (quarentena caso 24). Download via API oficial com MD5. Sem redistribuir `.mat`. |
| VitalDB Open Dataset (Lee et al. 2022 Sci Data + `vitaldb.net/dataset` + `clinical_data.csv` PhysioNet v1.0.0) | `vitaldb.net`: CC BY-NC-SA 4.0 + Data Use Agreement + citação obrigatória a Lee et al. 2022. Espelho PhysioNet tem `LICENSE.txt` CC BY 4.0 — vale o regime mais restritivo do provedor p/ tracks via API | Uso não-comercial, seletivo (`BIS/EEG1_WAV`, `BIS/BIS`), sem redistribuir corpus. Guardar só `.npz` normalizado + métricas. Agrupar por `subjectid`. Não chamar de validação externa confirmatória (é holdout histórico / avaliação cruzada). |
| DOSE-I (Zenodo `10.5281/zenodo.18483292`, 171 gravações) | CC BY 4.0 + `Data_Use_Agreement.txt` | Citado, **não usado** no treino (alvo MOAA/S ≠ BIS). Não converter rótulo. |
| Código BrainSniffer | MIT (`LICENSE`, `CITATION.cff`) | MIT não relicencia dados/pesos derivados. |

Detalhes: seção `Disponibilidade de dados e código` + `Ética` no `.tex`, `docs/data_catalog.md`, `docs/source_ledger.md`, `docs/_archive/research/reporting_ethics_checklist.md`. Antes de submeter: reconfirmar versão/termos e definir com a instituição se uso secundário precisa de parecer/dispensa (não presumir dispensa por ser público).

## Tecnologias

- **Linguagem/ambiente:** Python 3.12 (`.python-version`), `uv` + `uv.lock`, Hatchling, `pyproject.toml`.
- **ML/sinal:** PyTorch ≥2.4 (CNN + `sigmoid`), scikit-learn ≥1.5 (Random Forest), SciPy ≥1.14 (filtros `sosfilt`, resampler), NumPy ≥2.1, pandas ≥2.2, h5py ≥3.10 (MAT v7.3).
- **Interface:** Dash ≥2.18 + Plotly ≥5.24 (`dash_app.py`, única interface web), gunicorn ≥23, `assets/`.
- **Qualidade:** pytest ≥8.3 (`tests/`), ruff (lint, line-length 100), GitHub Actions CI, Dockerfile (Python 3.12, `uv sync --locked`).
- **Docs/artigo:** LaTeX template SBC (`docs/latex/`, `docs/tcc_brainsniffer.tex`, `docs/referencias.bib`, `scripts/build_tcc_article.sh`), Markdown de apoio (`docs/article.md`, `model_card.md`, `decisions.md`, etc.), `docs/generate_figures.py` + matplotlib (figuras `pipeline.pdf`, `bis_trajectory.pdf`, `comparison.pdf`, `bootstrap_intervals.pdf`, `offset_sensitivity.pdf`).

## Estrutura do repositório

```
brainsniffer/
├── OVERVIEW.md               ← este arquivo
├── README.md                 ← manual operacional completo
├── CITATION.cff / LICENSE   ← citação software / MIT (só código)
├── pyproject.toml / uv.lock / .python-version / Dockerfile
├── dash_app.py / assets/     ← dashboard (replay, trajetória, resultados, corpus)
├── src/brainsniffer/
│   ├── cli.py / config.py
│   ├── data/                 ← figshare.py, vitaldb.py, mat_reader.py, preprocess.py, corpus.py, split.py
│   ├── models/               ← cnn.py (Conv1DDepthEstimator + RobustConv1DDepthEstimator experimental)
│   └── pipeline/             ← training.py, metrics.py, baseline.py, realtime.py, streaming.py, stream_audit.py, benchmark.py, intake.py
├── docs/
│   ├── tcc_brainsniffer.tex / referencias.bib / tcc_brainsniffer.pdf  ← artigo
│   ├── article.md / model_card.md / decisions.md / source_ledger.md / data_catalog.md
│   ├── mixed_corpus.md / vitaldb_external_validation.md / talk_script.md
│   ├── project_status.md / model_card.md / decisions.md
│   ├── latex/ / figures/ / generate_figures.py
│   └── _archive/ (research/, prospective_protocol, live_acquisition, real_eeg_intake)
├── models/                   ← .pt + .json (ativo, misto fixed, smoke)
├── reports/                  ← corpus_manifest.json, *_holdout*.json, *_external*.json, offset_sensitivity.json
├── data/raw/ / data/vitaldb/ / data/vitaldb_train/  ← dados locais (não redistribuir; .gitignore não apaga histórico)
├── tests/                    ← 18 arquivos (cli, corpus, split, preprocess, metrics, vitaldb, streaming, etc.)
├── examples/                 ← stream_metadata.template.json + sintéticos
└── scripts/build_tcc_article.sh
```

## Rodar rápido

```bash
uv sync --locked --extra dev
uv run --locked --extra dev pytest -q
uv run brainsniffer download-data            # Figshare → data/raw/
uv run brainsniffer download-vitaldb --case 1 --out data/vitaldb
uv run brainsniffer build-corpus --out reports/corpus_manifest.json
uv run brainsniffer evaluate --checkpoint models/brainsniffer_cnn.pt --report reports/figshare_holdout_evaluation.json --bootstrap-samples 1000 --bootstrap-seed 42
uv run python -c 'from dash_app import app; app.run(host="127.0.0.1", port=8501, debug=False)'
# artigo:
bash scripts/build_tcc_article.sh  # ver docs/latex/README.md
```
