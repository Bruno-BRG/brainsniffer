# Documentação — índice

Entrada: [`../OVERVIEW.md`](../OVERVIEW.md) (visão geral) e [`../README.md`](../README.md) (manual operacional).

## Artigo

- `tcc_brainsniffer.tex` + `referencias.bib` → `tcc_brainsniffer.pdf` (fonte do TCC, formato SBC). Build: `scripts/build_tcc_article.sh`, detalhes em `latex/README.md`.
- `article.md` — nota de contexto (não é segundo manuscrito).
- `talk_script.md` — roteiro de apresentação.

## Dados e método (escopo do TCC)

- `data_catalog.md` — catálogo de datasets e compatibilidade.
- `mixed_corpus.md` — protocolo e resultado do corpus misto.
- `vitaldb_external_validation.md` — avaliação externa exploratória com VitalDB.
- `source_ledger.md` — ledger fontes ↔ decisões ↔ limites.
- `decisions.md` — registro de decisões técnicas (D1–D21).

## Modelo e operação

- `model_card.md` — uso pretendido, métricas, riscos.
- `project_status.md` — matriz requisitos/evidências/lacunas.

## Figuras

- `figures/` — PDFs usados no artigo + PNGs + `README.md` (proveniência).
- `generate_figures.py` — regenera figuras a partir de `reports/` e `models/` (padrão sem inferência/download).

## Arquivo morto (`_archive/`)

Docs fora do escopo do TCC ou históricos, mantidos para auditoria. Ver `_archive/README.md`.
