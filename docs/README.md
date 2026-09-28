# Documentação — índice

Entrada: [`../OVERVIEW.md`](../OVERVIEW.md) (visão geral) e [`../README.md`](../README.md) (manual operacional).

## Artigo (`artigo/`)

- `artigo/README.md` — como construir o artigo: receita de build (Tectonic), gates e render de páginas.
- `artigo/tcc_brainsniffer.tex` + `artigo/referencias.bib` → `artigo/tcc_brainsniffer.pdf` (fonte e PDF do TCC, formato SBC). Build: `scripts/build_tcc_article.sh`.
- `artigo/figures/` — figuras do artigo em PDF vetorial + PNG + `README.md` (proveniência).
- `artigo/generate_figures.py` — regenera as figuras a partir de `reports/` e `models/` (padrão sem inferência/download).
- `artigo/latex/` — `sbc-template.sty`, `sbc.bst` e `caption2.sty`.
- `artigo/historico/` — versão antiga de 19 páginas do TCC (`tcc_brainsniffer_19pp.tex`/`.pdf`), mantida como snapshot.

## Notas (`notas/`)

- `notas/article.md` — nota de contexto (não é segundo manuscrito).
- `notas/talk_script.md` — roteiro de apresentação.
- `notas/data_catalog.md` — catálogo de datasets e compatibilidade.
- `notas/mixed_corpus.md` — protocolo e resultado do corpus misto.
- `notas/vitaldb_external_validation.md` — avaliação externa exploratória com VitalDB.
- `notas/source_ledger.md` — ledger fontes ↔ decisões ↔ limites.
- `notas/decisions.md` — registro de decisões técnicas (D1–D21).
- `notas/model_card.md` — uso pretendido, métricas, riscos.
- `notas/project_status.md` — matriz requisitos/evidências/lacunas.

## Submissão JHI (`submissao-jhi/`)

- `submissao-jhi/build_jhi_docx.py` — converte o artigo do TCC (LaTeX/SBC) nos DOCX da submissão: manuscrito, material complementar e cadastro de autores.
- `submissao-jhi/fill_checklist.py` — preenche o checklist de conformidade da JHI a partir do manuscrito gerado.
- `submissao-jhi/jhi_brainsniffer.docx` — manuscrito no formato da revista.
- `submissao-jhi/jhi_material_complementar.docx` — figuras e tabelas deslocadas para material complementar.
- `submissao-jhi/jhi_cadastro_autores.docx` — ficha de cadastro de autores na plataforma.
- `submissao-jhi/jhi_checklist_conformidade.docx` — checklist de conformidade preenchido.

## Arquivo morto (`_archive/`)

Docs fora do escopo do TCC ou históricos, mantidos para auditoria. Ver `_archive/README.md`.
