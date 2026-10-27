# Versão completa (antes de enxugar)

Cópia das versões do TCC, dos slides e do roteiro **antes** de remover o baseline espectral, o experimento fatorial e o slide de ritmos. Nada aqui foi alterado depois da cópia.

| Pasta | Conteúdo |
|-------|----------|
| `artigo/` | `tcc_brainsniffer.tex` e `.pdf` (17 páginas), `referencias.bib`, `figures/` (9 figuras originais), `latex/` e `generate_figures.py` originais. Compila sozinha a partir desta pasta. |
| `slides/deck_fonte/` | Código-fonte do deck de 26 slides (`deck.json` e `slides/*.html`) com as notas do roteiro de 17 min. |
| `slides/TCC – BrainSniffer.pptx` | Deck antigo, anterior ao template UniJorge. |
| `roteiro/roteiro_apresentacao_17min.md` | Roteiro de 26 slides, com baseline e fatorial. |

**Slides publicados:** o deck online é o mesmo link de sempre e agora mostra a versão enxuta (24 slides). A versão completa corresponde à versão 26 do artefato; o fonte dela está em `slides/deck_fonte/`.

**Compilar o artigo completo:**

```bash
cd docs/versao_completa/artigo
pdflatex tcc_brainsniffer.tex && BSTINPUTS="latex//;" bibtex tcc_brainsniffer && pdflatex tcc_brainsniffer.tex && pdflatex tcc_brainsniffer.tex
```

**Versão enxuta (atual):** `docs/artigo/` (artigo e roteiro) e o link do deck.
