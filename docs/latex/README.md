# Artigo LaTeX no formato SBC

O arquivo principal do artigo é [`../tcc_brainsniffer.tex`](../tcc_brainsniffer.tex). A lista editável de referências permanece em [`../referencias.bib`](../referencias.bib); o PDF entregue usa uma bibliografia manual no próprio `.tex` para manter a compilação com Tectonic determinística e sem dependência de uma etapa externa do BibTeX. Os arquivos `sbc-template.sty`, `sbc.bst` e `caption2.sty` acompanham o projeto para que o artigo possa ser compilado localmente ou enviado ao Overleaf.

## Compilação local

Com `tectonic`:

```bash
cd docs
TEXINPUTS="./latex:" tectonic tcc_brainsniffer.tex
```

Com `pdflatex` e `bibtex`:

```bash
cd docs
TEXINPUTS="./latex:" BIBINPUTS="./latex:" pdflatex tcc_brainsniffer.tex
TEXINPUTS="./latex:" BIBINPUTS="./latex:" bibtex tcc_brainsniffer
TEXINPUTS="./latex:" BIBINPUTS="./latex:" pdflatex tcc_brainsniffer.tex
TEXINPUTS="./latex:" BIBINPUTS="./latex:" pdflatex tcc_brainsniffer.tex
```

Antes da entrega, substitua no `.tex` o nome do autor, instituição, curso, cidade, estado e e-mail. O texto usa os resultados de engenharia atualmente registrados no projeto; resultados futuros devem atualizar o método, a tabela e os relatórios JSON de forma conjunta.
