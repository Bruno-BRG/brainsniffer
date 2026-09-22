# Artigo LaTeX no formato SBC

O artigo principal é [`../tcc_brainsniffer.tex`](../tcc_brainsniffer.tex) e a
única fonte bibliográfica é [`../referencias.bib`](../referencias.bib). O texto
usa `\\bibliographystyle{sbc}` e `\\bibliography{referencias}`; não há uma
segunda bibliografia manual no `.tex`. `sbc-template.sty`, `sbc.bst` e
`caption2.sty` acompanham o repositório.

## Receita de build e verificações

Dependências externas (não instaladas pelo ambiente Python): Bash, Tectonic
executável, Poppler (`pdfinfo` e `pdftotext`), além dos utilitários usuais
`cp`, `mkdir`, `grep` e `awk`. A renderização opcional requer `pdftoppm`.
O Tectonic pode precisar de rede para baixar seu bundle TeX/fontes na primeira
compilação; para uso offline, prepare e valide o cache antecipadamente.
Registre a versão do Tectonic, do Poppler e do bundle/cache para comparar builds;
a receita não fixa essas dependências nem garante PDFs idênticos byte a byte.
O script não procura binários em diretórios temporários pessoais.

A partir da raiz do repositório:

```bash
scripts/build_tcc_article.sh
```

O script localiza `tectonic` no `PATH` ou aceita um binário explícito:

```bash
TECTONIC_BIN=/caminho/para/tectonic scripts/build_tcc_article.sh
```

O Tectonic resolve o ciclo LaTeX/BibTeX, mantém log e intermediários em
`tmp/pdfs/tcc-build/`, atualiza `docs/tcc_brainsniffer.pdf`, rejeita referências
indefinidas ou caixas `Overfull`, exige **exatamente 17 páginas** e valida a extração
de texto com Poppler. A cópia para `docs/tcc_brainsniffer.pdf` ocorre somente
depois desses gates; eles não substituem revisão visual ou científica.

**O PDF não é regenerado automaticamente ao editar fontes/documentação.**
Nesta rodada (21 set. 2026), o artigo fechou em **17 páginas A4**, referências
resolvidas, texto extraível e nenhum `Overfull`; os gates acima passaram. Não houve
geração de figuras, alteração de métricas, treino ou teste do sistema. Logs:
`tmp/pdfs/tcc-build/tcc_brainsniffer.log`. Há avisos não bloqueantes de Fontconfig,
`inputenc` ignorado, bytes antigos em comentários do template e `Underfull`.
As páginas de figuras têm espaço branco; não foi observado corte ou sobreposição.

No Windows (sem bash/Poppler completos), a compilação equivalente usa
`tmp/tectonic-install/tectonic.exe` com a flag `-f <caminho absoluto do .fmt do cache>`
em `%LOCALAPPDATA%\TectonicProject\Tectonic\cache\formats\`: sem ela, a busca do
formato `latex` colide com o diretório `docs/latex/` e falha com `Access is denied`.
A contagem de páginas sai do log (`Output written ... (N pages)`) e a extração de
texto do `pdftotext` do Git for Windows (`C:\Program Files\Git\mingw64\bin`).

```bash
TECTONIC_BIN=/home/fryits/.local/bin/tectonic scripts/build_tcc_article.sh
```

### Instalação local verificada

Tectonic **0.17.0**, x86_64 Linux musl, instalado sem sudo em
`/home/fryits/.local/bin/tectonic`; Poppler observado: **26.08.0**.
Release oficial: <https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0>.
Asset: `tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz`.
A URL direta de github.com falhou por conexão; o download HTTPS pela API oficial
(`<https://api.github.com/repos/tectonic-typesetting/tectonic/releases/assets/490852003>`,
header `Accept: application/octet-stream`) funcionou. Nenhuma credencial foi usada.
O SHA-256 publicado no campo `digest` da API foi comparado com `sha256sum -c`
**antes** de extrair e executar:

```text
8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7
```

Metadados da release e arquivo baixado: `tmp/pdfs/tectonic-install/`.
SHA-256 do executável instalado:
`a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da`.
O bundle/cache TeX não foi fixado nem teve versão identificada nesta rodada;
o build local não implica reprodução byte a byte em outra máquina.

Para inspeção visual de todas as páginas:

```bash
mkdir -p tmp/pdfs/tcc-render
pdftoppm -png -r 140 docs/tcc_brainsniffer.pdf tmp/pdfs/tcc-render/page
```

## Metadados institucionais

O nome do autor, curso, instituição, cidade e estado estão preenchidos no `.tex`.
O artigo não publica e-mail pessoal nem inventa orientador ou vínculo institucional.

As métricas do texto devem permanecer sincronizadas com os JSONs versionados em
`reports/` e com os metadados de checkpoint em `models/`. Os gráficos são
regenerados separadamente por `docs/generate_figures.py`.
