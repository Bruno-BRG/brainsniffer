#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
docs_dir="${repo_dir}/docs"
build_dir="${repo_dir}/tmp/pdfs/tcc-build"
source_name="tcc_brainsniffer.tex"
pdf_name="tcc_brainsniffer.pdf"

tectonic_bin="${TECTONIC_BIN:-}"
if [[ -z "${tectonic_bin}" ]]; then
  if command -v tectonic >/dev/null 2>&1; then
    tectonic_bin="$(command -v tectonic)"
  else
    echo "erro: tectonic não encontrado; instale Tectonic no PATH ou defina TECTONIC_BIN=/caminho/para/tectonic" >&2
    exit 1
  fi
fi
if ! command -v "${tectonic_bin}" >/dev/null 2>&1; then
  echo "erro: TECTONIC_BIN não é executável; informe um binário Tectonic válido ou use tectonic no PATH" >&2
  exit 1
fi

for dependency in pdfinfo pdftotext; do
  if ! command -v "${dependency}" >/dev/null 2>&1; then
    echo "erro: ${dependency} não encontrado (Poppler é obrigatório)" >&2
    exit 1
  fi
done

mkdir -p "${build_dir}"
cp "${docs_dir}/referencias.bib" "${build_dir}/referencias.bib"
cp "${docs_dir}/latex/sbc.bst" "${build_dir}/sbc.bst"
(
  cd "${docs_dir}"
  "${tectonic_bin}" \
    -Z search-path="${docs_dir}" \
    -Z search-path="${docs_dir}/latex" \
    --chatter minimal \
    --keep-logs \
    --keep-intermediates \
    --outdir "${build_dir}" \
    "${source_name}"
  # Uma segunda execução deixa o log final livre dos avisos normais da
  # primeira passagem, antes de BibTeX resolver as citações.
  "${tectonic_bin}" \
    -Z search-path="${docs_dir}" \
    -Z search-path="${docs_dir}/latex" \
    --chatter minimal \
    --keep-logs \
    --keep-intermediates \
    --outdir "${build_dir}" \
    "${source_name}"
)

log_file="${build_dir}/tcc_brainsniffer.log"
built_pdf="${build_dir}/${pdf_name}"
bibliography_file="${build_dir}/tcc_brainsniffer.bbl"
bibliography_log="${build_dir}/tcc_brainsniffer.blg"
if [[ ! -s "${built_pdf}" || ! -s "${log_file}" || ! -s "${bibliography_file}" ]]; then
  echo "erro: Tectonic não produziu PDF, log e bibliografia válidos" >&2
  exit 1
fi
if [[ -s "${bibliography_log}" ]] && grep -Eqi 'error message|I couldn.t open|I found no' "${bibliography_log}"; then
  echo "erro: BibTeX reportou falha" >&2
  grep -Ein 'error message|I couldn.t open|I found no' "${bibliography_log}" >&2
  exit 1
fi

if grep -Eqi 'Overfull \\[hv]box|undefined control sequence|citation.+undefined|there were undefined references' "${log_file}"; then
  echo "erro: o log contém overflow ou referência indefinida" >&2
  grep -Ein 'Overfull \\[hv]box|undefined control sequence|citation.+undefined|there were undefined references' "${log_file}" >&2
  exit 1
fi

# Limite a pedido do autor: o artigo precisa fechar em exatamente 17 páginas.
# O crescimento fora disso é sempre acidente de layout e volta a ser pego aqui.
page_count="$(pdfinfo "${built_pdf}" | awk '/^Pages:/ {print $2}')"
if [[ ! "${page_count}" =~ ^[0-9]+$ || "${page_count}" -ne 17 ]]; then
  echo "erro: PDF tem ${page_count:-contagem-indisponível} páginas; o alvo é exatamente 17" >&2
  exit 1
fi

pdftotext "${built_pdf}" "${build_dir}/tcc_brainsniffer.txt"
if [[ ! -s "${build_dir}/tcc_brainsniffer.txt" ]]; then
  echo "erro: extração de texto vazia" >&2
  exit 1
fi
if grep -Fq '??' "${build_dir}/tcc_brainsniffer.txt"; then
  echo "erro: referência não resolvida no texto extraído" >&2
  exit 1
fi

cp "${built_pdf}" "${docs_dir}/${pdf_name}"
echo "artigo validado: ${docs_dir}/${pdf_name} (${page_count} páginas)"
