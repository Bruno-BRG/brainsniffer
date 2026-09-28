# ruff: noqa: E501  (os resumos, títulos e referências são prosa, não código)
"""Converte o artigo do TCC (LaTeX/SBC) para o formato da JHI em DOCX.

O texto dos parágrafos, tabelas e legendas vem direto de ``docs/artigo/tcc_brainsniffer.tex``,
para não haver redigitação. As decisões editoriais estão declaradas aqui:

- Seções: a Fundamentação do TCC vira 1.1--1.3 dentro da Introdução, porque o
  template da JHI recomenda a estrutura Introdução/Métodos/Resultados/Discussão/
  Conclusão e recusa manuscritos fora do padrão.
- Elementos gráficos: o limite da revista é 10. Saem para material complementar
  ``corpus_panels``, ``training_panels`` e ``bootstrap_offset`` (figuras) e
  ``tab:fatorial``; ``tab:figshare-baseline`` é absorvida por ``tab:benchmarks``.
- Citações: convertidas para Vancouver numerado, na ordem de aparição.
- Anonimização: a URL do repositório identifica o autor e é omitida.

Saída: ``docs/submissao-jhi/jhi_brainsniffer.docx``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "docs" / "artigo" / "tcc_brainsniffer.tex"
FIGDIR = REPO / "docs" / "artigo" / "figures"
OUT = REPO / "docs" / "submissao-jhi" / "jhi_brainsniffer.docx"
SUPL_OUT = REPO / "docs" / "submissao-jhi" / "jhi_material_complementar.docx"

# Elementos gráficos que saem do corpo e viram Material Complementar.
SUPPLEMENTARY = {
    "fig:corpus": "corpus_panels",
    "fig:training": "training_panels",
    "fig:bootstrap": "bootstrap_offset",
    "tab:fatorial": "Tabela S1",
}
# Tabela absorvida por outra: o rótulo some e as linhas entram na destino.
MERGED_TABLES = {"tab:figshare-baseline": "tab:benchmarks"}
# Linhas extras acrescentadas à tabela destino (baseline espectral no Figshare).
EXTRA_ROWS = {
    "tab:benchmarks": [
        ["Figshare", "RF espectral (Figshare)", "8,86", "0,667", "0,722"],
    ],
}

# Ajustes de texto pontuais exigidos pela conversão (não são reescrita de conteúdo).
REWRITES = [
    # A tabela do baseline foi absorvida pela de benchmarks: a frase original
    # citava as duas e passaria a citar o mesmo número duas vezes seguidas.
    (
        r"A Tabela\s+(\d+) resume o agregado, e a Tabela\s+\1 estende a comparação",
        r"A Tabela \1 resume o agregado e estende a comparação",
    ),
    # A recursão do EWMA fica legível sem o motor matemático do LaTeX.
    (
        r"st = α yt \+ \(1-α\)st-1, com α = 2/\(span\+1\)",
        "s(t) = α·y(t) + (1 − α)·s(t − 1), com α = 2/(span + 1)",
    ),
    # Ao sair o substantivo ("na Figura X" -> "Material Complementar"), a
    # preposição contraída fica pendurada.
    (r"\bna Material Complementar", "no Material Complementar"),
    (r"sem retreino, Material Complementar\)", "sem retreino, no Material Complementar)"),
    # A tabela do fatorial virou material complementar e precisa ser citada.
    (
        r"ao lado do agregado por janela\.",
        "ao lado do agregado por janela (Tabela S1 do Material Complementar).",
    ),
]

SECTION_MAP = {
    "Introdução": "INTRODUÇÃO",
    "Materiais e métodos": "MÉTODOS",
    "Resultados": "RESULTADOS",
    "Discussão e limitações": "DISCUSSÃO",
    "Conclusão": "CONCLUSÃO",
}
FLATTEN_SECTIONS = {"Fundamentação e trabalhos relacionados"}
# Subtítulos da Fundamentação, que viram 1.1--1.3 dentro da Introdução.
FLAT_SUBS = [
    "O índice BIS e seus limites",
    "Vazamento entre janelas e partição por grupo",
    "Corpora, transparência e relato de modelos",
]

ITALIC_MARK = ("\x01", "\x02")
BOLD_MARK = ("\x03", "\x04")
CITE_MARK = ("\x05", "\x06")
REF_MARK = ("\x07", "\x08")

# --------------------------------------------------------------------------- #
# Conversão de LaTeX para texto simples
# --------------------------------------------------------------------------- #

MATH_SYMBOLS = {
    r"\pm": "±",
    r"\mu": "µ",
    r"\alpha": "α",
    r"\delta": "δ",
    r"\Delta": "Δ",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\to": "→",
    r"\times": "×",
    r"\approx": "≈",
    r"\cdot": "·",
    r"\ldots": "…",
    r"\,": " ",
    r"\ ": " ",
    r"\;": " ",
    r"\!": "",
    r"\%": "%",
    r"\&": "&",
    r"\#": "#",
    r"\_": "_",
    r"\{": "{",
    r"\}": "}",
    r"\$": "$",
}

SUPERSCRIPTS = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")


def _apply_math(expr: str) -> str:
    expr = re.sub(r"\^\{([^}]*)\}", lambda m: m.group(1).translate(SUPERSCRIPTS), expr)
    expr = re.sub(r"\^(\w)", lambda m: m.group(1).translate(SUPERSCRIPTS), expr)
    for tex, plain in MATH_SYMBOLS.items():
        expr = expr.replace(tex, plain)
    expr = re.sub(r"\\(?:mathrm|text|mathit)\{([^{}]*)\}", r"\1", expr)
    expr = re.sub(r"\{([^{}]*)\}", r"\1", expr)
    expr = re.sub(r"_\{([^{}]*)\}", r"\1", expr)
    return re.sub(r"_(\w)", r"\1", expr)


EN_DASH = "\x09"


def inline(text: str) -> str:
    """Converte marcação inline de LaTeX preservando itálico e negrito."""
    # O travessão sai do caminho antes da matemática: em ``-2,37--0,66`` o
    # separador cola no sinal de menos e virava um travessão triplo.
    text = text.replace("---", EN_DASH).replace("--", EN_DASH)
    text = re.sub(r"\\label\{[^}]*\}", "", text)

    # Símbolos que o texto usa como palavra, não como matemática.
    text = text.replace("P_K", "Pk")

    text = re.sub(r"\\textit\{([^{}]*)\}", ITALIC_MARK[0] + r"\1" + ITALIC_MARK[1], text)
    text = re.sub(r"\\emph\{([^{}]*)\}", ITALIC_MARK[0] + r"\1" + ITALIC_MARK[1], text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", BOLD_MARK[0] + r"\1" + BOLD_MARK[1], text)
    text = re.sub(r"\\(?:textsc|texttt|mbox|text)\{([^{}]*)\}", r"\1", text)

    text = re.sub(r"\$([^$]*)\$", lambda m: _apply_math(m.group(1)), text)
    text = re.sub(r"\\cite\{([^}]*)\}", lambda m: CITE_MARK[0] + m.group(1) + CITE_MARK[1], text)
    text = re.sub(r"\\ref\{([^}]*)\}", lambda m: REF_MARK[0] + m.group(1) + REF_MARK[1], text)
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)

    for tex, plain in MATH_SYMBOLS.items():
        text = text.replace(tex, plain)
    text = text.replace("~", "\u00a0")
    text = text.replace("``", "“").replace("''", "”")
    text = text.replace(EN_DASH, "–")
    text = re.sub(r"µ\s+V", "µV", text)
    text = re.sub(r"[ \t]+", " ", text)
    leftovers = re.findall(r"\\[a-zA-Z]+", text)
    if leftovers:
        raise ValueError(f"macro LaTeX não tratada: {sorted(set(leftovers))}")
    return text.strip()


# --------------------------------------------------------------------------- #
# Leitura e fatiamento do .tex
# --------------------------------------------------------------------------- #


def read_body() -> str:
    src = TEX.read_text(encoding="utf-8")
    body = src.split(r"\begin{document}", 1)[1].rsplit(r"\end{document}", 1)[0]
    body = body.split(r"\maketitle", 1)[1]
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.S)
    body = re.sub(r"\\begin\{resumo\}.*?\\end\{resumo\}", "", body, flags=re.S)
    # A seção Disponibilidade fica fora do corpo (a URL identifica o autor) e,
    # com ela, o que a segue no .tex (comandos de bibliografia).
    body = re.sub(
        r"\\sub?section\*?\{Disponibilidade\}.*", "", body, flags=re.S
    )
    return body


BLOCK_RE = re.compile(
    r"\\begin\{table\}.*?\\end\{table\}"
    r"|\\begin\{figure\}.*?\\end\{figure\}"
    r"|\\section\{[^}]*\}(?:\s*\\label\{[^}]*\})*"
    r"|\\subsection\{[^}]*\}(?:\s*\\label\{[^}]*\})*",
    re.S,
)


def blocks(body: str) -> list[tuple[str, str, str]]:
    """Fatia o corpo em blocos ordenados: (tipo, conteúdo, rótulo)."""
    out: list[tuple[str, str, str]] = []
    pos = 0
    for m in BLOCK_RE.finditer(body):
        prose = body[pos : m.start()].strip()
        for part in re.split(r"\n\s*\n", prose):
            if part.strip():
                out.append(("p", part.strip(), ""))
        chunk = m.group(0)
        title = inline(re.search(r"\{(.*?)\}", chunk).group(1))
        label = "|".join(re.findall(r"\\label\{([^}]*)\}", chunk))
        if chunk.startswith(r"\begin{table}"):
            out.append(("table", chunk, ""))
        elif chunk.startswith(r"\begin{figure}"):
            out.append(("figure", chunk, ""))
        elif chunk.startswith(r"\subsection"):
            out.append(("subsection", title, label))
        else:
            out.append(("section", title, label))
        pos = m.end()
    # A prosa após o último bloco (tipicamente os parágrafos da Conclusão)
    # também é corpo: sem isto, o trecho final do artigo some do DOCX.
    for part in re.split(r"\n\s*\n", body[pos:].strip()):
        if part.strip():
            out.append(("p", part.strip(), ""))
    return out


def grab(text: str, macro: str) -> str:
    """Conteúdo do primeiro ``\\macro{...}``, respeitando chaves aninhadas."""
    m = re.search(r"\\" + macro + r"\{", text)
    if not m:
        raise ValueError(f"\\{macro} não encontrado")
    depth = 0
    for i in range(m.end() - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[m.end() : i]
    raise ValueError(f"\\{macro} sem fechamento")


def parse_table(chunk: str) -> tuple[list[list[str]], str, str]:
    label = grab(chunk, "label")
    caption = grab(chunk, "caption")
    tabular = re.search(
        r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}", chunk, re.S
    )
    rows: list[list[str]] = []
    for raw in tabular.group(1).split(r"\\"):
        raw = raw.replace(r"\hline", "").strip()
        if not raw:
            continue
        rows.append([inline(c).strip() for c in raw.split("&")])
    return rows, inline(caption), label


def parse_figure(chunk: str) -> tuple[str, str, str]:
    label = grab(chunk, "label")
    graphic = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]*)\}", chunk).group(1)
    return graphic, inline(grab(chunk, "caption")), label


# --------------------------------------------------------------------------- #
# Formatação do DOCX
# --------------------------------------------------------------------------- #

BODY_FONT = "Times New Roman"


def style_run(run, size: int, *, bold=False, italic=False):
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), BODY_FONT)


def add_rich(par, text: str, size: int = 12, *, base_italic=False, base_bold=False):
    italic, bold = base_italic, base_bold
    for piece in re.split(r"(\x01|\x02|\x03|\x04)", text):
        if piece == ITALIC_MARK[0]:
            italic = True
        elif piece == ITALIC_MARK[1]:
            italic = False
        elif piece == BOLD_MARK[0]:
            bold = True
        elif piece == BOLD_MARK[1]:
            bold = False
        elif piece:
            style_run(par.add_run(piece), size, bold=bold, italic=italic)


def para(
    doc,
    text: str,
    *,
    size: int = 12,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    before: int = 6,
    after: int = 0,
    italic: bool = False,
    bold: bool = False,
):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    p.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    pf.first_line_indent = Cm(0)
    add_rich(p, text, size, base_italic=italic, base_bold=bold)
    return p


def setup(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(3.5), Cm(2.5)
    sec.left_margin = sec.right_margin = Cm(2.0)
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(12)
    normal.paragraph_format.space_before = Pt(6)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.0


# Larguras em cm (a área de texto tem 17,0 cm) e alinhamento por coluna.
TABLE_LAYOUT = {
    "tab:modelos": {
        "widths": [4.0, 3.6, 9.4],
        "aligns": ["l", "l", "l"],
    },
    "tab:config": {
        "widths": [3.4, 13.6],
        "aligns": ["l", "l"],
    },
    "tab:benchmarks": {
        "widths": [2.2, 3.2, 3.9, 3.85, 3.85],
        "aligns": ["l", "l", "c", "c", "c"],
    },
    "tab:ewma": {
        "widths": [2.3, 3.0, 2.9, 2.9, 2.9, 3.0],
        "aligns": ["l", "l", "c", "c", "c", "c"],
    },
    "tab:fatorial": {
        "widths": [2.2, 3.1, 3.5, 2.6, 2.8, 2.8],
        "aligns": ["l", "l", "c", "c", "c", "c"],
    },
    "cadastro": {
        "widths": [3.8, 4.4, 4.4, 4.4],
        "aligns": ["l", "l", "l", "l"],
    },
}

INNER = "D9D9D9"
RULE = "404040"

ALIGN_MAP = {
    "l": WD_ALIGN_PARAGRAPH.LEFT,
    "c": WD_ALIGN_PARAGRAPH.CENTER,
    "r": WD_ALIGN_PARAGRAPH.RIGHT,
}


def _twips(cm: float) -> str:
    return str(int(round(cm * 567)))


# O Word valida a ordem dos filhos de tblPr e tcPr; anexar fora de ordem
# produz arquivo que abre com aviso de reparo.
TBLPR_ORDER = [
    "tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
    "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd",
    "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
]
TCPR_ORDER = [
    "cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
    "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign",
]


def _insert(parent, element, order: list[str]) -> None:
    tag = element.tag.split("}")[1]
    posicao = order.index(tag)
    for filho in parent:
        nome = filho.tag.split("}")[1]
        if nome in order and order.index(nome) > posicao:
            filho.addprevious(element)
            return
    parent.append(element)


def _border(element, tag: str, sz: int, color: str) -> None:
    el = OxmlElement(f"w:{tag}")
    el.set(qn("w:val"), "single" if sz else "none")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), color if sz else "auto")
    element.append(el)


def _style_table(table, widths: list[float]) -> None:
    """Régua grossa em cima e embaixo, separadores leves, sem linhas verticais."""
    table.autofit = False
    tblPr = table._tbl.tblPr

    bordas = OxmlElement("w:tblBorders")
    _border(bordas, "top", 12, RULE)
    _border(bordas, "left", 0, RULE)
    _border(bordas, "bottom", 12, RULE)
    _border(bordas, "right", 0, RULE)
    _border(bordas, "insideH", 4, INNER)
    _border(bordas, "insideV", 0, RULE)
    _insert(tblPr, bordas, TBLPR_ORDER)

    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    _insert(tblPr, layout, TBLPR_ORDER)

    margens = OxmlElement("w:tblCellMar")
    for lado, valor in (("top", 70), ("left", 90), ("bottom", 70), ("right", 90)):
        el = OxmlElement(f"w:{lado}")
        el.set(qn("w:w"), str(valor))
        el.set(qn("w:type"), "dxa")
        margens.append(el)
    _insert(tblPr, margens, TBLPR_ORDER)

    grid = table._tbl.find(qn("w:tblGrid"))
    for coluna, largura in zip(grid.findall(qn("w:gridCol")), widths):
        coluna.set(qn("w:w"), _twips(largura))


def _header_rule(row) -> None:
    """Fecha o cabeçalho com régua própria e repete a linha em quebra de página."""
    trPr = row._tr.get_or_add_trPr()
    repetir = OxmlElement("w:tblHeader")
    repetir.set(qn("w:val"), "true")
    trPr.append(repetir)
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        bordas = OxmlElement("w:tcBorders")
        _border(bordas, "bottom", 8, RULE)
        _insert(tcPr, bordas, TCPR_ORDER)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "F2F2F2")
        _insert(tcPr, shd, TCPR_ORDER)


def emit_table(doc, label: str, rows: list[list[str]], caption: str, prefix: str) -> None:
    """Tabela nativa formatada: título acima, régua de cabeçalho, fonte abaixo."""
    para(doc, f"{prefix}: {caption}", size=11, align=WD_ALIGN_PARAGRAPH.LEFT,
         before=6, after=0, bold=True)
    layout = TABLE_LAYOUT[label]
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    _style_table(table, layout["widths"])
    for r, row in enumerate(rows):
        for c in range(ncols):
            cell = table.cell(r, c)
            cell.width = Cm(layout["widths"][c])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            par = cell.paragraphs[0]
            par.alignment = ALIGN_MAP[layout["aligns"][c]]
            par.paragraph_format.space_before = Pt(0)
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.line_spacing = 1.0
            if c < len(row):
                add_rich(par, row[c], 11, base_bold=(r == 0))
    _header_rule(table.rows[0])
    para(doc, "Fonte: elaboração própria.", size=11, before=0, after=6)


def limpar_metadados(doc) -> None:
    """O .docx não deve levar no rodapé de propriedades quem o gerou."""
    cp = doc.core_properties
    for attr in ("author", "last_modified_by", "comments", "title", "category", "subject"):
        setattr(cp, attr, "")
    cp.keywords = ""


def vancouver_group(numbers: list[int]) -> str:
    """'1,2;5' / '2-5' conforme a regra da revista (hífen e ponto e vírgula)."""
    nums = sorted(set(numbers))
    runs: list[list[int]] = []
    for n in nums:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    rendered = []
    for run in runs:
        if len(run) >= 3:
            rendered.append(f"{run[0]}-{run[-1]}")
        else:
            rendered.append(",".join(str(n) for n in run))
    return ";".join(rendered)


# --------------------------------------------------------------------------- #
# Conteúdo pré-textual, escrito para a JHI
# --------------------------------------------------------------------------- #

TITLES = {
    "pt": "BrainSniffer: uma rede convolucional compacta e causal para estimar a referência bispectral a partir do eletroencefalograma frontal",
    "es": "BrainSniffer: una red convolucional compacta y causal para estimar la referencia bispectral a partir del electroencefalograma frontal",
    "en": "BrainSniffer: a compact causal convolutional network to estimate the bispectral reference from frontal electroencephalography",
}

RESUMO_PT = """Objetivo: avaliar se uma rede convolucional compacta e causal estima o rótulo de referência do Índice Bispectral (BIS) do eletroencefalograma (EEG) frontal, com custo compatível com uso ao vivo.
Métodos: janelas de cinco segundos a 128 Hz, 0,15 milhão de parâmetros e dez épocas, com partições agrupadas por caso ou participante. O desfecho primário foi o erro absoluto médio (MAE) por janela, contra um controle espectral e a variabilidade do monitor; a ordenação usou a probabilidade de predição Pk. Dois checkpoints foram avaliados em cinco casos do Figshare e 15 do VitalDB, com suavização exponencial causal (EWMA) sobre predições congeladas.
Resultados: no Figshare o ativo obteve MAE 7,03 contra 8,86 do controle espectral (Pk 0,773). No VitalDB degradou para 12,43 (Pk 0,515, indistinguível do acaso). A suavização levou o misto a 6,11 no span 5, com 10 s de atraso, frente aos 45 a 95 s de memória que um modelo contemporâneo sete vezes maior precisou.
Conclusão: no corpus de origem, o erro fica na mesma ordem de grandeza da variabilidade reportada para o monitor, o que motiva avaliação prospectiva da cadeia; o desempenho, porém, não transporta para outra fonte. Os resultados são retrospectivos e exploratórios e não demonstram equivalência clínica."""

RESUMEN_ES = """Objetivo: evaluar si una red convolucional compacta y causal estima la etiqueta de referencia del Índice Bispectral (BIS) del electroencefalograma (EEG) frontal, con costo compatible con el uso en vivo.
Métodos: ventanas de cinco segundos a 128 Hz, 0,15 millones de parámetros y diez épocas, con particiones agrupadas por caso. El desenlace primario fue el error absoluto medio (MAE) por ventana, frente a un control espectral y a la variabilidad del monitor; el ordenamiento usó la probabilidad Pk. Se evaluaron dos checkpoints en cinco casos de Figshare y 15 del VitalDB, con suavizado exponencial causal (EWMA) sobre predicciones congeladas.
Resultados: en Figshare el activo obtuvo MAE 7,03 frente a 8,86 del control espectral (Pk 0,773). En VitalDB degradó a 12,43 (Pk 0,515, indistinguible del azar). El suavizado llevó al mixto a 6,11 en el span 5, con 10 s de retardo, frente a los 45 a 95 s que necesitó un modelo contemporáneo siete veces mayor.
Conclusión: en el corpus de origen, el error queda en el mismo orden de magnitud que la variabilidad reportada para el monitor, lo que motiva la evaluación prospectiva de la cadena; el desempeño, sin embargo, no se transporta a otra fuente. Los resultados son retrospectivos y exploratorios y no demuestran equivalencia clínica."""

ABSTRACT_EN = """Objective: to assess whether a compact, causal convolutional network estimates the Bispectral Index (BIS) reference label from frontal electroencephalography (EEG) at an execution cost compatible with live use.
Methods: five-second windows at 128 Hz, 0.15 million parameters and ten training epochs, with partitions grouped by case or participant. The primary outcome was window-level mean absolute error (MAE), against a spectral control and the monitor's own variability; ordering used the prediction probability Pk. Two checkpoints were evaluated on five Figshare cases and 15 VitalDB cases, with causal exponential smoothing (EWMA) over frozen predictions.
Results: on Figshare the active checkpoint reached MAE 7.03 against 8.86 for the spectral control, with Pk 0.773. On VitalDB it degraded to 12.43 (Pk 0.515, indistinguishable from chance). Smoothing took the mixed candidate to 6.11 at span 5, with 10 s of lag, a range in which a seven-times-larger contemporary model needed 45 to 95 s of memory.
Conclusion: in the source corpus, the error is of the same order of magnitude as the monitor's reported variability, which motivates prospective evaluation of the chain; the performance, however, does not transport to another source. Results are retrospective and exploratory and do not demonstrate clinical equivalence."""

KEYWORDS = {
    "pt": "Eletroencefalografia; Anestesia Geral; Redes Neurais de Computação",
    "es": "Electroencefalografía; Anestesia General; Redes Neurales de la Computación",
    "en": "Electroencephalography; Anesthesia, General; Neural Networks, Computer",
}

CONTRIBUICAO = (
    "Autor 1 participou da concepção do estudo, da implementação da cadeia de "
    "processamento, da condução dos experimentos, da análise dos dados e da redação "
    "da primeira versão do manuscrito. Autor 2 participou da orientação do trabalho, "
    "da concepção metodológica, da interpretação dos resultados e da revisão crítica "
    "do conteúdo. Autor 3 participou da coorientação, da discussão dos resultados e "
    "da revisão final do manuscrito. Todos os autores participaram da discussão dos "
    "resultados, revisaram criticamente o manuscrito e aprovaram a versão final."
)

CONFLITOS = (
    "Os autores declaram não haver conflito de interesses de natureza pessoal, "
    "profissional, financeira ou institucional relacionado a este manuscrito."
)

USO_IA = (
    "Os autores utilizaram a ferramenta Claude (Anthropic) como apoio à redação, à "
    "organização textual, à revisão linguística e à implementação dos scripts de "
    "análise e de geração de figuras. Todo o conteúdo, a interpretação dos resultados "
    "e as conclusões foram revisados e validados pelos autores, que assumem integral "
    "responsabilidade pelo trabalho apresentado. A ferramenta não foi utilizada como "
    "método científico autônomo nem substitui a autoria humana."
)

ETICA = (
    "Este estudo utiliza exclusivamente dados públicos já disponibilizados de forma "
    "aberta: o conjunto “EEG and BIS raw data”, publicado no Figshare sob licença "
    "CC BY 4.0, e o VitalDB, sob licença CC BY-NC-SA 4.0 e acordo de uso. Não houve "
    "coleta de dados primários, contato com participantes ou uso de dados "
    "identificáveis; por essa razão, e conforme as diretrizes da revista, não houve "
    "submissão a Comitê de Ética em Pesquisa."
)

LINKS = (
    "Código, relatórios de auditoria com hashes e o material complementar deste "
    "trabalho estão disponíveis em repositório público. A URL é omitida nesta versão "
    "para preservar a avaliação duplo-cega e será informada no sistema de submissão."
)

# Referências em estilo Vancouver (ICMJE/NLM), na ordem em que aparecem no texto.
VANCOUVER = {
    "johansen2000": "Johansen JW, Sebel PS. Development and clinical application of electroencephalographic bispectrum monitoring. Anesthesiology. 2000;93(5):1336-44. doi:10.1097/00000542-200011000-00029",
    "schuller2015": "Schuller PJ, Newell S, Strickland PA, Barry JJ. Response of bispectral index to neuromuscular block in awake volunteers. Br J Anaesth. 2015;115(Suppl 1):i95-i103. doi:10.1093/bja/aev072",
    "hans2005": "Hans P, Dewandre PY, Brichant JF, Bonhomme V. Comparative effects of ketamine on Bispectral Index and spectral entropy of the electroencephalogram under sevoflurane anaesthesia. Br J Anaesth. 2005;94(3):336-40. doi:10.1093/bja/aei047",
    "pilge2006": "Pilge S, Zanner R, Schneider G, Blum J, Kreuzer M, Kochs EF. Time delay of index calculation: analysis of cerebral state, bispectral, and Narcotrend indices. Anesthesiology. 2006;104(3):488-94. doi:10.1097/00000542-200603000-00016",
    "baware": "Myles PS, Leslie K, McNeil J, Forbes A, Chan MTV. Bispectral index monitoring to prevent awareness during anaesthesia: the B-Aware randomised controlled trial. Lancet. 2004;363(9423):1757-63. doi:10.1016/S0140-6736(04)16300-9",
    "bunaware": "Avidan MS, Zhang L, Burnside BA, et al. Anesthesia awareness and the bispectral index. N Engl J Med. 2008;358(11):1097-108. doi:10.1056/NEJMoa0707361",
    "bagrecall": "Avidan MS, Jacobsohn E, Glick D, et al. Prevention of intraoperative awareness in a high-risk surgical population. N Engl J Med. 2011;365(7):591-600. doi:10.1056/NEJMoa1100403",
    "sukriti2026": "Sukriti, Kriplani C, Kumar S, Singh A. Predicting depth of anaesthesia from single-channel EEG using a deep TCN-BiLSTM-attention model with EWMA. Sci Rep. 2026;16:25470. doi:10.1038/s41598-026-54608-8",
    "park2020": "Park Y, Han SH, Byun W, Kim JH, Lee HC, Kim SJ. A real-time depth of anesthesia monitoring system based on deep neural network with large EDO tolerant EEG analog front-end. IEEE Trans Biomed Circuits Syst. 2020;14(4):825-37. doi:10.1109/TBCAS.2020.2998172",
    "shi2023": "Shi M, Huang Z, Xiao G, Xu B, Ren Q, Zhao H. Estimating the depth of anesthesia from EEG signals based on a deep residual shrinkage network. Sensors (Basel). 2023;23(2):1008. doi:10.3390/s23021008",
    "saeb2017": "Saeb S, Lonini L, Jayaraman A, Mohr DC, Kording KP. The need to approximate the use-case in clinical machine learning. Gigascience. 2017;6(5):gix019. doi:10.1093/gigascience/gix019",
    "ma2017": "Ma L. EEG and BIS raw data [dataset]. Figshare; 2017. doi:10.6084/m9.figshare.5589841.v1",
    "nsugbe2022": "Nsugbe E, Connelly S. Multiscale depth of anaesthesia prediction for surgery using frontal cortex electroencephalography. Healthc Technol Lett. 2022;9(3):43-53. doi:10.1049/htl2.12025",
    "lee2022": "Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Sci Data. 2022;9:279. doi:10.1038/s41597-022-01411-5",
    "vitaldb": "VitalDB. Open dataset: overview and track list [Internet]. 2026 [citado 27 set 2026]. Disponível em: https://vitaldb.net/dataset/",
    "dosei": "Garbe J, Nguyen QV, Kantelhardt JW, et al. DOSE-I: a multimodal biosignal dataset of procedural sedation for endoscopy [dataset]. Zenodo; 2026. doi:10.5281/zenodo.18483292",
    "openbis": "Connor CW. Open reimplementation of the BIS algorithms for depth of anesthesia. Anesth Analg. 2022;135(4):855-64. doi:10.1213/ANE.0000000000006119",
    "tripodai": "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378",
    "archer2021": "Archer L, Snell KIE, Ensor J, Hudda MT, Collins GS, Riley RD. Minimum sample size for external validation of a clinical prediction model with a continuous outcome. Stat Med. 2021;40(1):133-46. doi:10.1002/sim.8766",
    "harada2021": "Harada H, Muta S, Kakuma T, et al. Accuracy of BIS monitoring using a novel interface device connecting conventional needle-electrodes and BIS sensors during frontal neurosurgical procedures. PLoS One. 2021;16(10):e0258647. doi:10.1371/journal.pone.0258647",
    "dahaba2010": "Dahaba AA, Xue JX, Zhao GG, et al. BIS-vista occipital montage in patients undergoing neurosurgical procedures during propofol-remifentanil anesthesia. Anesthesiology. 2010;112(3):645-51. doi:10.1097/ALN.0b013e3181cf4111",
    "shiraishi2004": "Shiraishi T, Uchino H, Sagara T, Ishii N. A comparison of frontal and occipital bispectral index values obtained during neurosurgical procedures. Anesth Analg. 2004;98(6):1773-5. doi:10.1213/01.ANE.0000121344.69058.09",
    "imdrfn88": "International Medical Device Regulators Forum. Good machine learning practice for medical device development: guiding principles. IMDRF/AIML WG/N88 FINAL:2025; 2025. Disponível em: https://www.imdrf.org",
    "smith1996": "Smith WD, Dutton RC, Smith NT. Measuring the performance of anesthetic depth indicators. Anesthesiology. 1996;84(1):38-51. doi:10.1097/00000542-199601000-00005",
    "field2007": "Field CA, Welsh AH. Bootstrapping clustered data. J R Stat Soc Series B Stat Methodol. 2007;69(3):369-90. doi:10.1111/j.1467-9868.2007.00593.x",
    "probastai": "Moons KGM, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ. 2025;388:e082505. doi:10.1136/bmj-2024-082505",
    "decideai": "Vasey B, Nagendran M, Campbell B, Clifton DA, et al. Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. Nat Med. 2022;28(5):924-33. doi:10.1038/s41591-022-01772-9",
    "liu2018": "Liu Q, Ma L, Fan SZ, Abbod MF, Shieh JS. Sample entropy analysis for the estimating depth of anaesthesia through human EEG signal at different levels of unconsciousness during surgeries. PeerJ. 2018;6:e4817. doi:10.7717/peerj.4817",
}


def build() -> None:
    doc = Document()
    setup(doc)
    items = blocks(read_body())

    # ---- Passada 1: numeração de figuras, tabelas, seções e citações -------- #
    numbers = {"fig": {}, "tab": {}, "sec": {}}
    headings: dict[str, str] = {}
    subheadings: dict[str, str] = {}
    sec_n = {"n": 0, "sub": 0}
    order: list[str] = []

    for kind, payload, label in items:
        if kind == "section":
            if payload in FLATTEN_SECTIONS:
                continue
            sec_n["n"] += 1
            sec_n["sub"] = 0
            headings[payload] = f"{sec_n['n']}. {SECTION_MAP[payload]}"
            for lab in filter(None, label.split("|")):
                numbers["sec"][lab] = str(sec_n["n"])
        elif kind == "subsection":
            if payload in FLAT_SUBS:
                idx = FLAT_SUBS.index(payload) + 1
                subheadings[payload] = f"1.{idx} {payload}"
                for lab in filter(None, label.split("|")):
                    numbers["sec"][lab] = f"1.{idx}"
            else:
                sec_n["sub"] += 1
                subheadings[payload] = f"{sec_n['n']}.{sec_n['sub']} {payload}"
                for lab in filter(None, label.split("|")):
                    numbers["sec"][lab] = f"{sec_n['n']}.{sec_n['sub']}"
        elif kind == "table":
            _, _, lab = parse_table(payload)
            if lab not in SUPPLEMENTARY and lab not in MERGED_TABLES:
                numbers["tab"][lab] = len(numbers["tab"]) + 1
        elif kind == "figure":
            _, _, lab = parse_figure(payload)
            if lab not in SUPPLEMENTARY:
                numbers["fig"][lab] = len(numbers["fig"]) + 1
        else:
            for key in re.findall(CITE_MARK[0] + r"([^" + CITE_MARK[1] + r"]*)" + CITE_MARK[1], inline(payload)):
                for k in key.split(","):
                    k = k.strip()
                    if k not in order:
                        order.append(k)

    cite_num = {k: i + 1 for i, k in enumerate(order)}
    missing = [k for k in order if k not in VANCOUVER]
    if missing:
        raise ValueError(f"sem entrada Vancouver: {missing}")

    def resolve(text: str) -> str:
        def cite_sub(m):
            nums = [cite_num[k.strip()] for k in m.group(1).split(",") if k.strip()]
            return f"({vancouver_group(nums)})"

        def ref_sub(m):
            lab = m.group(1)
            if lab in numbers["fig"]:
                return str(numbers["fig"][lab])
            if lab in numbers["tab"]:
                return str(numbers["tab"][lab])
            if lab in numbers["sec"]:
                return numbers["sec"][lab]
            if lab in MERGED_TABLES:
                return str(numbers["tab"][MERGED_TABLES[lab]])
            if lab in SUPPLEMENTARY:
                return "Material Complementar"
            raise ValueError(f"referência não resolvida: {lab}")

        text = re.sub(CITE_MARK[0] + r"([^" + CITE_MARK[1] + r"]*)" + CITE_MARK[1], cite_sub, text)
        text = re.sub(REF_MARK[0] + r"([^" + REF_MARK[1] + r"]*)" + REF_MARK[1], ref_sub, text)
        # "Figura~\ref{alguma_do_suplementar}" precisa perder o substantivo, senão
        # sai "Figura Material Complementar"; marcadores de painel (a,b) idem.
        text = re.sub(
            r"(?:Figura|Tabela|Seção|Figuras|Tabelas|Seções)\s*Material Complementar"
            r"[a-z]?(?:,[a-z])*",
            "Material Complementar",
            text,
        )
        for pattern, repl in REWRITES:
            text = re.sub(pattern, repl, text)
        return text

    # ---- Pré-textual ------------------------------------------------------- #
    para(doc, TITLES["pt"], size=14, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6, bold=True)
    # A revista exige o título também em espanhol e inglês.
    para(doc, TITLES["es"], size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
         before=0, after=0, italic=True)
    para(doc, TITLES["en"], size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
         before=0, after=6, italic=True)
    para(doc, "Autor 1¹, Autor 2², Autor 3³", size=11,
         align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=0)
    for mark in "¹²³":
        para(doc, f"{mark}Instituição, Cidade (UF), Brasil. ORCID: 0000-0000-0000-0000",
             size=11, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=0, italic=True)
    para(doc, "[e-mails e identificação omitidos na versão anônima]", size=11,
         align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6)

    for heading, body_text, key in (
        ("Resumo", RESUMO_PT, "Palavras-chave"),
        ("Resumen", RESUMEN_ES, "Palabras clave"),
        ("Abstract", ABSTRACT_EN, "Keywords"),
    ):
        para(doc, heading, size=12, align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=0, bold=True)
        for part in body_text.split("\n"):
            para(doc, part, size=11, italic=True)
        lang = {"Resumo": "pt", "Resumen": "es", "Abstract": "en"}[heading]
        para(doc, f"{key}: {KEYWORDS[lang]}", size=11, italic=True)

    # ---- Corpo ------------------------------------------------------------- #
    for kind, payload, _label in items:
        if kind == "section":
            if payload in FLATTEN_SECTIONS:
                continue
            para(doc, headings[payload], size=12,
                 align=WD_ALIGN_PARAGRAPH.LEFT, before=12, after=0, bold=True)
            continue

        if kind == "subsection":
            para(doc, subheadings[payload], size=12,
                 align=WD_ALIGN_PARAGRAPH.LEFT, before=12, after=0, bold=True)
            continue

        if kind == "table":
            rows, caption, label = parse_table(payload)
            if label in SUPPLEMENTARY or label in MERGED_TABLES:
                continue
            num = numbers["tab"][label]
            rows = rows + EXTRA_ROWS.get(label, [])
            if label == "tab:benchmarks":
                caption += (" Para o RF espectral no Figshare não há intervalo por "
                            "bootstrap; os valores são pontuais.")
            emit_table(
                doc,
                label,
                [[resolve(c) for c in row] for row in rows],
                resolve(caption),
                f"Tabela {num}",
            )
            continue

        if kind == "figure":
            graphic, caption, label = parse_figure(payload)
            if label in SUPPLEMENTARY:
                continue
            para(doc, f"Figura {numbers['fig'][label]}: {resolve(caption)}", size=11,
                 align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=0, bold=True)
            doc.add_picture(str(FIGDIR / graphic.replace(".pdf", ".png")), width=Cm(13.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            para(doc, "Fonte: elaboração própria.", size=11, before=0, after=6)
            continue

        text = resolve(inline(payload))
        if text:
            para(doc, text)

    # ---- Fecho ------------------------------------------------------------- #
    para(doc, "Contribuição dos autores", size=12, align=WD_ALIGN_PARAGRAPH.LEFT,
         before=12, after=0, bold=True)
    para(doc, CONTRIBUICAO)
    for heading, body_text in (
        ("Conflitos de interesse", CONFLITOS),
        ("Declaração do uso de Inteligência Artificial", USO_IA),
        ("Aspectos éticos", ETICA),
        ("Links", LINKS),
    ):
        para(doc, heading, size=12, align=WD_ALIGN_PARAGRAPH.LEFT, before=12, after=0, bold=True)
        para(doc, body_text)

    para(doc, "REFERÊNCIAS", size=12, align=WD_ALIGN_PARAGRAPH.LEFT, before=12, after=0, bold=True)
    for i, key in enumerate(order, start=1):
        para(doc, f"{i}.{VANCOUVER[key]}", align=WD_ALIGN_PARAGRAPH.LEFT)

    # Seções numeradas do corpo, corrigindo os títulos emitidos acima.
    limpar_metadados(doc)
    doc.save(OUT)
    print(f"figuras: {len(numbers['fig'])}  tabelas: {len(numbers['tab'])}")
    print(f"elementos gráficos: {len(numbers['fig']) + len(numbers['tab'])}")
    print(f"referências: {len(order)}")
    print(f"gravado: {OUT}")


SUPL_ORDER = [
    ("fig", "fig:corpus", "Figura S1"),
    ("fig", "fig:training", "Figura S2"),
    ("fig", "fig:bootstrap", "Figura S3"),
    ("tab", "tab:fatorial", "Tabela S1"),
]


def build_supplementary() -> None:
    """Arquivo separado com os elementos que excedem o limite de dez da revista."""
    doc = Document()
    setup(doc)
    para(doc, "Material Complementar", size=14, align=WD_ALIGN_PARAGRAPH.LEFT,
         before=0, after=6, bold=True)
    para(doc, TITLES["pt"], size=12, italic=True)
    para(doc, "Este arquivo reúne os elementos gráficos que excedem o limite de dez "
              "elementos do corpo do artigo. São citados no texto como Material "
              "Complementar.")

    found: dict[str, object] = {}
    for kind, chunk, _lab in blocks(read_body()):
        if kind == "figure":
            graphic, caption, label = parse_figure(chunk)
            found[label] = (graphic, caption)
        elif kind == "table":
            rows, caption, label = parse_table(chunk)
            found[label] = (rows, caption)

    for kind, label, prefix in SUPL_ORDER:
        if label not in found:
            raise ValueError(f"elemento ausente no .tex: {label}")
        payload = found[label]
        if kind == "fig":
            graphic, caption = payload
            para(doc, f"{prefix}: {inline(caption)}", size=11,
                 align=WD_ALIGN_PARAGRAPH.LEFT, before=12, after=0, bold=True)
            doc.add_picture(str(FIGDIR / graphic.replace(".pdf", ".png")), width=Cm(13.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            para(doc, "Fonte: elaboração própria.", size=11, before=0, after=6)
        else:
            rows, caption = payload
            emit_table(doc, label, rows, inline(caption), prefix)

    limpar_metadados(doc)
    doc.save(SUPL_OUT)
    print(f"gravado: {SUPL_OUT}")


CADASTRO_OUT = REPO / "docs" / "submissao-jhi" / "jhi_cadastro_autores.docx"

# Ordem definida pelo autor: quem executou, depois orientador e coorientador.
# A revista exige que a ordem na plataforma seja idêntica à do manuscrito.
CADASTRO = [
    ["Campo no OJS", "Autor 1", "Autor 2", "Autor 3"],
    ["Nome", "Bruno Rocha", "Igor", "José Garcia"],
    ["Sobrenome", "Guimarães", "Gonzalez", "Vivas Miranda"],
    ["E-mail", "", "", ""],
    ["ORCID iD", "", "", ""],
    ["Instituição", "Centro Universitário Jorge Amado", "", ""],
    ["Cidade / Estado / País", "Salvador (BA), Brasil", "", ""],
    ["Papel", "Execução do estudo", "Orientação", "Coorientação"],
    ["Contato principal", "SIM", "não", "não"],
]

NOTAS_CADASTRO = [
    "Todos os autores precisam de ORCID iD — a revista exige o identificador para cada um, "
    "e sem ele a submissão não avança. O registro é gratuito em orcid.org.",
    "Confira o par Nome/Sobrenome contra o Lattes de cada um. Errar a separação faz o "
    "nome aparecer trocado nas bases de indexação depois de publicado.",
    "Um e-mail é obrigatório por autor. O do contato principal é o que a revista usa "
    "para toda a correspondência, inclusive as decisões editoriais.",
    "A ordem cadastrada na plataforma tem de ser exatamente esta: a revista verifica se "
    "bate com a do manuscrito.",
    "Nada disso vai no arquivo .docx — ele segue anonimizado. Estes dados existem só na "
    "plataforma, e é por isso que o item da anonimização e o da identificação do "
    "correspondente convivem sem contradição.",
]


def build_cadastro() -> None:
    """Ficha para preencher o cadastro de autores no OJS, fora do arquivo anônimo."""
    doc = Document()
    setup(doc)
    para(doc, "Cadastro de autores na plataforma — JHI", size=14,
         align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=6, bold=True)
    para(doc, TITLES["pt"], size=12, italic=True)
    emit_table(doc, "cadastro", CADASTRO,
               "dados exigidos por autor no formulário de submissão. "
               "Preencher as células vazias antes de cadastrar.", "Quadro")
    for nota in NOTAS_CADASTRO:
        para(doc, nota, size=11)
    limpar_metadados(doc)
    doc.save(CADASTRO_OUT)
    print(f"gravado: {CADASTRO_OUT}")


if __name__ == "__main__":
    build()
    build_supplementary()
    build_cadastro()
    sys.exit(0)
