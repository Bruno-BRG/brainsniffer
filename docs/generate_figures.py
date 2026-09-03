"""Generate publication-ready BrainSniffer figures from recorded reports."""

# Figure labels are intentionally kept close to the visual layout.
# ruff: noqa: E501

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = Path(__file__).resolve().parent / "figures"

NAVY = "#102A43"
BLUE = "#247BA0"
TEAL = "#20A39E"
ORANGE = "#F18F01"
RED = "#D1495B"
INK = "#243B53"
MUTED = "#627D98"
GRID = "#D9E2EC"
PALE = "#F0F4F8"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def canvas(width: int = 1800, height: int = 980) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    return image, ImageDraw.Draw(image)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, color=INK, bold=False):
    draw.text(xy, value, fill=color, font=font(size, bold=bold))


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=None, radius=24, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start, end, color=BLUE, width=8):
    draw.line((*start, *end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 22
    points = [
        end,
        (
            end[0] - size * math.cos(angle - math.pi / 6),
            end[1] - size * math.sin(angle - math.pi / 6),
        ),
        (
            end[0] - size * math.cos(angle + math.pi / 6),
            end[1] - size * math.sin(angle + math.pi / 6),
        ),
    ]
    draw.polygon(points, fill=color)


def figure_pipeline() -> None:
    image, draw = canvas()
    text(draw, (90, 60), "BrainSniffer: pipeline de pesquisa", 42, NAVY, True)
    text(draw, (90, 115), "Separacao explicita entre sinal, modelo, auditoria e interpretacao", 24, MUTED)
    boxes = [
        ("1", "Dados publicos", "Figshare / VitalDB\nmanifesto e checksum", BLUE),
        ("2", "Pre-processamento", "128 Hz - 5 s\ncausal - SQI", TEAL),
        ("3", "CNN 1-D", "regressão 0–100\nSmooth L1", ORANGE),
        ("4", "Auditoria", "timestamps - lacunas\nABSTAIN", RED),
        ("5", "Saida", "BIS estimado\nestagio de pesquisa", NAVY),
    ]
    y = 300
    w, h = 285, 240
    gap = 55
    x0 = 75
    for i, (number, title, body, color) in enumerate(boxes):
        x = x0 + i * (w + gap)
        rounded(draw, (x, y, x + w, y + h), PALE, color, radius=28, width=5)
        draw.ellipse((x + 22, y + 22, x + 78, y + 78), fill=color)
        text(draw, (x + 41, y + 29), number, 28, WHITE, True)
        text(draw, (x + 24, y + 105), title, 28, color, True)
        for line_i, line in enumerate(body.split("\n")):
            text(draw, (x + 24, y + 153 + line_i * 34), line, 22, INK)
        if i < len(boxes) - 1:
            arrow(draw, (x + w + 8, y + h // 2), (x + w + gap - 12, y + h // 2), color=BLUE, width=7)
    rounded(draw, (200, 690, 1600, 850), "#FFF7E6", ORANGE, radius=22, width=3)
    text(draw, (240, 730), "Regra de seguranca do prototipo", 25, ORANGE, True)
    text(draw, (240, 775), "Sinal invalido ou silencio > 2 s  ->  estado stale/ABSTAIN  ->  filtro causal reiniciado", 25, INK)
    image.save(OUTPUT / "pipeline.png", optimize=True)


def axis(draw, left, top, right, bottom, y_ticks, y_max):
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    for value in y_ticks:
        y = bottom - (value / y_max) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        text(draw, (left - 58, y - 14), f"{value:g}", 20, MUTED)


def bar_chart(draw, area, title, labels, values, colors, y_max, value_format="{:.2f}"):
    left, top, right, bottom = area
    text(draw, (left, top - 58), title, 27, NAVY, True)
    axis(draw, left + 70, top, right, bottom, [0, y_max / 2, y_max], y_max)
    chart_left = left + 115
    slot = (right - chart_left) / len(labels)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = chart_left + i * slot + slot * 0.18
        width = slot * 0.64
        y = bottom - (value / y_max) * (bottom - top)
        draw.rounded_rectangle((x, y, x + width, bottom), radius=10, fill=color)
        text(draw, (x + width / 2 - 28, y - 38), value_format.format(value), 20, color, True)
        text(draw, (x + width / 2 - 52, bottom + 18), label, 20, INK)


def figure_comparison() -> None:
    internal = load("figshare_holdout_evaluation.json")["recomputed_test_metrics"]
    external = load("vitaldb_external_validation.json")["metrics"]
    image, draw = canvas()
    text(draw, (90, 60), "Desempenho: holdout interno versus mudanca de dominio", 39, NAVY, True)
    text(draw, (90, 115), "O checkpoint e mantido fixo na avaliacao externa; valores menores sao melhores em erro.", 23, MUTED)
    bar_chart(
        draw,
        (80, 245, 850, 760),
        "Erro absoluto médio (MAE)",
        ["Figshare", "VitalDB"],
        [internal["mae"], external["mae"]],
        [BLUE, RED],
        16,
    )
    bar_chart(
        draw,
        (940, 245, 1710, 760),
        "Correlação de Pearson", 
        ["Figshare", "VitalDB"],
        [internal["pearson_r"], external["pearson_r"]],
        [TEAL, ORANGE],
        1,
        value_format="{:.3f}",
    )
    rounded(draw, (270, 820, 1530, 905), "#F8FAFC", GRID, radius=18, width=2)
    text(draw, (310, 847), "Leitura: o resultado interno nao deve ser apresentado como generalizacao automatica.", 23, INK)
    image.save(OUTPUT / "comparison.png", optimize=True)


def figure_offset() -> None:
    report = load("offset_sensitivity.json")
    points = report["results"]
    offsets = [p["offset_seconds"] for p in points]
    maes = [p["metrics"]["mae"] for p in points]
    pears = [p["metrics"]["pearson_r"] for p in points]
    image, draw = canvas()
    text(draw, (90, 60), "Sensibilidade ao alinhamento do rotulo BIS", 40, NAVY, True)
    text(draw, (90, 115), "O checkpoint e a particao permanecem congelados; apenas o offset do rotulo varia.", 23, MUTED)

    def line_panel(top, bottom, values, ymin, ymax, title, color):
        left, right = 150, 1650
        text(draw, (150, top - 52), title, 27, NAVY, True)
        draw.line((left, top, left, bottom), fill=INK, width=3)
        draw.line((left, bottom, right, bottom), fill=INK, width=3)
        for tick in [ymin, (ymin + ymax) / 2, ymax]:
            y = bottom - ((tick - ymin) / (ymax - ymin)) * (bottom - top)
            draw.line((left, y, right, y), fill=GRID, width=2)
            text(draw, (90, y - 13), f"{tick:.2f}" if ymax < 2 else f"{tick:g}", 20, MUTED)
        for value in [-20, -10, 0, 10, 20]:
            x = left + ((value + 20) / 40) * (right - left)
            draw.line((x, top, x, bottom), fill=GRID, width=2)
            text(draw, (x - 24, bottom + 15), f"{value:+g}", 20, INK)
        points_xy = [
            (
                left + ((xv + 20) / 40) * (right - left),
                bottom - ((yv - ymin) / (ymax - ymin)) * (bottom - top),
            )
            for xv, yv in zip(offsets, values)
        ]
        draw.line(points_xy, fill=color, width=7)
        for point in points_xy:
            draw.ellipse((point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9), fill=color, outline=WHITE, width=3)

    line_panel(250, 455, maes, 6, 12, "MAE (menor e melhor)", BLUE)
    line_panel(570, 775, pears, 0.7, 0.82, "Pearson (maior e melhor)", TEAL)
    rounded(draw, (340, 790, 1460, 900), "#FFF7E6", ORANGE, radius=18, width=2)
    text(draw, (380, 815), "Interpretacao: tendencia pos-hoc; nao escolher o offset em pacientes.", 24, INK)
    image.save(OUTPUT / "offset_sensitivity.png", optimize=True)


def interval_bar(draw, y, label, mean, lower, upper, min_value, max_value, color):
    left, right = 470, 1600
    span = max_value - min_value
    def project(value):
        return left + ((value - min_value) / span) * (right - left)
    x = project(mean)
    lo = project(max(min_value, lower))
    hi = project(min(max_value, upper))
    text(draw, (100, y - 15), label, 25, INK, True)
    draw.line((lo, y, hi, y), fill=color, width=9)
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=color, outline=WHITE, width=3)
    text(draw, (1640, y - 15), f"{mean:.3f}", 23, color, True)


def figure_bootstrap() -> None:
    internal = load("figshare_holdout_evaluation.json")["case_bootstrap"]
    external = load("vitaldb_external_validation.json")["case_bootstrap"]
    image, draw = canvas()
    text(draw, (90, 60), "Incerteza exploratoria por caso", 40, NAVY, True)
    text(draw, (90, 115), "Ponto = media reamostrada; linha = intervalo de 95% das cirurgias inteiras.", 23, MUTED)
    text(draw, (100, 215), "Pearson", 29, NAVY, True)
    for y, label, source, color in [
        (300, "Figshare · 5 casos", internal["pearson_r"], BLUE),
        (430, "VitalDB · 15 casos", external["pearson_r"], RED),
    ]:
        interval_bar(draw, y, label, source["mean"], source["lower_95"], source["upper_95"], -0.2, 1.0, color)
    text(draw, (100, 610), "MAE (escala 0-16)", 29, NAVY, True)
    for y, label, source, color in [
        (695, "Figshare · 5 casos", internal["mae"], BLUE),
        (825, "VitalDB · 15 casos", external["mae"], RED),
    ]:
        interval_bar(draw, y, label, source["mean"], source["lower_95"], source["upper_95"], 0.0, 16.0, color)
    image.save(OUTPUT / "bootstrap_intervals.png", optimize=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure_pipeline()
    figure_comparison()
    figure_offset()
    figure_bootstrap()
    print(f"generated {len(list(OUTPUT.glob('*.png')))} figures in {OUTPUT}")


if __name__ == "__main__":
    main()
