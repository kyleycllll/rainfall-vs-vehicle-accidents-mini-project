#!/usr/bin/env python3
"""Reconstruct a .ipynb notebook from a Colab/nbconvert HTML export.

The HTML export preserves, per cell, the exact source (code cells) or
rendered markdown (markdown cells), plus outputs (stream text, stderr,
display_data HTML tables, and base64 PNG images). This script walks that
structure in document order and reassembles a valid nbformat v4 notebook.
"""
import base64
import re
import sys
from pathlib import Path

import nbformat
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_md

SOURCE_HTML = Path(__file__).resolve().parent.parent / "source" / "SDAB_Group1.html"
OUTPUT_IPYNB = Path(__file__).resolve().parent.parent / "SDAB_Group1.ipynb"


def clean_markdown_html(md_div):
    """Strip nbconvert's anchor-link pilcrows before HTML->Markdown conversion."""
    soup_copy = BeautifulSoup(str(md_div), "lxml")
    for a in soup_copy.select("a.anchor-link"):
        a.decompose()
    return soup_copy.div or soup_copy


def markdown_cell_from(cell_div):
    md_div = cell_div.select_one("div.jp-RenderedMarkdown")
    if md_div is None:
        return nbformat.v4.new_markdown_cell("")
    cleaned = clean_markdown_html(md_div)
    text = html_to_md(str(cleaned), heading_style="ATX", bullets="-").strip()
    # collapse >2 consecutive blank lines left behind by block-level tags
    text = re.sub(r"\n{3,}", "\n\n", text)
    return nbformat.v4.new_markdown_cell(text)


def extract_execution_count(cell_div):
    prompt = cell_div.select_one("div.jp-InputPrompt")
    if not prompt:
        return None
    m = re.search(r"In\s*\[\s*(\d+)\s*\]", prompt.get_text())
    return int(m.group(1)) if m else None


def extract_source(cell_div):
    pre = cell_div.select_one("div.jp-InputArea-editor pre")
    if pre is None:
        return ""
    return pre.get_text()


def dataframe_html_from_colab_wrapper(html_div):
    """Colab wraps df outputs in a .colab-df-container with interactive JS
    buttons. Pull out just the <style> + <table class="dataframe"> + the
    trailing 'N rows x M columns' <p>, matching plain pandas HTML repr."""
    container = html_div.select_one(".colab-df-container")
    if container is None:
        return None
    inner = container.find("div")
    if inner is None:
        return None
    parts = []
    style = inner.find("style")
    table = inner.find("table", class_="dataframe")
    trailer = inner.find("p")
    if style:
        parts.append(str(style))
    if table:
        parts.append(str(table))
    if trailer:
        parts.append(str(trailer))
    if not parts:
        return None
    return "\n".join(parts)


def html_output_data(html_div):
    df_html = dataframe_html_from_colab_wrapper(html_div)
    if df_html is not None:
        return df_html
    # generic text/html output: strip <script> tags (not renderable/safe in
    # a static notebook viewer) and keep the rest
    soup_copy = BeautifulSoup(str(html_div), "lxml")
    for script in soup_copy.find_all("script"):
        script.decompose()
    target = soup_copy.find(attrs={"data-mime-type": "text/html"})
    return "".join(str(c) for c in target.contents) if target else ""


IMG_SRC_RE = re.compile(r"^data:image/(?P<fmt>\w+);base64,(?P<data>.+)$", re.DOTALL)


def build_output(child_div):
    img = child_div.select_one("div.jp-RenderedImage img")
    if img is not None:
        m = IMG_SRC_RE.match(img.get("src", ""))
        if m:
            return {
                "output_type": "display_data",
                "data": {f"image/{m.group('fmt')}": m.group("data")},
                "metadata": {},
            }

    mime_div = child_div.find(attrs={"data-mime-type": True})
    if mime_div is None:
        return None
    mime = mime_div.get("data-mime-type")

    if mime == "text/plain":
        pre = mime_div.find("pre")
        text = pre.get_text() if pre else mime_div.get_text()
        return {"output_type": "stream", "name": "stdout", "text": text}

    if mime == "application/vnd.jupyter.stderr":
        pre = mime_div.find("pre")
        text = pre.get_text() if pre else mime_div.get_text()
        return {"output_type": "stream", "name": "stderr", "text": text}

    if mime == "text/html":
        return {
            "output_type": "display_data",
            "data": {"text/html": html_output_data(mime_div)},
            "metadata": {},
        }

    return None


def code_cell_from(cell_div):
    source = extract_source(cell_div)
    exec_count = extract_execution_count(cell_div)
    outputs = []
    output_area = cell_div.select_one("div.jp-OutputArea")
    if output_area is not None:
        for child in output_area.select("div.jp-OutputArea-child"):
            out = build_output(child)
            if out is not None:
                outputs.append(nbformat.v4.new_output(**out))
    cell = nbformat.v4.new_code_cell(source, execution_count=exec_count, outputs=outputs)
    return cell


def convert(html_path: Path, out_path: Path):
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    title = soup.title.get_text(strip=True) if soup.title else out_path.stem

    nb = nbformat.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"name": title, "provenance": []},
    }

    cells = []
    for cell_div in soup.select("div.jp-Cell"):
        classes = cell_div.get("class", [])
        if "jp-MarkdownCell" in classes:
            cells.append(markdown_cell_from(cell_div))
        elif "jp-CodeCell" in classes:
            cells.append(code_cell_from(cell_div))
    nb["cells"] = cells

    nbformat.validate(nb)
    nbformat.write(nb, str(out_path))
    return nb


if __name__ == "__main__":
    nb = convert(SOURCE_HTML, OUTPUT_IPYNB)
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    n_outputs = sum(len(c.get("outputs", [])) for c in nb.cells)
    print(f"Wrote {OUTPUT_IPYNB}")
    print(f"cells: {len(nb.cells)} (markdown={n_md}, code={n_code}), total outputs={n_outputs}")
