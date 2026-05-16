"""Convert explainer.md → explainer.html → explainer.pdf.

Pipeline:
  1. Read explainer.md, strip YAML frontmatter, parse into title-block + body.
  2. Convert body markdown to HTML via python-markdown + pymdown-extensions.
  3. Wrap with HTML template (links style.css, loads MathJax CDN for math).
  4. Invoke Chrome headless with --print-to-pdf.
  5. Verify PDF (size > 100KB, page count 10~20).

Run from repo root:
  ./analysis/plan-004/.venv_pdf/bin/python analysis/plan-004/build_pdf.py
"""
import re
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MD_PATH = HERE / "explainer.md"
HTML_PATH = HERE / "explainer.html"
PDF_PATH = HERE / "explainer.pdf"
CSS_PATH = HERE / "style.css"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="{css}">
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
    displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],
    processEscapes: true
  }},
  svg: {{ fontCache: 'global' }},
  startup: {{
    pageReady: () => MathJax.startup.defaultPageReady().then(() => {{
      document.body.setAttribute('data-mathjax-done', '1');
    }})
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
<div class="title-block">
  <div class="title">{title}</div>
  <div class="subtitle">{subtitle}</div>
  <div class="meta">{author} · {date} · LB {lb_score}</div>
</div>
{body}
</body>
</html>
"""


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Strip YAML frontmatter and return (metadata dict, body markdown)."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    fm = text[4:end]
    body = text[end + 5:]
    meta: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body


def main() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")
    meta, body_md = parse_frontmatter(md_text)

    body_html = markdown.markdown(
        body_md,
        extensions=[
            "tables",
            "fenced_code",
            "attr_list",
            "sane_lists",
            "pymdownx.arithmatex",
            "pymdownx.superfences",
        ],
        extension_configs={
            "pymdownx.arithmatex": {"generic": True},
        },
    )

    html = HTML_TEMPLATE.format(
        title=meta.get("title", "Explainer"),
        subtitle=meta.get("subtitle", ""),
        author=meta.get("author", ""),
        date=meta.get("date", ""),
        lb_score=meta.get("lb_score", ""),
        css=CSS_PATH.name,
        body=body_html,
    )
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"[1/3] HTML written → {HTML_PATH} ({len(html):,} bytes)")

    if not Path(CHROME).exists():
        print(f"ERROR: Chrome not found at {CHROME}", file=sys.stderr)
        return 1

    # Chrome headless print-to-pdf.
    # virtual-time-budget gives MathJax time to render before snapshot.
    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--virtual-time-budget=15000",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH.resolve().as_uri(),
    ]
    print(f"[2/3] Chrome headless → {PDF_PATH}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print("Chrome failed:", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    # Verify
    size = PDF_PATH.stat().st_size
    print(f"[3/3] PDF size: {size:,} bytes")
    assert size > 100_000, f"PDF size too small: {size}"

    # Page count via pdfplumber (already installed system-wide)
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(PDF_PATH) as pdf:
            n_pages = len(pdf.pages)
        print(f"      Page count: {n_pages}")
        if not (8 <= n_pages <= 25):
            print(f"      WARNING: page count {n_pages} outside expected range 8~25")
    except ImportError:
        print("      (pdfplumber unavailable — skip page count check)")
    except Exception as e:
        print(f"      (pdfplumber error: {e})")

    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
