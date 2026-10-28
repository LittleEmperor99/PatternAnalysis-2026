"""Render the README to A4 PDF with a headless local Chrome/Edge browser.

Run with the optional report environment (requirements-report.txt). No network
upload occurs. The feasibility review occupies its own page; output stays local.
"""

import argparse
import html
from pathlib import Path
import re
import subprocess
import tempfile

from markdown_it import MarkdownIt

PROJECT = Path(__file__).resolve().parent
GITHUB_FILES = "https://github.com/LittleEmperor99/PatternAnalysis-2026/blob/topic-recognition/recognition/Facebook_GCN_Jonte/"

CSS = """
@page { size: A4; margin: 14mm;
        @bottom-right { content: counter(page) " / " counter(pages);
                        font: 8pt Arial, sans-serif; color: #657080; } }
* { box-sizing: border-box; }
body { font: 10.5pt/1.38 Arial, sans-serif; color: #18202b; margin: 0; }
h1 { font-size: 22pt; line-height: 1.15; color: #183c61; }
h2 { font-size: 15pt; margin-top: 18pt; color: #183c61; }
h3 { font-size: 12pt; margin-top: 14pt; }
h1,h2,h3 { break-after: avoid; }
p,li { orphans: 3; widows: 3; }
a { color: #175789; text-decoration: none; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; table-layout: fixed;
        font-size: 7.8pt; line-height: 1.25; margin: 10pt 0; }
th,td { border: .5pt solid #b5c2cf; padding: 4pt; vertical-align: top;
        overflow-wrap: anywhere; }
th { background: #e8eff6; text-align: left; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; padding: 8pt;
      background: #eef2f6; font-size: 8pt; line-height: 1.35; }
code { font-family: Consolas, monospace; font-size: .9em; overflow-wrap: anywhere; }
img { max-width: 100%; max-height: 205mm; width: auto; height: auto;
      display: block; margin: 10pt auto; break-inside: avoid; }
p:has(>img) { break-inside: avoid; }
.feasibility { break-before: page; break-after: page; font-size: 10.5pt; line-height: 1.3; }
.feasibility h2 { margin-top: 0; }
"""


def render_html(readme):
    md = MarkdownIt("commonmark", {"html": False}).enable("table")

    def heading_open(tokens, index, options, env):
        label = tokens[index + 1].content.lower()
        slug = re.sub(r"[^\w -]", "", label).replace(" ", "-")
        tokens[index].attrSet("id", slug)
        return md.renderer.renderToken(tokens, index, options, env)

    md.renderer.rules["heading_open"] = heading_open
    body = md.render(readme.read_text(encoding="utf-8"))
    body, count = re.subn(
        r'(<h2[^>]*>One-page feasibility review</h2>.*?)(?=<h2\b)',
        r'<section class="feasibility">\1</section>', body, flags=re.S,
    )
    if count != 1:
        raise ValueError("Expected exactly one feasibility review section")
    # Preserve local image loading, but make code/document links portable in PDF.
    body = re.sub(r'href="(?!https?://|#)([^\"]+)"',
                  lambda m: 'href="' + GITHUB_FILES + m.group(1) + '"', body)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<base href='{html.escape(readme.parent.as_uri())}/'>"
            f"<title>COMP3710 Facebook GCN Technical Report</title><style>{CSS}</style>"
            f"</head><body>{body}</body></html>")


def export_pdf(readme, output, browser=None):
    readme, output = Path(readme).resolve(), Path(output).resolve()
    candidates = [Path(browser)] if browser else [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ]
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        raise FileNotFoundError("Install Chrome/Edge or pass --browser to its executable")
    output.parent.mkdir(parents=True, exist_ok=True)
    html_file = output.with_suffix(".html")
    html_file.write_text(render_html(readme), encoding="utf-8")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="browser_") as profile:
        result = subprocess.run(
            [str(executable), "--headless=new", "--disable-gpu", "--no-first-run",
             "--disable-background-networking", "--disable-extensions", "--no-pdf-header-footer",
             "--allow-file-access-from-files", f"--user-data-dir={profile}",
             f"--print-to-pdf={output}", html_file.as_uri()],
            capture_output=True, text=True, timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if not output.exists() or output.stat().st_size < 1000:
        raise RuntimeError(f"PDF export failed ({result.returncode}): {result.stderr[-2000:]}")
    print(f"Exported {output} ({output.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=PROJECT / "README.md")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/submission/COMP3710_Facebook_GCN_Report.pdf")
    parser.add_argument("--browser", type=Path)
    args = parser.parse_args()
    export_pdf(args.readme, args.output, args.browser)


if __name__ == "__main__":
    main()
