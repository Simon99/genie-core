from __future__ import annotations

import html

DEFAULT_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
                 "PingFang TC", "Noto Sans CJK TC", "Microsoft JhengHei", sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 2em 1.5em;
    line-height: 1.7;
    color: #24292e;
}
h1, h2, h3 { line-height: 1.3; }
h1 { border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; }
h2 { border-bottom: 1px solid #eaecef; padding-bottom: 0.2em; margin-top: 1.6em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #dfe2e5; padding: 0.5em 0.8em; text-align: left; }
th { background: #f6f8fa; }
tr:nth-child(even) { background: #fafbfc; }
code, pre { font-family: "SF Mono", Menlo, Consolas, monospace; background: #f6f8fa; }
code { padding: 0.15em 0.35em; border-radius: 3px; font-size: 0.9em; }
pre { padding: 1em; border-radius: 6px; overflow-x: auto; }
img { max-width: 100%; }
blockquote { margin: 1em 0; padding: 0 1em; color: #6a737d; border-left: 4px solid #dfe2e5; }
.timestamp { color: #6a737d; font-size: 0.9em; }
""".strip()


def esc(text) -> str:
    """HTML-escape untrusted text (LLM output, transcripts, user input)."""
    return html.escape(str(text))


def html_page(title: str, body_html: str, css: str = None) -> str:
    """Wrap body_html in a complete HTML5 document.

    title is escaped; body_html is inserted as-is (escape untrusted pieces
    with esc() when building it). css defaults to a readable report style.
    """
    if css is None:
        css = DEFAULT_CSS
    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-Hant">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>%s</title>\n"
        "<style>\n%s\n</style>\n"
        "</head>\n"
        "<body>\n%s\n</body>\n"
        "</html>\n"
    ) % (esc(title), css, body_html)
