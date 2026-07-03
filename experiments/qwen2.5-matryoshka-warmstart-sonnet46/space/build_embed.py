"""Generate embed.html — the fully static, client-side explorer for blog posts.

No server-side anything: the page is plain HTML/CSS/JS that fetches
precache.json (the same file the Space uses) and renders the token panel +
FVE cards entirely in the browser. Host embed.html anywhere (or iframe it);
the default data URL points at the Space repo, so redeploying the Space
(deploy.sh) refreshes the page's data with no HTML rebuild.

Usage:
    python3 build_embed.py                     # → embed.html (fetches from the Space repo)
    python3 build_embed.py --data-url URL      # fetch from elsewhere
    python3 build_embed.py --inline            # → embed.html with the data baked in
                                               #   (self-contained, ~0.5MB, no CORS/network)
    python3 build_embed.py --widget            # → embed_widget.html: a document-shell-free
                                               #   fragment for sandboxed-iframe embeds
                                               #   (e.g. LessWrong post widgets)

The --widget fragment swaps the token panel's vh-based max-height for a fixed
one (an auto-height-measured iframe makes vh circular) and trims page padding.

Rebuild only when embed_template.html changes (or to switch data source);
data-only changes flow through precache.json + deploy.sh automatically.
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_URL = ("https://huggingface.co/spaces/syvb/nla-v3-explorer"
                    "/resolve/main/precache.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-url", default=DEFAULT_DATA_URL)
    ap.add_argument("--inline", action="store_true",
                    help="bake precache.json into the page instead of fetching")
    ap.add_argument("--widget", action="store_true",
                    help="emit a <style>+body fragment for sandboxed-iframe embeds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or (HERE / ("embed_widget.html" if args.widget else "embed.html")))

    if args.inline:
        data = json.loads((HERE / "precache.json").read_text())
        assert all("pieces" in e for e in data["entries"]), (
            "precache.json has no 'pieces' — regenerate it "
            "(bash precompute_on_vast.sh)")
        # </script> inside a JSON string would end the script block early
        source = json.dumps({"inline": data}).replace("</", "<\\/")
    else:
        source = json.dumps({"url": args.data_url})

    template = (HERE / "embed_template.html").read_text()
    assert template.count("__DATA_SOURCE__") == 1
    page = template.replace("__DATA_SOURCE__", source)
    if args.widget:
        style = re.search(r"<style>.*?</style>", page, re.DOTALL).group(0)
        body = re.search(r"<body>(.*)</body>", page, re.DOTALL).group(1)
        page = (style
                + "<style>.wrap{padding:4px 2px 10px;}"
                  ".tokscroll{max-height:320px;}"
                  # side-by-side panels: the iframe viewport (340-700px) never
                  # reaches the page's 880px two-column breakpoint
                  "@media (min-width:560px){"
                  ".cols{grid-template-columns:minmax(0,1fr) minmax(0,1fr);}}"
                  "</style>"
                + body)
        # LessWrong's publish pipeline entity-decodes widget source once; the
        # fragment must be a fixed point of that transform or publishing
        # breaks the JS (learned the hard way with "&quot;" in the escaper)
        import html as html_lib
        assert html_lib.unescape(page) == page, (
            "widget fragment contains decodable HTML entities — publish-unsafe")
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, "
          f"{'widget fragment, ' if args.widget else ''}"
          f"{'inline data' if args.inline else 'fetches ' + args.data_url})")


if __name__ == "__main__":
    main()
