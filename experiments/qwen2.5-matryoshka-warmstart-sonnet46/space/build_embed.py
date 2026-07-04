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
    assert template.count("__IS_WIDGET__") == 1
    page = (template.replace("__DATA_SOURCE__", source)
            .replace("__IS_WIDGET__", "true" if args.widget else "false"))
    if args.widget:
        style = re.search(r"<style>.*?</style>", page, re.DOTALL).group(0)
        body = re.search(r"<body>(.*)</body>", page, re.DOTALL).group(1)
        # widget-only overrides: fixed-px token panel (vh is circular inside
        # auto-height iframes), side-by-side panels at LW desktop width (the
        # page's 880px breakpoint never fires in a 340-700px iframe), and a
        # denser results card — the right column sets the widget's height.
        overrides = """<style>
.wrap{padding:4px 2px 8px;}
.tokscroll{max-height:320px;font-size:13px;line-height:1.85;}
.tokhead{padding:6px 12px;}
.tabs{margin-bottom:8px;}
.tabs button{padding:4px 10px;font-size:11.5px;}
.modebar button{padding:4px 10px;font-size:11px;}
.nlaviz{padding:10px 10px 10px 8px;}
.nlaviz .sub{margin-bottom:8px;}
.nlaviz .chips{gap:16px;margin-bottom:10px;}
.nlaviz .chip .v{font-size:16px;}
.nlaviz .row,.nlaviz .axisrow{grid-template-columns:10px minmax(0,1fr) 68px 40px;gap:6px;}
.nlaviz .row{padding:3px 2px;}
.nlaviz .line{font-size:11px;line-height:1.35;}
.nlaviz .idx{font-size:9.5px;}
.nlaviz .val{font-size:10.5px;}
.nlaviz .note{margin-top:6px;font-size:10.5px;}
/* the 72px axis track can't fit lo + zero labels without collision */
.nlaviz .axislab span:first-child:nth-last-child(3){display:none;}
/* sandbox can't open links; attribution lives in the post body instead */
footer{display:none;}
@media (min-width:560px){.cols{grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;}}
/* desktop: toggle shares the tabs row (out of the right column's flow).
   .side must drop its sticky positioning or IT becomes the toggle's
   containing block (sticky is inert in an auto-height iframe anyway). */
@media (min-width:640px){
.wrap{position:relative;}
.side{position:static;}
.modebar{position:absolute;top:4px;right:2px;margin-bottom:0;}
}
</style>"""
        page = style + overrides + body
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
