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

Rebuild only when embed_template.html changes (or to switch data source);
data-only changes flow through precache.json + deploy.sh automatically.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_URL = ("https://huggingface.co/spaces/syvb/nla-v3-explorer"
                    "/resolve/main/precache.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-url", default=DEFAULT_DATA_URL)
    ap.add_argument("--inline", action="store_true",
                    help="bake precache.json into the page instead of fetching")
    ap.add_argument("--out", default=str(HERE / "embed.html"))
    args = ap.parse_args()

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
    out = Path(args.out)
    out.write_text(template.replace("__DATA_SOURCE__", source))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, "
          f"{'inline data' if args.inline else 'fetches ' + args.data_url})")


if __name__ == "__main__":
    main()
