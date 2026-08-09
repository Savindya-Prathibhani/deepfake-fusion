"""
run_cli.py — headless entry point. Prints the same JSON the UI shows.

    python run_cli.py path/to/file.mp4
    python run_cli.py path/to/file.wav --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import DeepfakePipeline, load_settings   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Audio deepfake detection (AASIST + BAM + fusion)")
    ap.add_argument("input", nargs="+", help="audio or video file(s)")
    ap.add_argument("--out", help="write JSON here instead of stdout")
    ap.add_argument("--config", help="alternative fusion.yaml")
    ap.add_argument("--device", help="cpu or cuda (default: auto)")
    ap.add_argument("--quiet", action="store_true",
                    help="JSON only, no summary line on stderr")
    args = ap.parse_args()

    pipeline = DeepfakePipeline(settings=load_settings(args.config), device=args.device)

    results = []
    for path in args.input:
        result = pipeline.analyze(path)
        results.append({k: v for k, v in result.items() if not k.startswith("_")})
        if not args.quiet:
            print(summarise(results[-1]), file=sys.stderr)

    payload = results[0] if len(results) == 1 else results
    text = json.dumps(payload, indent=2)

    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


def summarise(result: dict) -> str:
    """One line per file, on stderr so it never contaminates piped JSON.

    Tampered ranges are printed in the `start:end` seconds form the
    integration spec asks for, e.g. `2.23:2.90 , 3.89:4.90`.
    """
    fusion = result["fusion"]
    line = (f"{result['input']['filename']}: {fusion['final_prediction']} "
            f"({fusion['confidence']*100:.1f}%, {fusion['rule_fired']})")
    ranges = fusion.get("tampered_ranges")
    if ranges:
        line += "  tampered: " + " , ".join(ranges)
    for warning in fusion.get("warnings", []):
        line += f"\n  ! {warning}"
    return line


if __name__ == "__main__":
    raise SystemExit(main())
