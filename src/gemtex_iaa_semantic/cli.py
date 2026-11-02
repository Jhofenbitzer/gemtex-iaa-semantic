"""Command line interface: ``uv run gemtex-iaa-semantic <command> <config>``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gemtex_iaa_semantic.config import load_config

log = logging.getLogger("gemtex_iaa_semantic")


def _setup_logging(verbose: int, out_dir: Path = None) -> None:
    handlers = [logging.StreamHandler()]
    if out_dir is not None:
        handlers.append(logging.FileHandler(out_dir / "gemtex-iaa-semantic.log", encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s", handlers=handlers, force=True)


def cmd_inspect(args, cfg):
    from gemtex_iaa_semantic.inspection import inspect_project
    from gemtex_iaa_semantic.io.loader import get_project
    from gemtex_iaa_semantic.snomed import SnomedStore
    _setup_logging(cfg.VERBOSE)
    data = get_project(cfg, use_cache=not args.no_cache)
    store = SnomedStore(cfg.HDF5) if Path(cfg.HDF5).exists() else None
    if store is None:
        print(f"(HDF5 {cfg.HDF5} not found: SNOMED CT resolution statistics skipped)")
    print(inspect_project(data, cfg, store))


def cmd_site_report(args, cfg):
    from gemtex_iaa_semantic.consortium import site_report
    from gemtex_iaa_semantic.report import run_directory
    if not cfg.SITE:
        raise SystemExit("Set SITE in [default] of the configuration (the name of your site).")
    out_dir = run_directory(cfg.DIR_OUTPUT, "site-report")
    (out_dir / "local").mkdir()
    _setup_logging(cfg.VERBOSE, out_dir / "local")
    report = site_report(cfg, out_dir, use_cache=not args.no_cache)
    print(f"report: {report}\nsend only the folder {report.parent}; {out_dir / 'local'} stays at your site")


def cmd_pool(args):
    from gemtex_iaa_semantic.consortium import pool, write_pool
    from gemtex_iaa_semantic.report import run_directory
    _setup_logging(0)
    result = pool([Path(d) for d in args.shares], n_boot=args.bootstrap, seed=args.seed)
    for warning in result["warnings"]:
        log.warning(warning)
    print(f"report: {write_pool(result, run_directory(Path(args.out), 'pool'))}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gemtex-iaa-semantic", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="print text-free statistics of the exports (layers, formats, annotators)")
    p.add_argument("config", nargs="?", default="config/site.cfg")
    p.add_argument("--no-cache", action="store_true")
    p = sub.add_parser("site-report", help="agreement of the multiply annotated documents of one site, "
                                           "pseudonymised (share/) and with names (local/)")
    p.add_argument("config", nargs="?", default="config/site.cfg")
    p.add_argument("--no-cache", action="store_true")
    p = sub.add_parser("pool", help="combine the share folders of several sites")
    p.add_argument("shares", nargs="+", help="share folders (or site-report run directories)")
    p.add_argument("--out", default="data/results", help="directory for the pooled run (default: data/results)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=13)

    args = parser.parse_args(argv)
    if args.command == "pool":
        return cmd_pool(args) or 0
    cfg = load_config(args.config)
    return {"inspect": cmd_inspect, "site-report": cmd_site_report}[args.command](args, cfg) or 0


if __name__ == "__main__":
    sys.exit(main())
