"""Offline CSV → FAISS ingest CLI.

Usage (from ELIXIR/):
  python -m ingest.cli --csv fixtures/sample_wines.csv --out indexes/<version>
  python -m ingest.cli --csv-dir ../combined_csv --content-mode enriched
  python -m ingest.cli --csv path/to/wines.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingest.build_index import (
    build_and_save_index,
    make_version_id,
    sha256_file,
    sha256_paths,
)
from ingest.csv_loader import load_documents_from_csv, load_documents_from_csv_dir
from ingest.schema import CatalogColumnMapping, ColumnMapping, SchemaKind
from rag.config import BASE_DIR, get_settings
from rag.embeddings import get_embeddings, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest wine-review or Living Liquidz catalog CSV(s) into FAISS.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--csv",
        type=Path,
        help="Path to a single CSV file",
    )
    src.add_argument(
        "--csv-dir",
        type=Path,
        help="Directory of CSVs (e.g. workspace combined_csv/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: indexes/<auto-version>)",
    )
    parser.add_argument(
        "--content-mode",
        choices=("description", "enriched"),
        default=None,
        help="Document text mode (default: INGEST_CONTENT_MODE or description)",
    )
    parser.add_argument(
        "--schema",
        choices=("auto", "catalog", "winemag"),
        default="auto",
        help="CSV schema (default: auto-detect from headers)",
    )
    parser.add_argument("--glob", default="*.csv", help="Glob when using --csv-dir")
    # Winemag overrides
    parser.add_argument("--description-col", default="description")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--variety-col", default="variety")
    parser.add_argument("--winery-col", default="winery")
    parser.add_argument("--province-col", default="province")
    parser.add_argument("--region-1-col", default="region_1")
    parser.add_argument("--country-col", default="country")
    parser.add_argument("--points-col", default="points")
    parser.add_argument("--price-col", default="price")
    parser.add_argument("--taster-col", default="taster_name")
    parser.add_argument("--id-col", default="id")
    return parser


def _resolve_existing(path: Path) -> Path | None:
    if path.is_absolute() and path.exists():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    base_candidate = (BASE_DIR / path).resolve()
    if base_candidate.exists():
        return base_candidate
    # Workspace parent (combined_csv next to ELIXIR/)
    parent_candidate = (BASE_DIR.parent / path).resolve()
    if parent_candidate.exists():
        return parent_candidate
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings(require_api_key=False)
    content_mode = args.content_mode or settings.ingest_content_mode

    schema_kind: SchemaKind | None
    if args.schema == "auto":
        schema_kind = None
    elif args.schema == "catalog":
        schema_kind = SchemaKind.CATALOG
    else:
        schema_kind = SchemaKind.WINEMAG

    winemag_mapping = ColumnMapping(
        description=args.description_col,
        title=args.title_col,
        variety=args.variety_col,
        winery=args.winery_col,
        province=args.province_col,
        region_1=args.region_1_col,
        country=args.country_col,
        points=args.points_col,
        price=args.price_col,
        taster_name=args.taster_col,
        id_col=args.id_col,
    )
    catalog_mapping = CatalogColumnMapping()

    source_files: list[str]
    try:
        if args.csv_dir is not None:
            csv_dir = _resolve_existing(args.csv_dir)
            if csv_dir is None or not csv_dir.is_dir():
                print(f"error: CSV directory not found: {args.csv_dir}", file=sys.stderr)
                return 1
            paths = sorted(csv_dir.glob(args.glob))
            if not paths:
                print(
                    f"error: no files matching {args.glob!r} in {csv_dir}",
                    file=sys.stderr,
                )
                return 1
            source_hash = sha256_paths(paths)
            version_seed = source_hash
            if args.out is not None:
                out_dir = args.out
                if not out_dir.is_absolute():
                    out_dir = (BASE_DIR / out_dir).resolve()
                version = out_dir.name
            else:
                version = make_version_id(version_seed)
                out_dir = (BASE_DIR / "indexes" / version).resolve()

            print(f"Loading CSV dir: {csv_dir} ({len(paths)} files)")
            documents, source_files = load_documents_from_csv_dir(
                csv_dir,
                ingest_version=version,
                embedding_model=settings.embedding_model,
                content_mode=content_mode,
                schema_kind=schema_kind,
                pattern=args.glob,
            )
        else:
            csv_path = _resolve_existing(args.csv)
            if csv_path is None or not csv_path.is_file():
                print(f"error: CSV not found: {args.csv}", file=sys.stderr)
                return 1
            source_hash = sha256_file(csv_path)
            if args.out is not None:
                out_dir = args.out
                if not out_dir.is_absolute():
                    out_dir = (BASE_DIR / out_dir).resolve()
                version = out_dir.name
            else:
                version = make_version_id(source_hash)
                out_dir = (BASE_DIR / "indexes" / version).resolve()

            print(f"Loading CSV: {csv_path}")
            documents = load_documents_from_csv(
                csv_path,
                ingest_version=version,
                embedding_model=settings.embedding_model,
                content_mode=content_mode,
                schema_kind=schema_kind,
                winemag_mapping=winemag_mapping,
                catalog_mapping=catalog_mapping,
            )
            source_files = [csv_path.name]
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Documents: {len(documents)} (content_mode={content_mode})")
    device = resolve_device(settings.torch_device_setting)
    print(f"Embedding model={settings.embedding_model} device={device}")
    embeddings = get_embeddings(settings=settings, device=device)

    print(f"Writing index -> {out_dir}")
    build_and_save_index(
        documents,
        embeddings,
        out_dir,
        version=version,
        embedding_model=settings.embedding_model,
        source_files=source_files,
        source_sha256=source_hash,
        content_mode=content_mode,
        retriever_k=settings.retriever_k,
    )
    print(f"Done. version={version} path={out_dir}")
    print(f"Point FAISS_INDEX_PATH at this directory (or set INDEX_VERSION={version}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
