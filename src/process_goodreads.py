from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.util import hash_pandas_object

from src.config import CATEGORIES, CategoryConfig
from src.utils.cleaning import cap_outlier_features, clean_books, clean_interactions
from src.utils.io import read_jsonl_chunks, read_jsonl_sample, remove_path, safe_write_parquet


BOOK_FEATURE_COLUMNS = [
    "book_id",
    "work_id",
    "title",
    "title_without_series",
    "average_rating",
    "ratings_count",
    "text_reviews_count",
    "num_pages",
    "publication_year",
    "publication_date",
    "language_code",
    "format",
    "publisher",
    "is_ebook",
    "primary_author_id",
    "author_ids",
    "top_shelves",
]


def _stabilize_parquet_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.select_dtypes(include=["object"]).columns:
        out[column] = out[column].astype("string")
    return out


def _clean_existing_outputs(config: CategoryConfig) -> None:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "books_curated.parquet",
        "interactions_curated.parquet",
        "_interactions_parts",
        "_interactions_buckets",
        "_books_buckets",
        "quality_report.json",
    ]:
        remove_path(config.processed_dir / name)


def _assign_bucket(df: pd.DataFrame, key: pd.Series, bucket_count: int) -> pd.DataFrame:
    out = df.copy()
    out["_bucket"] = (hash_pandas_object(key.astype("string").fillna("__missing__"), index=False) % bucket_count).astype(
        "int64"
    )
    return out


def _write_bucketed_parts(
    df: pd.DataFrame,
    bucket_root: Path,
    batch_idx: int,
    bucket_column: str = "_bucket",
) -> None:
    for bucket, part in df.groupby(bucket_column, sort=False):
        bucket_dir = bucket_root / f"bucket={int(bucket):04d}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        part = part.drop(columns=[bucket_column])
        part = _stabilize_parquet_dtypes(part)
        part.to_parquet(bucket_dir / f"part-{batch_idx:05d}.parquet", index=False, engine="pyarrow")


def _build_books_dataset(config: CategoryConfig, chunksize: int, bucket_count: int) -> pd.DataFrame:
    bucket_root = config.processed_dir / "_books_buckets"
    bucket_root.mkdir(parents=True, exist_ok=True)

    for idx, chunk in enumerate(read_jsonl_chunks(config.books_file, chunksize=chunksize)):
        cleaned = clean_books(chunk)
        if "book_id" not in cleaned.columns:
            continue
        bucketed = _assign_bucket(cleaned, cleaned["book_id"], bucket_count)
        _write_bucketed_parts(bucketed, bucket_root, idx)
        print(f"[{config.key}] books chunk {idx + 1} processed: {len(cleaned):,} rows", flush=True)

    curated_parts: list[pd.DataFrame] = []
    for bucket_dir in sorted(bucket_root.glob("bucket=*")):
        bucket_df = pd.read_parquet(bucket_dir)
        if "book_id" in bucket_df.columns:
            bucket_df = (
                bucket_df.sort_values(["ratings_count", "text_reviews_count"], ascending=False, na_position="last")
                .drop_duplicates(subset=["book_id"], keep="first")
            )
        curated_parts.append(bucket_df)

    remove_path(bucket_root)
    if not curated_parts:
        return pd.DataFrame()
    return pd.concat(curated_parts, ignore_index=True)


def _quality_report(
    config: CategoryConfig,
    raw_books_sample: pd.DataFrame,
    books: pd.DataFrame,
    raw_interactions_sample: pd.DataFrame,
    interactions_rows: int | None,
) -> dict[str, object]:
    return {
        "category": config.key,
        "books_raw_sample_rows": int(len(raw_books_sample)),
        "books_curated_rows": int(len(books)),
        "interactions_raw_sample_rows": int(len(raw_interactions_sample)),
        "interactions_curated_rows": interactions_rows,
        "books_duplicate_book_id": int(books.duplicated("book_id").sum()) if "book_id" in books else None,
        "sample_interactions_duplicate_review_id": (
            int(raw_interactions_sample.duplicated("review_id").sum())
            if "review_id" in raw_interactions_sample
            else None
        ),
        "books_nulls": books.isna().sum().astype(int).to_dict(),
    }


def _write_interactions_dataset(
    config: CategoryConfig,
    books: pd.DataFrame,
    chunksize: int,
    bucket_count: int,
) -> tuple[int | None, pd.DataFrame]:
    bucket_root = config.processed_dir / "_interactions_buckets"
    final_dir = config.processed_dir / "interactions_curated.parquet"
    bucket_root.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    book_features = books[[column for column in BOOK_FEATURE_COLUMNS if column in books.columns]].copy()
    total_rows = 0
    aggregate_parts: list[pd.DataFrame] = []
    for idx, chunk in enumerate(read_jsonl_chunks(config.interactions_file, chunksize=chunksize)):
        cleaned = clean_interactions(chunk)
        if "review_id" in cleaned.columns:
            key = cleaned["review_id"]
        else:
            key = cleaned.get("user_id", pd.Series(index=cleaned.index, dtype="string")).astype("string") + "|" + cleaned.get(
                "book_id", pd.Series(index=cleaned.index, dtype="string")
            ).astype("string")
        bucketed = _assign_bucket(cleaned, key, bucket_count)
        _write_bucketed_parts(bucketed, bucket_root, idx)
        total_rows += len(cleaned)
        print(f"[{config.key}] interactions chunk {idx + 1} staged: {len(cleaned):,} rows", flush=True)

    rows = 0
    for bucket_idx, bucket_dir in enumerate(sorted(bucket_root.glob("bucket=*")), start=1):
        bucket_df = pd.read_parquet(bucket_dir)
        if "review_id" in bucket_df.columns:
            bucket_df = bucket_df.sort_values(["review_id", "date_updated"]).drop_duplicates(
                subset=["review_id"], keep="last"
            )
        elif {"user_id", "book_id"}.issubset(bucket_df.columns):
            bucket_df = bucket_df.drop_duplicates(subset=["user_id", "book_id"], keep="last")

        if "book_id" in bucket_df.columns:
            part_agg = bucket_df.groupby("book_id", dropna=False).agg(
                interaction_count=("book_id", "size"),
                read_count=("is_read", "sum"),
                explicit_rating_count=("rating_clean", "count"),
                rating_sum=("rating_clean", "sum"),
                review_text_count=("review_text_clean", lambda values: values.notna().sum()),
            )
            aggregate_parts.append(part_agg.reset_index())

        if "book_id" in bucket_df.columns and not book_features.empty:
            bucket_df = bucket_df.merge(book_features, on="book_id", how="left", suffixes=("", "_book"))

        bucket_df = _stabilize_parquet_dtypes(bucket_df)
        bucket_df.to_parquet(final_dir / f"part-{bucket_idx:04d}.parquet", index=False, engine="pyarrow")
        rows += len(bucket_df)
        print(f"[{config.key}] interactions bucket {bucket_idx} finalized: {len(bucket_df):,} rows", flush=True)

    remove_path(bucket_root)

    if aggregate_parts:
        aggregates = pd.concat(aggregate_parts, ignore_index=True).groupby("book_id", dropna=False).sum().reset_index()
        aggregates["mean_user_rating"] = aggregates["rating_sum"] / aggregates["explicit_rating_count"]
        aggregates.loc[aggregates["explicit_rating_count"].eq(0), "mean_user_rating"] = pd.NA
        aggregates = aggregates.drop(columns=["rating_sum"])
    else:
        aggregates = pd.DataFrame()

    return rows, aggregates


def process_category(category: str, chunksize: int = 100_000, bucket_count: int = 256) -> dict[str, object]:
    config = CATEGORIES[category]
    _clean_existing_outputs(config)

    raw_books_sample = read_jsonl_sample(config.books_file, nrows=50_000)
    books = _build_books_dataset(config, chunksize=chunksize, bucket_count=min(bucket_count, 64))

    raw_interactions_sample = read_jsonl_sample(config.interactions_file, nrows=50_000)
    clean_interactions(raw_interactions_sample)

    interaction_rows, aggregates = _write_interactions_dataset(
        config,
        books,
        chunksize=chunksize,
        bucket_count=bucket_count,
    )
    if not aggregates.empty and "book_id" in books.columns:
        books = books.merge(aggregates, on="book_id", how="left")
    books = cap_outlier_features(
        books,
        ["ratings_count", "text_reviews_count", "num_pages", "interaction_count"],
    )

    safe_write_parquet(_stabilize_parquet_dtypes(books), config.processed_dir / "books_curated.parquet")

    report = _quality_report(config, raw_books_sample, books, raw_interactions_sample, interaction_rows)
    report_path = config.processed_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate Goodreads datasets by category.")
    parser.add_argument("--category", choices=sorted(CATEGORIES), required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--bucket-count", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = process_category(args.category, chunksize=args.chunksize, bucket_count=args.bucket_count)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
