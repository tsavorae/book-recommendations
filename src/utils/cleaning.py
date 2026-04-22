from __future__ import annotations

import html
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from src.config import BOOK_NUMERIC_COLUMNS, GOODREADS_DATE_COLUMNS


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def empty_strings_to_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(r"^\s*$", pd.NA, regex=True)


def normalize_review_text(value: Any) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = html.unescape(str(value))
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text if text else pd.NA


def to_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def parse_goodreads_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in GOODREADS_DATE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_datetime(
                out[column],
                format="%a %b %d %H:%M:%S %z %Y",
                errors="coerce",
                utc=True,
            )
    return out


def parse_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .astype("boolean")
    )


def _json_dumps(value: Any) -> str | pd.NA:
    if value is None or (not isinstance(value, list) and pd.isna(value)):
        return pd.NA
    return json.dumps(value, ensure_ascii=False)


def _authors_summary(authors: Any) -> dict[str, Any]:
    if not isinstance(authors, list):
        return {
            "author_ids": pd.NA,
            "primary_author_id": pd.NA,
            "primary_author_role": pd.NA,
            "author_count": 0,
        }
    ids = [str(item.get("author_id")) for item in authors if isinstance(item, dict) and item.get("author_id")]
    roles = [str(item.get("role", "")) for item in authors if isinstance(item, dict)]
    return {
        "author_ids": "|".join(ids) if ids else pd.NA,
        "primary_author_id": ids[0] if ids else pd.NA,
        "primary_author_role": roles[0] if roles else pd.NA,
        "author_count": len(ids),
    }


def _shelves_summary(shelves: Any, top_n: int = 10) -> dict[str, Any]:
    if not isinstance(shelves, list):
        return {
            "top_shelves": pd.NA,
            "top_shelves_json": pd.NA,
            "shelf_count": 0,
            "to_read_count": np.nan,
        }
    cleaned = []
    to_read_count = np.nan
    for item in shelves:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        count = pd.to_numeric(item.get("count"), errors="coerce")
        if name == "to-read":
            to_read_count = count
        if name:
            cleaned.append({"name": str(name), "count": None if pd.isna(count) else int(count)})
    top = cleaned[:top_n]
    return {
        "top_shelves": "|".join(item["name"] for item in top) if top else pd.NA,
        "top_shelves_json": json.dumps(top, ensure_ascii=False) if top else pd.NA,
        "shelf_count": len(cleaned),
        "to_read_count": to_read_count,
    }


def clean_books(df: pd.DataFrame) -> pd.DataFrame:
    out = empty_strings_to_na(df.copy())
    out = to_numeric_columns(out, BOOK_NUMERIC_COLUMNS)

    if "is_ebook" in out.columns:
        out["is_ebook"] = parse_bool_series(out["is_ebook"])

    if "authors" in out.columns:
        authors = out["authors"].map(_authors_summary).apply(pd.Series)
        out = pd.concat([out.drop(columns=["authors"]), authors], axis=1)

    if "popular_shelves" in out.columns:
        shelves = out["popular_shelves"].map(_shelves_summary).apply(pd.Series)
        out = pd.concat([out.drop(columns=["popular_shelves"]), shelves], axis=1)

    if "series" in out.columns:
        out["series_count"] = out["series"].map(lambda value: len(value) if isinstance(value, list) else 0)
        out["series_json"] = out["series"].map(_json_dumps)
        out = out.drop(columns=["series"])

    if "similar_books" in out.columns:
        out["similar_books_count"] = out["similar_books"].map(
            lambda value: len(value) if isinstance(value, list) else 0
        )
        out["similar_books_json"] = out["similar_books"].map(_json_dumps)
        out = out.drop(columns=["similar_books"])

    for column in ["description", "title", "title_without_series", "publisher", "format"]:
        if column in out.columns:
            out[column] = out[column].astype("string").str.strip()

    date_parts = ["publication_year", "publication_month", "publication_day"]
    if all(column in out.columns for column in date_parts):
        years = out["publication_year"]
        months = out["publication_month"].fillna(1).clip(lower=1, upper=12)
        days = out["publication_day"].fillna(1).clip(lower=1, upper=31)
        out["publication_date"] = pd.to_datetime(
            {"year": years, "month": months, "day": days},
            errors="coerce",
        )

    if "book_id" in out.columns:
        out = (
            out.sort_values(["ratings_count", "text_reviews_count"], ascending=False, na_position="last")
            .drop_duplicates(subset=["book_id"], keep="first")
        )

    return out.reset_index(drop=True)


def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = empty_strings_to_na(df.copy())
    out = parse_goodreads_dates(out)

    if "rating" in out.columns:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
        out["rating_missing"] = out["rating"].isna() | out["rating"].eq(0)
        out["rating_clean"] = out["rating"].where(out["rating"].between(1, 5), pd.NA)

    if "review_text_incomplete" in out.columns:
        out["review_text_clean"] = out["review_text_incomplete"].map(normalize_review_text)
        out["review_text_length"] = out["review_text_clean"].fillna("").str.len()

    if "is_read" in out.columns:
        out["is_read"] = out["is_read"].astype("boolean")

    out = out.drop_duplicates()
    if "review_id" in out.columns:
        sort_cols = [column for column in ["review_id", "date_updated"] if column in out.columns]
        out = out.sort_values(sort_cols).drop_duplicates(subset=["review_id"], keep="last")
    elif {"user_id", "book_id"}.issubset(out.columns):
        out = out.drop_duplicates(subset=["user_id", "book_id"], keep="last")

    return out.reset_index(drop=True)


def add_interaction_aggregates(books: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions.empty or "book_id" not in interactions.columns:
        return books
    grouped = interactions.groupby("book_id", dropna=False).agg(
        interaction_count=("book_id", "size"),
        read_count=("is_read", "sum"),
        explicit_rating_count=("rating_clean", "count"),
        mean_user_rating=("rating_clean", "mean"),
        review_text_count=("review_text_clean", lambda values: values.notna().sum()),
    )
    grouped = grouped.reset_index()
    return books.merge(grouped, on="book_id", how="left")


def cap_outlier_features(df: pd.DataFrame, columns: list[str], quantile: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            cap = out[column].quantile(quantile)
            out[f"{column}_p99_capped"] = out[column].clip(upper=cap)
    return out
