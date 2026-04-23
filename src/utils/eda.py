from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.utils.cleaning import clean_books, clean_interactions


def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    def unique_count(column: pd.Series) -> int:
        try:
            return int(column.nunique(dropna=True))
        except TypeError:
            return int(column.map(repr).nunique(dropna=True))

    return (
        pd.DataFrame(
            {
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "non_null": df.notna().sum().values,
                "nulls": df.isna().sum().values,
                "unique": [unique_count(df[column]) for column in df.columns],
            }
        )
        .sort_values("column")
        .reset_index(drop=True)
    )


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    empty_strings = df.astype("string").eq("").sum()
    return (
        pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing": df.isna().sum(),
                "empty_strings": empty_strings,
                "missing_pct": df.isna().mean().round(4),
            }
        )
        .sort_values(["missing_pct", "empty_strings"], ascending=False)
        .reset_index(names="column")
    )


def duplicate_summary(books: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame, columns in [
        ("books.book_id", books, ["book_id"]),
        ("books.work_id", books, ["work_id"]),
        ("interactions.review_id", interactions, ["review_id"]),
        ("interactions.user_id_book_id", interactions, ["user_id", "book_id"]),
    ]:
        if all(column in frame.columns for column in columns):
            rows.append(
                {
                    "key": name,
                    "duplicates": int(frame.duplicated(subset=columns).sum()),
                    "records": int(len(frame)),
                }
            )
    return pd.DataFrame(rows)


def numeric_profile(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    selected = columns or df.select_dtypes(include="number").columns.tolist()
    selected = [column for column in selected if column in df.columns]
    if not selected:
        return pd.DataFrame()
    return df[selected].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.reset_index(names="column")


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "upper_iqr_bound": upper,
                "outliers_iqr": int((values > upper).sum()),
            }
        )
    return pd.DataFrame(rows)


def categorical_profile(df: pd.DataFrame, columns: list[str] | None = None, top_n: int = 20) -> pd.DataFrame:
    selected = columns or ["language_code", "format", "publisher", "country_code", "is_ebook"]
    rows = []
    for column in selected:
        if column not in df.columns:
            continue
        counts = df[column].astype("string").fillna("<missing>").value_counts(dropna=False).head(top_n)
        total = len(df)
        for value, count in counts.items():
            rows.append({"column": column, "value": value, "count": int(count), "pct": count / total if total else 0})
    return pd.DataFrame(rows)


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _bin_counts(series: pd.Series, bins: list[float], labels: list[str], count_name: str) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=[series.name or "bin", count_name, "pct"])
    binned = pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
    counts = binned.value_counts(sort=False).reset_index()
    counts.columns = [series.name or "bin", count_name]
    total = counts[count_name].sum()
    counts["pct"] = counts[count_name] / total if total else 0
    return counts


def author_profile(books: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = []
    for authors in books.get("authors", pd.Series(dtype=object)):
        parsed = _iter_dicts(authors)
        rows.append(
            {
                "author_ids": [str(item.get("author_id")) for item in parsed if item.get("author_id")],
                "primary_author_id": str(parsed[0].get("author_id")) if parsed and parsed[0].get("author_id") else pd.NA,
                "primary_author_role": str(parsed[0].get("role", "")) if parsed else pd.NA,
                "author_count": sum(1 for item in parsed if item.get("author_id")),
            }
        )
    parsed_authors = pd.DataFrame(rows)
    if parsed_authors.empty:
        parsed_authors = pd.DataFrame(columns=["author_ids", "primary_author_id", "primary_author_role", "author_count"])

    author_counts = parsed_authors["author_count"].fillna(0).astype(int)
    author_count_dist = author_counts.value_counts().sort_index().reset_index()
    author_count_dist.columns = ["author_count_bin", "books"]
    author_count_dist["author_count_bin"] = author_count_dist["author_count_bin"].astype(str)
    author_count_dist["pct"] = author_count_dist["books"] / len(parsed_authors) if len(parsed_authors) else 0

    role_dist = parsed_authors["primary_author_role"].fillna("<missing>").replace("", "<blank>").value_counts().reset_index()
    role_dist.columns = ["role", "books"]
    role_dist["pct"] = role_dist["books"] / len(parsed_authors) if len(parsed_authors) else 0

    top_authors = parsed_authors["primary_author_id"].dropna().value_counts().reset_index()
    top_authors.columns = ["author_id", "book_count"]

    return {
        "author_count_dist": author_count_dist,
        "role_dist": role_dist,
        "top_authors": top_authors,
    }


def shelves_profile(books: pd.DataFrame, top_n: int = 30) -> dict[str, pd.DataFrame]:
    shelf_counts = []
    to_read_counts = []
    shelf_counter: Counter[str] = Counter()
    pair_counter: Counter[tuple[str, str]] = Counter()

    for shelves in books.get("popular_shelves", pd.Series(dtype=object)):
        parsed = _iter_dicts(shelves)
        names = []
        to_read = np.nan
        for item in parsed:
            name = item.get("name")
            if not name:
                continue
            name = str(name)
            names.append(name)
            count = pd.to_numeric(item.get("count"), errors="coerce")
            shelf_counter[name] += 1
            if name == "to-read":
                to_read = count
        shelf_counts.append(len(names))
        to_read_counts.append(to_read)
        for pair in combinations(sorted(set(names[:top_n])), 2):
            pair_counter[pair] += 1

    shelf_count_series = pd.Series(shelf_counts, name="shelf_count")
    shelf_count_dist = shelf_count_series.value_counts().sort_index().reset_index()
    shelf_count_dist.columns = ["shelf_count", "books"]
    shelf_count_dist["pct"] = shelf_count_dist["books"] / len(shelf_count_series) if len(shelf_count_series) else 0

    to_read_series = pd.to_numeric(pd.Series(to_read_counts, name="to_read_count"), errors="coerce").dropna()
    to_read_dist = (
        to_read_series.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .reset_index()
        .rename(columns={"index": "stat", "to_read_count": "value"})
        if not to_read_series.empty
        else pd.DataFrame(columns=["stat", "value"])
    )

    top_shelf_names = pd.DataFrame(
        [{"shelf_name": name, "book_count": count} for name, count in shelf_counter.most_common(top_n)]
    )
    cooccurrence_sample = pd.DataFrame(
        [
            {"shelf_a": pair[0], "shelf_b": pair[1], "book_count": count}
            for pair, count in pair_counter.most_common(top_n)
        ]
    )

    return {
        "shelf_count_dist": shelf_count_dist,
        "to_read_dist": to_read_dist,
        "top_shelf_names": top_shelf_names,
        "cooccurrence_sample": cooccurrence_sample,
    }


def series_profile(books: pd.DataFrame) -> dict[str, Any]:
    counts = books.get("series", pd.Series(dtype=object)).map(lambda value: len(value) if isinstance(value, list) else 0)
    standalone = counts.eq(0)
    standalone_vs_series = pd.DataFrame(
        [
            {"group": "standalone", "books": int(standalone.sum()), "pct": standalone.mean() if len(counts) else 0},
            {"group": "in_series", "books": int((~standalone).sum()), "pct": (~standalone).mean() if len(counts) else 0},
        ]
    )
    series_length_dist = counts.value_counts().sort_index().reset_index()
    series_length_dist.columns = ["series_count", "books"]
    series_length_dist["pct"] = series_length_dist["books"] / len(counts) if len(counts) else 0
    in_series = counts[counts.gt(0)]
    return {
        "standalone_vs_series": standalone_vs_series,
        "series_length_dist": series_length_dist,
        "summary": {
            "pct_in_series": float(counts.gt(0).mean()) if len(counts) else 0.0,
            "median_length": float(in_series.median()) if not in_series.empty else 0.0,
        },
    }


def combine_books_interactions(books: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join between books and interactions to have a complete dataset for cross-analysis.
    """
    return pd.merge(
        books,
        interactions,
        on="book_id",
        how="inner",
    )


def engagement_profile(books: pd.DataFrame, interactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if interactions.empty:
        empty_modes = pd.DataFrame(columns=["mode", "interactions", "pct"])
        return {
            "engagement_modes": empty_modes,
            "rating_agreement": pd.DataFrame(columns=["book_id", "average_rating", "mean_user_rating"]),
            "popularity_distribution": pd.DataFrame(columns=["book_id", "interaction_count"]),
        }

    data = interactions.copy()
    if "rating_clean" not in data.columns and "rating" in data.columns:
        data["rating_clean"] = pd.to_numeric(data["rating"], errors="coerce").where(lambda s: s.between(1, 5), pd.NA)
    if "review_text_clean" not in data.columns:
        data["review_text_clean"] = pd.NA

    has_rating = data.get("rating_clean", pd.Series(index=data.index, dtype="float")).notna()
    has_review = data.get("review_text_clean", pd.Series(index=data.index, dtype="object")).notna()
    is_read = data.get("is_read", pd.Series(False, index=data.index)).fillna(False).astype(bool)

    modes = pd.Series(np.select([has_review, has_rating, is_read], ["review", "rating_only", "read_no_rating"], "shelf_only"))
    engagement_modes = modes.value_counts().reset_index()
    engagement_modes.columns = ["mode", "interactions"]
    engagement_modes["pct"] = engagement_modes["interactions"] / len(data) if len(data) else 0

    if "book_id" in data.columns:
        popularity_distribution = data.groupby("book_id", dropna=False).size().reset_index(name="interaction_count")
        rating_agreement = data.groupby("book_id", dropna=False)["rating_clean"].mean().reset_index(name="mean_user_rating")
        if {"book_id", "average_rating"}.issubset(books.columns):
            average = books[["book_id", "average_rating"]].copy()
            average["average_rating"] = pd.to_numeric(average["average_rating"], errors="coerce")
            rating_agreement = average.merge(rating_agreement, on="book_id", how="inner")
    else:
        popularity_distribution = pd.DataFrame(columns=["book_id", "interaction_count"])
        rating_agreement = pd.DataFrame(columns=["book_id", "average_rating", "mean_user_rating"])

    return {
        "engagement_modes": engagement_modes,
        "rating_agreement": rating_agreement,
        "popularity_distribution": popularity_distribution,
    }


def reading_duration_profile(interactions: pd.DataFrame) -> dict[str, Any]:
    if {"started_at", "read_at"}.issubset(interactions.columns):
        started = pd.to_datetime(interactions["started_at"], errors="coerce", utc=True)
        read = pd.to_datetime(interactions["read_at"], errors="coerce", utc=True)
        duration = (read - started).dt.days
    elif "reading_duration_days" in interactions.columns:
        duration = pd.to_numeric(interactions["reading_duration_days"], errors="coerce")
    else:
        duration = pd.Series(dtype="float64")

    valid = duration.where(duration.between(0, 365)).dropna()
    duration_stats = (
        valid.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .reset_index()
        .rename(columns={"index": "stat", 0: "value"})
        if not valid.empty
        else pd.DataFrame(columns=["stat", "value"])
    )
    if "value" not in duration_stats.columns and len(duration_stats.columns) == 2:
        duration_stats.columns = ["stat", "value"]

    return {
        "duration_distribution": valid,
        "duration_stats": duration_stats,
        "pct_with_duration": float(valid.count() / len(interactions)) if len(interactions) else 0.0,
    }


def user_rating_behavior(interactions: pd.DataFrame) -> dict[str, Any]:
    if interactions.empty or "user_id" not in interactions.columns:
        return {
            "ratings_per_user": pd.DataFrame(columns=["stat", "value"]),
            "mean_rating_per_user": pd.Series(dtype="float64"),
            "user_type_dist": pd.DataFrame(columns=["user_type", "users", "pct"]),
        }

    rating = interactions.get("rating_clean")
    if rating is None and "rating" in interactions.columns:
        rating = pd.to_numeric(interactions["rating"], errors="coerce").where(lambda s: s.between(1, 5), pd.NA)
    if rating is None:
        rating = pd.Series(pd.NA, index=interactions.index)

    rated = interactions.assign(rating_clean=rating).dropna(subset=["rating_clean"])
    counts = rated.groupby("user_id")["rating_clean"].size()
    means = rated.groupby("user_id")["rating_clean"].mean()

    ratings_per_user = (
        counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .reset_index()
        .rename(columns={"index": "stat", "rating_clean": "value"})
        if not counts.empty
        else pd.DataFrame(columns=["stat", "value"])
    )
    if "value" not in ratings_per_user.columns and len(ratings_per_user.columns) == 2:
        ratings_per_user.columns = ["stat", "value"]

    eligible = pd.DataFrame({"rating_count": counts, "mean_rating": means})
    eligible = eligible[eligible["rating_count"].ge(3)]
    labels = np.select(
        [eligible["mean_rating"].lt(3), eligible["mean_rating"].between(3, 3.67, inclusive="left")],
        ["strict", "balanced"],
        "generous",
    )
    user_type_dist = pd.Series(labels).value_counts().reset_index() if len(eligible) else pd.DataFrame(columns=["index", "count"])
    user_type_dist.columns = ["user_type", "users"]
    user_type_dist["pct"] = user_type_dist["users"] / len(eligible) if len(eligible) else 0

    return {
        "ratings_per_user": ratings_per_user,
        "mean_rating_per_user": means,
        "user_type_dist": user_type_dist,
    }


def isbn_quality_profile(books: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["isbn", "isbn13", "asin", "kindle_asin"]:
        if column in books.columns:
            present = books[column].astype("string").str.strip().replace("", pd.NA).notna()
            rows.append(
                {
                    "column": column,
                    "present": int(present.sum()),
                    "missing": int((~present).sum()),
                    "present_pct": float(present.mean()) if len(present) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def platform_growth_profile(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame(columns=["year", "new_interactions", "pct_with_rating"])
    date_column = "date_added" if "date_added" in interactions.columns else "date_updated"
    if date_column not in interactions.columns:
        return pd.DataFrame(columns=["year", "new_interactions", "pct_with_rating"])
    data = interactions.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce", utc=True)
    data = data.dropna(subset=[date_column])
    if data.empty:
        return pd.DataFrame(columns=["year", "new_interactions", "pct_with_rating"])
    if "rating_clean" not in data.columns and "rating" in data.columns:
        data["rating_clean"] = pd.to_numeric(data["rating"], errors="coerce").where(lambda s: s.between(1, 5), pd.NA)
    data["year"] = data[date_column].dt.year
    return (
        data.groupby("year")
        .agg(
            new_interactions=("year", "size"),
            pct_with_rating=("rating_clean", lambda values: values.notna().mean() if len(values) else 0),
        )
        .reset_index()
        .sort_values("year")
    )


def build_eda_profile(
    books_raw: pd.DataFrame,
    interactions_raw: pd.DataFrame,
    book_numeric_columns: list[str] | None = None,
    interaction_numeric_columns: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    books_clean = clean_books(books_raw)
    interactions_clean = clean_interactions(interactions_raw)

    artifacts: dict[str, Any] = {
        "books_schema": schema_summary(books_raw),
        "interactions_schema": schema_summary(interactions_raw),
        "books_missing": missing_summary(books_raw),
        "interactions_missing": missing_summary(interactions_raw),
        "books_numeric_profile": numeric_profile(books_clean, book_numeric_columns),
        "interactions_numeric_profile": numeric_profile(interactions_clean, interaction_numeric_columns),
        "books_outliers": iqr_outlier_summary(books_clean, book_numeric_columns or []),
        "interactions_outliers": iqr_outlier_summary(interactions_clean, interaction_numeric_columns or []),
        "duplicates": duplicate_summary(books_clean, interactions_clean),
        "categoricals": categorical_profile(books_clean),
    }

    for profile in [
        author_profile(books_raw),
        shelves_profile(books_raw),
        series_profile(books_raw),
        engagement_profile(books_clean, interactions_clean),
        reading_duration_profile(interactions_clean),
        user_rating_behavior(interactions_clean),
    ]:
        artifacts.update(profile)

    summary = pd.DataFrame(
        [
            {"metric": "books_raw_rows", "value": len(books_raw)},
            {"metric": "interactions_raw_rows", "value": len(interactions_raw)},
            {"metric": "books_clean_rows", "value": len(books_clean)},
            {"metric": "interactions_clean_rows", "value": len(interactions_clean)},
            {"metric": "books_columns", "value": books_raw.shape[1]},
            {"metric": "interactions_columns", "value": interactions_raw.shape[1]},
            {
                "metric": "explicit_rating_pct",
                "value": float(interactions_clean["rating_clean"].notna().mean())
                if "rating_clean" in interactions_clean.columns and len(interactions_clean)
                else 0.0,
            },
        ]
    )
    artifacts["summary"] = summary
    return artifacts, summary
