from __future__ import annotations

import pandas as pd


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

