from __future__ import annotations

import pandas as pd

from src.utils.cleaning import clean_books, clean_interactions, normalize_review_text


def test_normalize_review_text_removes_html_and_collapses_spaces() -> None:
    assert normalize_review_text("A<br />  great&nbsp;book") == "A great book"


def test_clean_interactions_creates_rating_missing_and_clean_rating() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "book_id": ["b1", "b2"],
            "review_id": ["r1", "r2"],
            "rating": [0, 5],
            "is_read": [False, True],
            "review_text_incomplete": ["", "<b>Nice</b>"],
            "date_added": ["Fri Sep 08 10:44:24 -0700 2017", ""],
            "date_updated": ["Fri Sep 08 10:44:24 -0700 2017", ""],
            "read_at": ["", ""],
            "started_at": ["", ""],
        }
    )
    cleaned = clean_interactions(df)
    assert cleaned.loc[cleaned["review_id"].eq("r1"), "rating_missing"].item() is True
    assert pd.isna(cleaned.loc[cleaned["review_id"].eq("r1"), "rating_clean"].item())
    assert cleaned.loc[cleaned["review_id"].eq("r2"), "rating_clean"].item() == 5
    assert cleaned.loc[cleaned["review_id"].eq("r2"), "review_text_clean"].item() == "Nice"


def test_clean_interactions_keeps_latest_review_id() -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "book_id": ["b1", "b1"],
            "review_id": ["r1", "r1"],
            "rating": [3, 4],
            "is_read": [True, True],
            "review_text_incomplete": ["old", "new"],
            "date_added": ["Fri Sep 08 10:44:24 -0700 2017", "Fri Sep 08 10:44:24 -0700 2017"],
            "date_updated": ["Fri Sep 08 10:44:24 -0700 2017", "Sat Sep 09 10:44:24 -0700 2017"],
            "read_at": ["", ""],
            "started_at": ["", ""],
        }
    )
    cleaned = clean_interactions(df)
    assert len(cleaned) == 1
    assert cleaned["rating_clean"].item() == 4
    assert cleaned["review_text_clean"].item() == "new"


def test_clean_books_flattens_authors_and_deduplicates_book_id() -> None:
    df = pd.DataFrame(
        {
            "book_id": ["b1", "b1"],
            "work_id": ["w1", "w1"],
            "authors": [[{"author_id": "a1", "role": ""}], [{"author_id": "a2", "role": "Editor"}]],
            "popular_shelves": [[{"name": "to-read", "count": "10"}], []],
            "series": [[], []],
            "similar_books": [[], []],
            "is_ebook": ["false", "true"],
            "average_rating": ["4.0", "4.1"],
            "ratings_count": ["1", "10"],
            "text_reviews_count": ["1", "2"],
            "num_pages": ["100", "110"],
            "publication_year": ["2000", "2001"],
            "publication_month": ["1", "2"],
            "publication_day": ["1", "3"],
        }
    )
    cleaned = clean_books(df)
    assert len(cleaned) == 1
    assert cleaned["primary_author_id"].item() == "a2"
    assert cleaned["ratings_count"].item() == 10
