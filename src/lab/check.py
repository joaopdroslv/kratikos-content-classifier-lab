"""Smoke check: read-only access to Postgres and Qdrant. Prints counts only, never a credential.

uv run python -m lab.check
"""

from sqlalchemy import text

from lab.config import get_engine, get_qdrant


def main() -> None:

    with get_engine().connect() as conn:
        read_only = conn.execute(text("SHOW transaction_read_only")).scalar_one()
        news = conn.execute(
            text("SELECT count(*) FROM public.news_articles")
        ).scalar_one()
        posts = conn.execute(text("SELECT count(*) FROM public.posts")).scalar_one()
    print(f"postgres  read_only={read_only}  news_articles={news}  posts={posts}")

    qdrant = get_qdrant()
    for collection in qdrant.get_collections().collections:
        count = qdrant.count(collection.name, exact=False).count
        print(f"qdrant    collection {collection.name}  points~{count}")
    for alias in qdrant.get_aliases().aliases:
        count = qdrant.count(alias.alias_name, exact=False).count
        print(
            f"qdrant    {alias.alias_name} -> {alias.collection_name}  points~{count}"
        )


if __name__ == "__main__":
    main()
