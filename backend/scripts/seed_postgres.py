"""
One-time script to seed Postgres with Phase 1 ecommerce data.

Usage (from inside the backend container):
    python scripts/seed_postgres.py

Or from the host:
    docker compose exec backend python scripts/seed_postgres.py

Sources:
  - products, product_images: McAuley-Lab/Amazon-Reviews-2023 raw_meta_{category},
    4 categories, up to 500 products/category
  - reviews: McAuley-Lab/Amazon-Reviews-2023 raw_review_{category}, joined
    to products via parent_asin
  - users, addresses: synthetic, generated with Faker (no real user identities exist
    in the source dataset — review user_id values are opaque hashes, not usable as
    real user records)

Run this before scripts/seed_elasticsearch.py, which reads products/reviews
back out of Postgres to embed and index into Elasticsearch (#51) — this
script has no Elasticsearch dependency of its own.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from faker import Faker
from sqlalchemy import func, select, text

from db import Base, get_engine, get_session
from models_db import Address, Product, ProductImage, Review, User

CATEGORIES = [
    "Sports_and_Outdoors",
    "Electronics",
    "Home_and_Kitchen",
    "Toys_and_Games",
]
PRODUCTS_PER_CATEGORY = 500
REVIEWS_PER_CATEGORY = 50 * len(CATEGORIES)  # cap, not per-product — keeps seed volume reasonable
N_USERS = 50

CATEGORY_DISPLAY = {
    "Sports_and_Outdoors": "Sports & Outdoors",
    "Electronics": "Electronics",
    "Home_and_Kitchen": "Home & Kitchen",
    "Toys_and_Games": "Toys & Games",
}


def _parse_amazon_product(item: dict, category: str) -> dict | None:
    """Filters and shapes a raw Amazon product record for insertion into
    Product. Product.id (= parent_asin) is what scripts/seed_elasticsearch.py
    later uses as the ES document _id when it reads these rows back out —
    since that script derives IDs directly from Postgres rather than
    re-parsing the dataset independently, there's no second copy of this
    filtering logic to keep in sync with anymore (#51).
    """
    title = (item.get("title") or "").strip()
    description = " ".join(item.get("description") or []).strip()
    price_raw = item.get("price")

    if not title or not description:
        return None

    try:
        price = float(str(price_raw).replace("$", "").replace(",", "")) if price_raw else None
    except ValueError:
        price = None

    if price is None or price <= 0:
        return None

    return {
        "id": item["parent_asin"],
        "name": title,
        "description": description[:500],
        "category": CATEGORY_DISPLAY[category],
        "price": round(price, 2),
        "original_price": None,
        "rating": float(item.get("average_rating") or 0),
        "reviews": int(item.get("rating_number") or 0),
        "image": next(iter((item.get("images") or {}).get("large", [])), ""),
    }


def load_products_and_images() -> tuple[list[dict], list[dict]]:
    """Streams product metadata, returning (product_rows, image_rows).
    Unlike scripts/seed_elasticsearch.py, this does NOT generate embeddings —
    Postgres doesn't need vectors — and extracts every image URL (not just
    the first) into ordered product_images rows."""
    product_rows: list[dict] = []
    image_rows: list[dict] = []

    for category in CATEGORIES:
        print(f"\nLoading {category}...")
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        image_count = 0
        for item in dataset:
            product = _parse_amazon_product(item, category)
            if product is None:
                continue

            product_rows.append(product)

            urls = (item.get("images") or {}).get("large") or []
            if not urls:
                urls = (item.get("images") or {}).get("hi_res") or (item.get("images") or {}).get("thumb") or []
            for position, url in enumerate(urls):
                if url:
                    image_rows.append({"product_id": product["id"], "image_url": url, "position": position})
                    image_count += 1

            count += 1
            if count >= PRODUCTS_PER_CATEGORY:
                break

        print(f"  Loaded {count} products, {image_count} images")

    return product_rows, image_rows


def load_reviews(product_ids: set[str]) -> list[dict]:
    """Streams review data, filtered to products actually seeded (avoids
    orphaned FK rows), capped per category to keep seed volume reasonable —
    full review files are much larger than the metadata files."""
    review_rows: list[dict] = []

    for category in CATEGORIES:
        print(f"\nLoading reviews for {category}...")
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        for item in dataset:
            parent_asin = item.get("parent_asin")
            if parent_asin not in product_ids:
                continue

            review_rows.append({
                "product_id": parent_asin,
                "rating": float(item.get("rating") or 0),
                "title": item.get("title"),
                "text": item.get("text"),
                "verified_purchase": bool(item.get("verified_purchase", False)),
                "helpful_vote": int(item.get("helpful_vote") or 0),
                "timestamp": item.get("timestamp"),
            })

            count += 1
            if count >= REVIEWS_PER_CATEGORY // len(CATEGORIES):
                break

        print(f"  Loaded {count} reviews")

    return review_rows


def build_synthetic_users_and_addresses(n_users: int = N_USERS) -> tuple[list[dict], list[dict]]:
    """Synthetic (Faker-generated) — no real user dataset exists; review
    user_id values in the source dataset are opaque anonymized hashes, not
    usable as real user records."""
    fake = Faker("en_US")
    Faker.seed(0)

    user_rows: list[dict] = []
    address_rows: list[dict] = []

    for i in range(n_users):
        user_rows.append({
            "id": i + 1,
            "email": fake.unique.email(),
            "name": fake.name(),
        })

        for _ in range(fake.random_int(min=1, max=3)):
            address_rows.append({
                "user_id": i + 1,
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zip_code": fake.zipcode(),
                "country": "US",
            })

    return user_rows, address_rows


async def main():
    engine = get_engine()

    try:
        async with engine.connect():
            pass
    except Exception as e:
        print(f"ERROR: Cannot connect to Postgres: {e}")
        sys.exit(1)

    # Wipe and recreate schema for a clean run
    print("Dropping and recreating Postgres schema for a clean run...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("Schema recreated.")

    product_rows, image_rows = load_products_and_images()
    product_ids = {p["id"] for p in product_rows}
    review_rows = load_reviews(product_ids)
    user_rows, address_rows = build_synthetic_users_and_addresses()

    async with get_session() as session:
        print(f"\nInserting {len(product_rows)} products...")
        session.add_all(Product(**row) for row in product_rows)
        await session.commit()

        print(f"Inserting {len(image_rows)} product images...")
        session.add_all(ProductImage(**row) for row in image_rows)
        await session.commit()

        print(f"Inserting {len(review_rows)} reviews...")
        session.add_all(Review(**row) for row in review_rows)
        await session.commit()

        print(f"Inserting {len(user_rows)} users and {len(address_rows)} addresses...")
        session.add_all(User(**row) for row in user_rows)
        await session.commit()
        session.add_all(Address(**row) for row in address_rows)
        await session.commit()

        # User.id is assigned explicitly above (not autoincrement-generated),
        # which leaves the users_id_seq sequence at its default starting
        # value. Resync it to the actual max id so later autoincrement
        # inserts (e.g. shadow users created via session_identity.py) don't
        # collide with these seeded rows.
        await session.execute(
            text("SELECT setval('users_id_seq', (SELECT MAX(id) FROM users))")
        )
        await session.commit()

        print("\nDone. Row counts:")
        for model in (Product, ProductImage, User, Address, Review):
            count = await session.scalar(select(func.count()).select_from(model))
            print(f"  {model.__tablename__}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
