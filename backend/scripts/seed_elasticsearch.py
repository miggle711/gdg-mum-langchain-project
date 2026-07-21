"""
One-time script to load products and reviews from Postgres into
Elasticsearch. Requires scripts/seed_postgres.py to have been run first —
this script has no HF dataset dependency of its own; Postgres is the
single source of truth for what gets embedded and indexed.

Usage (from inside the backend container):
    python scripts/seed_elasticsearch.py

Or from the host:
    docker compose exec backend python scripts/seed_elasticsearch.py

Wipes and recreates both the 'products' and 'reviews' ES indices for a
clean run (mirrors seed_postgres.py's drop/recreate convention).
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from db import get_session
from models_db import Product, Review
from search import (
    get_es,
    get_embedding_model,
    init_es_index,
    init_reviews_index,
    es_bulk_index,
    es_bulk_index_reviews,
    ES_INDEX,
    REVIEWS_ES_INDEX,
)

BATCH_SIZE = 256


async def load_products() -> list[dict]:
    async with get_session() as session:
        result = await session.execute(select(Product))
        products = result.scalars().all()

    docs = []
    for p in products:
        docs.append({
            "id": p.id,
            "name": p.name,
            "description": p.description or "",
            "category": p.category,
            "price": p.price,
            "original_price": p.original_price,
            "rating": p.rating,
            "reviews": p.reviews or 0,
            "image": p.image or "",
        })

    print(f"Generating embeddings for {len(docs)} products...")
    model = get_embedding_model()
    texts = [f"Represent this product for retrieval: {d['name']}. {d['description']}" for d in docs]
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False).tolist()
        for j, emb in enumerate(embeddings):
            docs[i + j]["embedding"] = emb
    return docs


async def load_reviews() -> list[dict]:
    async with get_session() as session:
        result = await session.execute(
            select(Review, Product.name).join(Product, Review.product_id == Product.id)
        )
        rows = result.all()

    docs = []
    for r, product_name in rows:
        docs.append({
            "id": str(r.id),
            "product_id": r.product_id,
            "product_name": product_name,
            "rating": r.rating,
            "title": r.title or "",
            "text": r.text or "",
            "verified_purchase": r.verified_purchase,
            "helpful_vote": r.helpful_vote,
        })

    print(f"Generating embeddings for {len(docs)} reviews...")
    model = get_embedding_model()
    texts = [f"Represent this review for retrieval: {d['title']}. {d['text']}" for d in docs]
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False).tolist()
        for j, emb in enumerate(embeddings):
            docs[i + j]["embedding"] = emb
    return docs


async def main():
    es = get_es()
    if not await es.ping():
        print("ERROR: Cannot connect to Elasticsearch at", os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
        sys.exit(1)

    for index in (ES_INDEX, REVIEWS_ES_INDEX):
        if await es.indices.exists(index=index):
            print(f"Deleting existing '{index}' index...")
            await es.indices.delete(index=index)

    await init_es_index()
    await init_reviews_index()
    print(f"Created '{ES_INDEX}' and '{REVIEWS_ES_INDEX}' indices.")

    product_docs = await load_products()
    print(f"\nIndexing {len(product_docs)} products into Elasticsearch...")
    await es_bulk_index(product_docs)

    review_docs = await load_reviews()
    print(f"\nIndexing {len(review_docs)} reviews into Elasticsearch...")
    await es_bulk_index_reviews(review_docs)

    print("\nDone.")
    print(f"  {ES_INDEX}: {(await es.count(index=ES_INDEX))['count']} documents")
    print(f"  {REVIEWS_ES_INDEX}: {(await es.count(index=REVIEWS_ES_INDEX))['count']} documents")

    await es.close()


if __name__ == "__main__":
    asyncio.run(main())
