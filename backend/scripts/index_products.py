"""
One-time script to load Amazon product data into Elasticsearch.

Usage (from inside the backend container):
    python scripts/index_products.py

Or from the host:
    docker compose exec backend python scripts/index_products.py

Loads 4 Amazon categories (~2k products), generates BGE embeddings,
and bulk-indexes into the 'products' ES index.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import get_es, init_es_index, es_bulk_index, ES_INDEX, EMBEDDING_MODEL, EMBEDDING_DIM
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

CATEGORIES = [
    "Sports_and_Outdoors",
    "Electronics",
    "Home_and_Kitchen",
    "Toys_and_Games",
]
PRODUCTS_PER_CATEGORY = 500
BATCH_SIZE = 256

CATEGORY_DISPLAY = {
    "Sports_and_Outdoors": "Sports & Outdoors",
    "Electronics": "Electronics",
    "Home_and_Kitchen": "Home & Kitchen",
    "Toys_and_Games": "Toys & Games",
}


def load_amazon_products() -> list[dict]:
    """Stream Amazon product metadata for selected categories."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    all_docs = []

    for category in CATEGORIES:
        print(f"\nLoading {category}...")
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )

        products = []
        for item in dataset:
            title = (item.get("title") or "").strip()
            description = " ".join(item.get("description") or []).strip()
            price_raw = item.get("price")

            if not title or not description:
                continue

            try:
                price = float(str(price_raw).replace("$", "").replace(",", "")) if price_raw else None
            except ValueError:
                price = None

            if price is None or price <= 0:
                continue

            products.append({
                "id": item["parent_asin"],
                "name": title,
                "description": description[:500],
                "category": CATEGORY_DISPLAY[category],
                "price": round(price, 2),
                "original_price": None,
                "rating": float(item.get("average_rating") or 0),
                "reviews": int(item.get("rating_number") or 0),
                "image": next(iter((item.get("images") or {}).get("large", [])), ""),
            })

            if len(products) >= PRODUCTS_PER_CATEGORY:
                break

        print(f"  Loaded {len(products)} products")

        # Generate embeddings in batches
        print(f"  Generating embeddings...")
        texts = [
            f"Represent this product for retrieval: {p['name']}. {p['description']}"
            for p in products
        ]
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False).tolist()
            for j, emb in enumerate(embeddings):
                products[i + j]["embedding"] = emb

        all_docs.extend(products)

    return all_docs


def main():
    es = get_es()
    if not es.ping():
        print("ERROR: Cannot connect to Elasticsearch at", os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
        sys.exit(1)

    # Wipe and recreate index for a clean run
    if es.indices.exists(index=ES_INDEX):
        print(f"Deleting existing '{ES_INDEX}' index...")
        es.indices.delete(index=ES_INDEX)

    init_es_index()
    print(f"Created '{ES_INDEX}' index.")

    docs = load_amazon_products()
    print(f"\nIndexing {len(docs)} total products into Elasticsearch...")
    es_bulk_index(docs)

    count = es.count(index=ES_INDEX)["count"]
    print(f"Done. {count} products indexed.")


if __name__ == "__main__":
    main()
