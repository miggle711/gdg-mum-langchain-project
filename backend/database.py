import os
import json
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gdg:gdg@localhost:5432/ecommerce")

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768


def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            icon TEXT NOT NULL,
            colorClass TEXT NOT NULL
        )
    """)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            originalPrice REAL,
            rating REAL NOT NULL,
            reviews INTEGER NOT NULL,
            image TEXT NOT NULL,
            category_id TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )
    """)

    # Note: IVFFlat index needs many more rows than we have in dev.
    # For production with 10k+ products, add:
    # CREATE INDEX ON products USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    # For now, pgvector falls back to exact sequential scan which is fine at this scale.

    conn.commit()
    conn.close()


def seed_db():
    """Fetch products from DummyJSON and seed the database with embeddings."""
    import urllib.request
    from sentence_transformers import SentenceTransformer

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM categories")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return

    # Fetch all 194 products from DummyJSON
    req = urllib.request.Request(
        "https://dummyjson.com/products?limit=194",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    raw_products = data["products"]

    # Build categories from unique category slugs
    seen_categories = {}
    category_icons = {
        "beauty": "💄", "fragrances": "🌸", "furniture": "🛋️",
        "groceries": "🛒", "home-decoration": "🏠", "kitchen-accessories": "🍳",
        "laptops": "💻", "mens-shirts": "👔", "mens-shoes": "👟",
        "mens-watches": "⌚", "mobile-accessories": "📱", "motorcycle": "🏍️",
        "skin-care": "🧴", "smartphones": "📱", "sports-accessories": "⚽",
        "sunglasses": "🕶️", "tablets": "📱", "tops": "👕",
        "vehicle": "🚗", "womens-bags": "👜", "womens-dresses": "👗",
        "womens-jewellery": "💍", "womens-shoes": "👠", "womens-watches": "⌚",
    }
    color_classes = [
        "bg-blue-100", "bg-purple-100", "bg-green-100", "bg-orange-100",
        "bg-yellow-100", "bg-pink-100", "bg-red-100", "bg-indigo-100",
    ]
    for p in raw_products:
        slug = p["category"]
        if slug not in seen_categories:
            seen_categories[slug] = {
                "id": f"cat-{slug}",
                "name": p["category"].replace("-", " ").title(),
                "icon": category_icons.get(slug, "📦"),
                "colorClass": color_classes[len(seen_categories) % len(color_classes)],
            }

    cursor.executemany(
        "INSERT INTO categories (id, name, icon, colorClass) VALUES (%s, %s, %s, %s)",
        [(c["id"], c["name"], c["icon"], c["colorClass"]) for c in seen_categories.values()]
    )

    # Generate embeddings using title + description for richer semantic representation
    print(f"Generating embeddings for {len(raw_products)} products...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [
        f"Represent this product for retrieval: {p['title']}. {p['description']}"
        for p in raw_products
    ]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    for p, embedding in zip(raw_products, embeddings):
        original_price = round(p["price"] / (1 - p["discountPercentage"] / 100), 2) if p.get("discountPercentage") else None
        cursor.execute(
            "INSERT INTO products (id, name, price, originalPrice, rating, reviews, image, category_id, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                f"prod-{p['id']}",
                p["title"],
                p["price"],
                original_price,
                p["rating"],
                len(p.get("reviews", [])),
                p.get("thumbnail", ""),
                f"cat-{p['category']}",
                embedding,
            )
        )

    conn.commit()
    conn.close()
    print("Seeding complete.")


def query_products(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    query = """
        SELECT
            p.id, p.name, p.price, p.originalPrice, p.rating, p.reviews,
            c.name as category_name
        FROM products p
        JOIN categories c ON p.category_id = c.id
        WHERE 1=1
    """
    # where 1=1 is a common SQL trick to simplify appending additional conditions
    params = []

    if "category" in filters:
        query += " AND (c.name = %s OR c.id = %s)"
        params.extend([filters["category"], filters["category"]])

    if "price_max" in filters:
        query += " AND p.price <= %s"
        params.append(filters["price_max"])

    if "price_min" in filters:
        query += " AND p.price >= %s"
        params.append(filters["price_min"])

    if "rating_min" in filters:
        query += " AND p.rating >= %s"
        params.append(filters["rating_min"])

    if "search" in filters:
        query += " AND p.name ILIKE %s"
        params.append(f"%{filters['search']}%")

    query += " ORDER BY p.rating DESC, p.reviews DESC LIMIT 20"

    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return results


def semantic_search(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Find products most similar to the query embedding using cosine distance."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT
            p.id, p.name, p.price, p.originalprice, p.rating, p.reviews,
            c.name as category_name,
            1 - (p.embedding <=> %s::vector) as similarity
        FROM products p
        JOIN categories c ON p.category_id = c.id
        WHERE p.embedding IS NOT NULL
        ORDER BY p.embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, query_embedding, limit))

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_categories() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT id, name, icon FROM categories ORDER BY name")
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


# Redis connection pool — created once, reused across all requests
_redis_pool = redis.ConnectionPool.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True,
    max_connections=20,
)

CONVERSATION_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def _get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=_redis_pool)


def save_messages(conversation_id: str, messages: List[BaseMessage]) -> None:
    """Serialize and save a conversation's messages to Redis."""
    r = _get_redis()
    key = f"conversation:{conversation_id}"
    r.set(key, json.dumps(messages_to_dict(messages)), ex=CONVERSATION_TTL_SECONDS)


def load_messages(conversation_id: str) -> List[BaseMessage]:
    """Load and deserialize a conversation's messages from Redis. Returns [] if not found."""
    r = _get_redis()
    key = f"conversation:{conversation_id}"
    data = r.get(key)
    if data is None:
        return []
    return messages_from_dict(json.loads(data))
