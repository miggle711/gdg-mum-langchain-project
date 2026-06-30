import os
import json
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gdg:gdg@localhost:5432/ecommerce")


def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            icon TEXT NOT NULL,
            colorClass TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            originalPrice REAL,
            rating REAL NOT NULL,
            reviews INTEGER NOT NULL,
            image TEXT NOT NULL,
            category_id TEXT NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )
    """)

    conn.commit()
    conn.close()


def seed_db():
    """Seed the database with sample data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM categories")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return

    categories = [
        ("cat-1", "Electronics", "🖥️", "bg-blue-100"),
        ("cat-2", "Clothing", "👕", "bg-purple-100"),
        ("cat-3", "Home & Garden", "🏠", "bg-green-100"),
        ("cat-4", "Sports", "⚽", "bg-orange-100"),
        ("cat-5", "Books", "📚", "bg-yellow-100"),
    ]
    cursor.executemany(
        "INSERT INTO categories (id, name, icon, colorClass) VALUES (%s, %s, %s, %s)",
        categories
    )

    products = [
        # Electronics
        ("prod-1", "Wireless Headphones", 49.99, 79.99, 4.5, 128, "https://via.placeholder.com/300?text=Headphones", "cat-1"),
        ("prod-2", "USB-C Hub", 29.99, None, 4.2, 45, "https://via.placeholder.com/300?text=USB+Hub", "cat-1"),
        ("prod-3", "Portable Speaker", 89.99, 119.99, 4.7, 203, "https://via.placeholder.com/300?text=Speaker", "cat-1"),
        ("prod-4", "Phone Stand", 15.99, None, 4.1, 78, "https://via.placeholder.com/300?text=Phone+Stand", "cat-1"),
        ("prod-5", "Mechanical Keyboard", 79.99, 129.99, 4.6, 312, "https://via.placeholder.com/300?text=Keyboard", "cat-1"),

        # Clothing
        ("prod-6", "Cotton T-Shirt", 19.99, None, 4.3, 89, "https://via.placeholder.com/300?text=T-Shirt", "cat-2"),
        ("prod-7", "Denim Jeans", 59.99, 89.99, 4.4, 156, "https://via.placeholder.com/300?text=Jeans", "cat-2"),
        ("prod-8", "Running Shoes", 89.99, 129.99, 4.5, 234, "https://via.placeholder.com/300?text=Shoes", "cat-2"),
        ("prod-9", "Winter Jacket", 129.99, 179.99, 4.6, 178, "https://via.placeholder.com/300?text=Jacket", "cat-2"),

        # Home & Garden
        ("prod-10", "Plant Pot (Small)", 12.99, None, 4.2, 45, "https://via.placeholder.com/300?text=Pot", "cat-3"),
        ("prod-11", "Desk Lamp", 34.99, 49.99, 4.4, 92, "https://via.placeholder.com/300?text=Lamp", "cat-3"),
        ("prod-12", "Throw Pillow", 24.99, None, 4.1, 67, "https://via.placeholder.com/300?text=Pillow", "cat-3"),

        # Sports
        ("prod-13", "Yoga Mat", 22.99, None, 4.3, 145, "https://via.placeholder.com/300?text=Yoga+Mat", "cat-4"),
        ("prod-14", "Dumbbells (Set)", 49.99, 79.99, 4.5, 201, "https://via.placeholder.com/300?text=Dumbbells", "cat-4"),
        ("prod-15", "Resistance Bands", 16.99, None, 4.2, 123, "https://via.placeholder.com/300?text=Bands", "cat-4"),

        # Books
        ("prod-16", "Python Programming", 39.99, None, 4.6, 89, "https://via.placeholder.com/300?text=Python+Book", "cat-5"),
        ("prod-17", "The Great Gatsby", 12.99, None, 4.7, 234, "https://via.placeholder.com/300?text=Gatsby", "cat-5"),
        ("prod-18", "Atomic Habits", 18.99, None, 4.8, 456, "https://via.placeholder.com/300?text=Atomic+Habits", "cat-5"),
    ]
    cursor.executemany(
        "INSERT INTO products (id, name, price, originalPrice, rating, reviews, image, category_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        products
    )

    conn.commit()
    conn.close()


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
