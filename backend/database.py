import sqlite3
import os
import json
import redis
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

# Database file path (can be set via environment variable)
DB_PATH = os.getenv("DB_PATH", "products.db")

def get_db_connection():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Enable dict-like access to rows
    return conn

def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()

    # the cursor is used to execute SQL commands
    cursor = conn.cursor() 


    # Create tables according to the backend/docs/db-models.md specification
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            icon TEXT NOT NULL,
            colorClass TEXT NOT NULL
        )
    """)

    # Create products table
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

    # Commit changes and close the connection
    conn.commit()
    conn.close()

def seed_db():
    """Seed the database with sample data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM categories")
    if cursor.fetchone()[0] > 0:
        # Data already exists don't seed again
        conn.close()
        return

    # Insert categories
    categories = [
        ("cat-1", "Electronics", "🖥️", "bg-blue-100"),
        ("cat-2", "Clothing", "👕", "bg-purple-100"),
        ("cat-3", "Home & Garden", "🏠", "bg-green-100"),
        ("cat-4", "Sports", "⚽", "bg-orange-100"),
        ("cat-5", "Books", "📚", "bg-yellow-100"),
    ]
    # executemany allows us to insert multiple rows in one command
    cursor.executemany(
        "INSERT INTO categories (id, name, icon, colorClass) VALUES (?, ?, ?, ?)",
        categories
    )

    # Insert products
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
        "INSERT INTO products (id, name, price, originalPrice, rating, reviews, image, category_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        products
    )

    conn.commit()
    conn.close()


def query_products(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Query products based on filters.

    Supported filters:
    - category: category name or ID
    - price_max: maximum price
    - price_min: minimum price
    - rating_min: minimum rating
    - search: search term in product name

    Args:

        filters (Dict[str, Any]): Dictionary of optional filter criteria.

        Example:

            filters = {

                "category": "Electronics",   # category name or ID

                "price_min": 100,

                "price_max": 500,

                "rating_min": 4.5,

                "search": "Laptop"

            }

    Returns:

        List[Dict[str, Any]]: List of matching products with category info.

        Example:

            [

                {

                    "id": "prod-1",

                    "name": "Wireless Headphones",

                    "price": 49.99,

                    "originalPrice": 79.99,

                    "rating": 4.5,

                    "reviews": 128,

                    "category_name": "Electronics"

                }

            ]
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT
            p.id, p.name, p.price, p.originalPrice, p.rating, p.reviews,
            c.name as category_name
        FROM products p
        JOIN categories c ON p.category_id = c.id
        WHERE 1=1 
    """
    # where 1=1 is a common SQL trick to simplify appending additional conditions
    # we can simply add "AND ..." without worrying about whether it's the first condition or not
    # if there are no filters, it will just return all products no sweat
    params = [] # we're using parameterized queries to prevent SQL injection and handle user input safely

    if "category" in filters:
        query += " AND (c.name = ? OR c.id = ?)"
        # we allow filtering by either category name or ID for flexibility
        params.extend([filters["category"], filters["category"]])

    if "price_max" in filters:
        query += " AND p.price <= ?"
        # check for all products with price less than or equal to the specified max price
        params.append(filters["price_max"])

    if "price_min" in filters:
        query += " AND p.price >= ?"
        # check for all products with price greater than or equal to the specified min price
        params.append(filters["price_min"])

    if "rating_min" in filters:
        query += " AND p.rating >= ?"
        # check for all products with rating greater than or equal to the specified min rating
        params.append(filters["rating_min"])

    if "search" in filters:
        query += " AND p.name LIKE ?"
        # we use LIKE for simple substring search in product names
        params.append(f"%{filters['search']}%")

    # we order results by rating and number of reviews to show the best products first
    query += " ORDER BY p.rating DESC, p.reviews DESC LIMIT 20"

    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return results

def get_categories() -> List[Dict[str, Any]]:
    """Get all product categories."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, icon FROM categories ORDER BY name")
    # fetchall returns a list of rows
    # convert each row to a dict for easier access in the frontend
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
