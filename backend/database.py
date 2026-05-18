import sqlite3
import os
from typing import List, Dict, Any

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

