# Database Setup

The backend uses SQLite for product catalog storage. The database is automatically initialized on startup.

## Schema

### Categories Table
- `id` - UUID (primary key)
- `name` - Category name (unique)
- `icon` - Emoji or icon for the category
- `colorClass` - CSS class for styling

### Products Table
- `id` - UUID (primary key)
- `name` - Product name
- `price` - Current price
- `originalPrice` - Original/list price (optional)
- `rating` - Average rating (0-5)
- `reviews` - Number of reviews
- `image` - Product image URL
- `category_id` - Foreign key to categories table

## Database File

The SQLite database is stored at `products.db` in the backend directory (or the path specified by `DB_PATH` environment variable).

## Initialization

When the backend starts:
1. `init_db()` creates the tables if they don't exist
2. `seed_db()` populates sample data (only runs if tables are empty)

## Sample Data

The database is seeded with sample products across 5 categories:
- Electronics (headphones, keyboard, speaker, USB hub, phone stand)
- Clothing (t-shirts, jeans, shoes, jacket)
- Home & Garden (pots, lamp, pillow)
- Sports (yoga mat, dumbbells, resistance bands)
- Books (Python, Gatsby, Atomic Habits)

## Tools for Product Queries

The agent has access to three tools for querying products:

### search_products(search_term)
Search for products by name.
```
Input: "headphones"
Returns: Products matching "headphones"
```

### filter_products(filters_json)
Filter products by multiple criteria.
```
Input: {"category": "Electronics", "price_max": 100, "rating_min": 4.0}
Returns: Products in Electronics category under $100 with 4+ rating
```

Supported filters:
- `category` - Category name or ID
- `price_min` - Minimum price
- `price_max` - Maximum price
- `rating_min` - Minimum rating (0-5)
- `search` - Product name search term

### list_categories()
Get all available categories.
```
Returns: List of categories with names and icons
```

## Testing

To test database queries without the full chat flow:

```python
from database import query_products, get_categories

# Search products
results = query_products({"search": "headphones"})
print(results)

# Filter products
results = query_products({"category": "Electronics", "price_max": 100})
print(results)

# List categories
categories = get_categories()
print(categories)
```

## Future Improvements

- Migrate to PostgreSQL for production
- Add product inventory tracking
- Implement order history
- Add customer wishlists
- Support for dynamic product updates via API
