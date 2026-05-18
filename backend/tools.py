from langchain.tools import Tool
from database import query_products, get_categories
import json


def query_products_impl(filters_str: str) -> str:
    """
    Query products from the catalog using optional filters.

    Input: JSON string with any combination of:
    - search: product name keyword (e.g., "headphones")
    - category: category name (e.g., "Electronics")
    - price_min: minimum price (e.g., 10)
    - price_max: maximum price (e.g., 100)
    - rating_min: minimum rating out of 5 (e.g., 4.0)

    Examples:
    - {"search": "headphones"}
    - {"category": "Electronics", "price_max": 100}
    - {"category": "Books", "rating_min": 4.5}
    - {} to get all products
    """
    try:
        filters = json.loads(filters_str) if filters_str.strip() else {}
        results = query_products(filters)

        if not results:
            return json.dumps({"results": [], "message": "No products found matching the criteria"})

        formatted_results = []
        for p in results:
            # we format the price and rating nicely for display purposes, and include the number of reviews to help customers make informed decisions
            formatted_results.append({
                "id": p["id"],
                "name": p["name"],
                "price": f"${p['price']:.2f}",
                "original_price": f"${p['originalPrice']:.2f}" if p["originalPrice"] else None,
                "rating": f"{p['rating']:.1f}/5",
                "reviews": p["reviews"],
                "category": p["category_name"],
            })

        return json.dumps({
            "results": formatted_results,
            "count": len(formatted_results),
        })
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input", "results": []})
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})


def list_categories_impl(_: str = "") -> str:
    """Get all available product categories."""
    try:
        categories = get_categories()
        return json.dumps({
            "categories": [{"name": c["name"], "icon": c["icon"]} for c in categories],
        })
    except Exception as e:
        return json.dumps({"error": str(e), "categories": []})


PRODUCT_TOOLS = [
    Tool(
        name="query_products",
        func=query_products_impl,
        description=(
            "Search and filter products from the catalog. "
            "Input: JSON with optional filters: search (keyword), category (name), "
            "price_min, price_max, rating_min. "
            'Examples: {"search": "headphones"} or {"category": "Electronics", "price_max": 100} or {} for all products.'
        ),
    ),
    Tool(
        name="list_categories",
        func=list_categories_impl,
        description="List all available product categories. Pass an empty string as input.",
    ),
]
