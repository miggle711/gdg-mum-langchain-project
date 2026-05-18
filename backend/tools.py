from langchain.tools import Tool
from database import query_products, get_categories
import json

def search_products_impl(query_str: str) -> str:
    """
    Search for products based on natural language query.
    The agent can use this to find products matching user requests.
    """
    # For now, a simple implementation that searches by name
    # In a production system, you'd parse the query more intelligently
    try:
        results = query_products({"search": query_str})
        if not results:
            return json.dumps({"results": [], "message": f"No products found matching '{query_str}'"})

        formatted_results = []
        for p in results:
            formatted_results.append({
                "id": p["id"],
                "name": p["name"],
                "price": p["price"],
                "original_price": p["originalPrice"],
                "rating": p["rating"],
                "reviews": p["reviews"],
                "category": p["category_name"]
            })

        return json.dumps({
            "results": formatted_results,
            "count": len(formatted_results),
            "message": f"Found {len(formatted_results)} products"
        })
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})

def filter_products_impl(filters_str: str) -> str:
    """
    Filter products by category, price range, minimum rating, or search term.

    Input should be JSON with any of these filters:
    - category: "Electronics" or category name
    - price_min: minimum price (e.g., 10)
    - price_max: maximum price (e.g., 100)
    - rating_min: minimum rating (e.g., 4.0)
    - search: product name search term

    Example: {"category": "Electronics", "price_max": 100, "rating_min": 4.0}
    """
    try:
        filters = json.loads(filters_str)
        results = query_products(filters)

        if not results:
            return json.dumps({"results": [], "message": "No products match the filters"})

        formatted_results = []
        for p in results:
            formatted_results.append({
                "id": p["id"],
                "name": p["name"],
                "price": f"${p['price']:.2f}",
                "original_price": f"${p['originalPrice']:.2f}" if p["originalPrice"] else "N/A",
                "rating": f"{p['rating']:.1f}/5",
                "reviews": p["reviews"],
                "category": p["category_name"]
            })

        return json.dumps({
            "results": formatted_results,
            "count": len(formatted_results),
            "message": f"Found {len(formatted_results)} products matching filters"
        })
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format for filters", "results": []})
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})

def list_categories_impl() -> str:
    """Get all available product categories."""
    try:
        categories = get_categories()
        formatted = [{"name": c["name"], "icon": c["icon"]} for c in categories]
        return json.dumps({
            "categories": formatted,
            "count": len(formatted)
        })
    except Exception as e:
        return json.dumps({"error": str(e), "categories": []})

# Define tools for LangChain
search_products_tool = Tool(
    name="search_products",
    func=search_products_impl,
    description="Search for products by name. Input: search term (e.g., 'headphones', 'shoes'). Returns matching products with prices, ratings, and reviews."
)

filter_products_tool = Tool(
    name="filter_products",
    func=filter_products_impl,
    description="Filter products by category, price range, or minimum rating. Input: JSON with filters like {\"category\": \"Electronics\", \"price_max\": 100}. Returns products matching all filters."
)

list_categories_tool = Tool(
    name="list_categories",
    func=list_categories_impl,
    description="List all available product categories. Returns category names and icons."
)

# All available tools
PRODUCT_TOOLS = [search_products_tool, filter_products_tool, list_categories_tool]
