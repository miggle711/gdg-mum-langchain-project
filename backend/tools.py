from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from database import query_products, get_categories
from typing import Optional
import json


class QueryProductsInput(BaseModel):
    search: Optional[str] = Field(None, description="Keyword to search in product names")
    category: Optional[str] = Field(None, description="Category name, e.g. 'Electronics'")
    price_min: Optional[float] = Field(None, description="Minimum price")
    price_max: Optional[float] = Field(None, description="Maximum price")
    rating_min: Optional[float] = Field(None, description="Minimum rating out of 5")


def query_products_impl(
    search: Optional[str] = None,
    category: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    rating_min: Optional[float] = None,
) -> str:
    filters = {}
    if search is not None:
        filters["search"] = search
    if category is not None:
        filters["category"] = category
    if price_min is not None:
        filters["price_min"] = price_min
    if price_max is not None:
        filters["price_max"] = price_max
    if rating_min is not None:
        filters["rating_min"] = rating_min

    try:
        results = query_products(filters)

        if not results:
            return json.dumps({"results": [], "message": "No products found matching the criteria"})

        formatted_results = []
        for p in results:
            formatted_results.append({
                "id": p["id"],
                "name": p["name"],
                "price": f"${p['price']:.2f}",
                "original_price": f"${p['originalPrice']:.2f}" if p["originalPrice"] else None,
                "rating": f"{p['rating']:.1f}/5",
                "reviews": p["reviews"],
                "category": p["category_name"],
            })

        return json.dumps({"results": formatted_results, "count": len(formatted_results)})
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})


def list_categories_impl() -> str:
    try:
        categories = get_categories()
        return json.dumps({
            "categories": [{"name": c["name"], "icon": c["icon"]} for c in categories],
        })
    except Exception as e:
        return json.dumps({"error": str(e), "categories": []})


PRODUCT_TOOLS = [
    StructuredTool(
        name="query_products",
        func=query_products_impl,
        args_schema=QueryProductsInput,
        description="Search and filter products from the catalog by keyword, category, price range, or minimum rating.",
    ),
    StructuredTool.from_function(
        func=list_categories_impl,
        name="list_categories",
        description="List all available product categories.",
    ),
]
