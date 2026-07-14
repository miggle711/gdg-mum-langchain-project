from search import query_products

all_test_filters = [
    {"category": "Electronics", "price_max": 20},
    {"category": "Electronics", "price_min": 50, "price_max": 100},
    {"category": "Electronics", "rating_min": 4.5},
    {"category": "Home & Kitchen", "price_max": 15},
    {"category": "Home & Kitchen", "price_min": 30},
    {"category": "Sports & Outdoors", "price_max": 15},
    {"category": "Sports & Outdoors", "rating_min": 4.5, "price_min": 50},
    {"category": "Toys & Games", "price_max": 10},
    {"category": "Toys & Games", "price_min": 50},
    {"category": "Electronics", "price_max": 3},       # candidate no-match edge case
    {"category": "Toys & Games", "price_min": 500},    # candidate no-match edge case
    {"search": "JBL"},
    {"search": "camera"},
    {"search": "backyard"},
]

test_filters = [
    {"category": "Electronics", "price_min": 5000},
    {"category": "Toys & Games", "price_max": 0.50},
    {"search": "unicorn horn costume"},
    {"search": "xyzxyz_nonexistent_product_query"},
]

for f in test_filters:
    results = query_products(f)
    print(f"\nFilter: {f}")
    print(f"  {len(results)} result(s)")
    for p in results[:3]:
        print(f"    {p['name']} | ${p['price']} | rating {p['rating']} | {p['reviews']} reviews")