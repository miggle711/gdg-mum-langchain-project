from search import query_products, get_categories

categories = get_categories()
print("Actual indexed categories:")
for c in categories:
    print(f"  {c['name']}")

print("\nSample products per category:")
for c in categories:
    results = query_products({"category": c["name"]})
    print(f"\n-- {c['name']} ({len(results)} shown, may be capped) --")
    for p in results[:5]:
        print(f"  {p['name']} | ${p['price']} | rating {p['rating']} | {p['reviews']} reviews")