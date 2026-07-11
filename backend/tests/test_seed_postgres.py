"""Unit tests for the pure-logic helpers in seed_postgres.py.

Does NOT test load_products_and_images()/load_reviews() against the real
streamed HF dataset — slow, network-dependent, same reason
index_products.py's dataset-loading code has no direct unit test today.
"""

from scripts.seed_postgres import (
    N_USERS,
    _parse_amazon_product,
    build_synthetic_users_and_addresses,
)


def test_parse_amazon_product_valid_item():
    item = {
        "title": "Wireless Mouse",
        "description": ["A great mouse", "with two buttons"],
        "price": "$19.99",
        "parent_asin": "B000TEST01",
        "average_rating": 4.5,
        "rating_number": 100,
        "images": {"large": ["https://example.com/1.jpg"]},
    }

    result = _parse_amazon_product(item, "Electronics")

    assert result == {
        "id": "B000TEST01",
        "name": "Wireless Mouse",
        "description": "A great mouse with two buttons",
        "category": "Electronics",
        "price": 19.99,
        "original_price": None,
        "rating": 4.5,
        "reviews": 100,
        "image": "https://example.com/1.jpg",
    }


def test_parse_amazon_product_missing_title_is_skipped():
    item = {"title": "", "description": ["desc"], "price": "$10", "parent_asin": "x"}
    assert _parse_amazon_product(item, "Electronics") is None


def test_parse_amazon_product_missing_description_is_skipped():
    item = {"title": "Widget", "description": [], "price": "$10", "parent_asin": "x"}
    assert _parse_amazon_product(item, "Electronics") is None


def test_parse_amazon_product_zero_or_missing_price_is_skipped():
    item = {"title": "Widget", "description": ["desc"], "price": None, "parent_asin": "x"}
    assert _parse_amazon_product(item, "Electronics") is None

    item["price"] = "$0.00"
    assert _parse_amazon_product(item, "Electronics") is None


def test_parse_amazon_product_unparseable_price_is_skipped():
    item = {"title": "Widget", "description": ["desc"], "price": "not-a-price", "parent_asin": "x"}
    assert _parse_amazon_product(item, "Electronics") is None


def test_parse_amazon_product_truncates_long_description():
    item = {
        "title": "Widget",
        "description": ["x" * 600],
        "price": "$5",
        "parent_asin": "x",
    }
    result = _parse_amazon_product(item, "Electronics")
    assert len(result["description"]) == 500


def test_build_synthetic_users_and_addresses_default_count():
    users, addresses = build_synthetic_users_and_addresses()

    assert len(users) == N_USERS
    assert all(u["email"] and u["name"] for u in users)
    # every address must reference a real seeded user id
    user_ids = {u["id"] for u in users}
    assert all(a["user_id"] in user_ids for a in addresses)
    # 1-3 addresses per user
    assert len(addresses) >= len(users)


def test_build_synthetic_users_and_addresses_custom_count():
    users, addresses = build_synthetic_users_and_addresses(n_users=5)
    assert len(users) == 5
