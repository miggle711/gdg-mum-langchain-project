"""Structural ORM tests for models_db.py — columns, relationships, cascades.

Uses an in-memory async SQLite engine rather than a real Postgres connection.
This is an acceptable stand-in specifically because these are structural
ORM tests (do the FKs/relationships/cascades behave as declared), not tests
of Postgres-specific SQL (no JSONB operators, no Postgres-only functions
used in Phase 1's simple CRUD). 
"""

import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from db import Base
from models_db import (
    Address,
    Cart,
    CartItem,
    Order,
    OrderItem,
    Payment,
    Product,
    ProductImage,
    Review,
    User,
)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # SQLite does not enforce foreign keys by default (PRAGMA foreign_keys
    # is off unless explicitly turned on per-connection) — without this,
    # ondelete=CASCADE/RESTRICT would silently no-op here even though real
    # Postgres enforces them, giving a false sense of coverage.
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s

    await engine.dispose()


async def test_product_with_images_and_reviews(session):
    product = Product(id="p1", name="Widget", category="Electronics", price=9.99)
    session.add(product)
    await session.commit()

    session.add(ProductImage(product_id="p1", image_url="https://example.com/1.jpg", position=0))
    session.add(Review(product_id="p1", rating=4.5, title="Great", text="Loved it"))
    await session.commit()

    result = await session.execute(
        select(Product)
        .where(Product.id == "p1")
        .options(selectinload(Product.images), selectinload(Product.reviews_rel))
    )
    loaded = result.scalar_one()

    assert len(loaded.images) == 1
    assert loaded.images[0].image_url == "https://example.com/1.jpg"
    assert len(loaded.reviews_rel) == 1
    assert loaded.reviews_rel[0].text == "Loved it"


async def test_deleting_product_cascades_to_images_and_reviews(session):
    product = Product(id="p1", name="Widget", category="Electronics", price=9.99)
    session.add(product)
    await session.commit()

    session.add(ProductImage(product_id="p1", image_url="https://example.com/1.jpg"))
    session.add(Review(product_id="p1", rating=4.5))
    await session.commit()

    await session.delete(product)
    await session.commit()

    images = (await session.execute(select(ProductImage))).scalars().all()
    reviews = (await session.execute(select(Review))).scalars().all()

    assert images == []
    assert reviews == []


async def test_user_with_addresses(session):
    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.commit()

    session.add(Address(user_id=user.id, street="1 Main St", city="Springfield", zip_code="00000", country="US"))
    await session.commit()

    result = await session.execute(
        select(User).where(User.email == "a@example.com").options(selectinload(User.addresses))
    )
    loaded = result.scalar_one()

    assert len(loaded.addresses) == 1
    assert loaded.addresses[0].city == "Springfield"


async def test_deleting_user_cascades_to_addresses(session):
    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.commit()

    session.add(Address(user_id=user.id, street="1 Main St", city="Springfield", zip_code="00000", country="US"))
    await session.commit()

    await session.delete(user)
    await session.commit()

    addresses = (await session.execute(select(Address))).scalars().all()
    assert addresses == []


async def test_product_reviews_column_and_relationship_are_independent(session):
    """Product.reviews is a denormalized review COUNT column (carried over
    from the ES schema); Product.reviews_rel is the real relationship to
    Review rows. They're intentionally named differently to avoid a
    collision — this test guards against accidentally conflating them."""
    product = Product(id="p1", name="Widget", category="Electronics", price=9.99, reviews=42)
    session.add(product)
    await session.commit()

    result = await session.execute(
        select(Product).where(Product.id == "p1").options(selectinload(Product.reviews_rel))
    )
    loaded = result.scalar_one()

    assert loaded.reviews == 42
    assert loaded.reviews_rel == []


# --- Phase 2: cart / orders / payments ---


async def test_cart_with_items(session):
    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.flush()

    product = Product(id="p1", name="Widget", category="Electronics", price=9.99)
    session.add(product)
    await session.flush()

    cart = Cart(user_id=user.id)
    session.add(cart)
    await session.flush()

    session.add(CartItem(cart_id=cart.id, product_id="p1", quantity=2))
    await session.commit()

    result = await session.execute(
        select(Cart).where(Cart.user_id == user.id).options(selectinload(Cart.items))
    )
    loaded = result.scalar_one()

    assert len(loaded.items) == 1
    assert loaded.items[0].product_id == "p1"
    assert loaded.items[0].quantity == 2


async def test_cart_user_id_must_be_unique(session):
    """One cart per user — enforced at the DB level."""
    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.flush()

    session.add(Cart(user_id=user.id))
    await session.commit()

    session.add(Cart(user_id=user.id))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_deleting_cart_cascades_to_cart_items(session):
    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.flush()

    product = Product(id="p1", name="Widget", category="Electronics", price=9.99)
    session.add(product)
    await session.flush()

    cart = Cart(user_id=user.id)
    session.add(cart)
    await session.flush()

    session.add(CartItem(cart_id=cart.id, product_id="p1", quantity=1))
    await session.commit()

    await session.delete(cart)
    await session.commit()

    items = (await session.execute(select(CartItem))).scalars().all()
    assert items == []


async def test_order_with_items_and_payment(session):
    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.flush()

    address = Address(user_id=user.id, street="1 Main St", city="Springfield", zip_code="00000", country="US")
    session.add(address)
    await session.flush()

    product = Product(id="p1", name="Widget", category="Electronics", price=9.99)
    session.add(product)
    await session.flush()

    order = Order(user_id=user.id, address_id=address.id, status="paid")
    session.add(order)
    await session.flush()

    session.add(OrderItem(order_id=order.id, product_id="p1", quantity=2, unit_price=9.99))
    session.add(Payment(order_id=order.id, amount=19.98, status="succeeded", provider_reference="mock_ref"))
    await session.commit()

    result = await session.execute(
        select(Order)
        .where(Order.id == order.id)
        .options(selectinload(Order.items), selectinload(Order.payment))
    )
    loaded = result.scalar_one()

    assert len(loaded.items) == 1
    assert loaded.items[0].unit_price == 9.99
    assert loaded.payment.status == "succeeded"
    assert loaded.payment.provider_reference == "mock_ref"


async def test_order_item_unit_price_stays_frozen_after_product_price_changes(session):
    """The core invariant of the checkout flow: unit_price is captured at
    checkout time and must never change even if the product's current price
    changes afterward."""
    product = Product(id="p1", name="Widget", category="Electronics", price=9.99)
    session.add(product)
    await session.flush()

    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.flush()

    address = Address(user_id=user.id, street="1 Main St", city="Springfield", zip_code="00000", country="US")
    session.add(address)
    await session.flush()

    order = Order(user_id=user.id, address_id=address.id, status="paid")
    session.add(order)
    await session.flush()

    session.add(OrderItem(order_id=order.id, product_id="p1", quantity=1, unit_price=9.99))
    await session.commit()

    product.price = 999.99
    await session.commit()

    result = await session.execute(select(OrderItem).where(OrderItem.order_id == order.id))
    order_item = result.scalar_one()

    assert order_item.unit_price == 9.99


async def test_deleting_order_cascades_to_order_items_and_payment(session):
    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.flush()

    address = Address(user_id=user.id, street="1 Main St", city="Springfield", zip_code="00000", country="US")
    session.add(address)
    await session.flush()

    product = Product(id="p1", name="Widget", category="Electronics", price=9.99)
    session.add(product)
    await session.flush()

    order = Order(user_id=user.id, address_id=address.id, status="paid")
    session.add(order)
    await session.flush()

    session.add(OrderItem(order_id=order.id, product_id="p1", quantity=1, unit_price=9.99))
    session.add(Payment(order_id=order.id, amount=9.99, status="succeeded", provider_reference="mock_ref"))
    await session.commit()

    await session.delete(order)
    await session.commit()

    items = (await session.execute(select(OrderItem))).scalars().all()
    payments = (await session.execute(select(Payment))).scalars().all()

    assert items == []
    assert payments == []


async def test_deleting_address_with_order_history_is_restricted(session):
    """orders.address_id uses ondelete=RESTRICT (not CASCADE, unlike every
    other FK in this file) — orders are append-only financial history, so
    deleting a referenced address must fail loudly, not silently erase it."""
    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.flush()

    address = Address(user_id=user.id, street="1 Main St", city="Springfield", zip_code="00000", country="US")
    session.add(address)
    await session.flush()

    order = Order(user_id=user.id, address_id=address.id, status="paid")
    session.add(order)
    await session.commit()

    await session.delete(address)
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_deleting_product_with_order_history_is_restricted(session):
    """order_items.product_id uses ondelete=RESTRICT, same reasoning as
    orders.address_id above."""
    user = User(email="a@example.com", name="Alice")
    session.add(user)
    await session.flush()

    address = Address(user_id=user.id, street="1 Main St", city="Springfield", zip_code="00000", country="US")
    session.add(address)
    await session.flush()

    product = Product(id="p1", name="Widget", category="Electronics", price=9.99)
    session.add(product)
    await session.flush()

    order = Order(user_id=user.id, address_id=address.id, status="paid")
    session.add(order)
    await session.flush()

    session.add(OrderItem(order_id=order.id, product_id="p1", quantity=1, unit_price=9.99))
    await session.commit()

    await session.delete(product)
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()
