"""Structural tests for the product write-path's Postgres-side logic
(backend/app/routes/products.py) — exercised directly against the ORM via
the same in-memory async SQLite fixture pattern as test_models_db.py, not
through HTTP/TestClient. See test_models_db.py's session fixture docstring
for why in-memory SQLite is an acceptable stand-in here, and see the
products.py route file for why a real TestClient-based integration test
suite was deliberately not attempted (a known aiosqlite/TestClient
event-loop bug caused state to leak between tests when this was tried for
the cart routes).
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db import Base
from models_db import Product, ProductImage, Review


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s

    await engine.dispose()


async def test_create_product_with_generated_uuid_id(session):
    """Mirrors products.py's create_product: id = uuid.uuid4().hex,
    not a dataset parent_asin."""
    product = Product(
        id=uuid.uuid4().hex,
        name="Widget",
        category="Electronics",
        price=9.99,
    )
    session.add(product)
    await session.commit()

    assert len(product.id) == 32  # uuid4().hex is a 32-char hex string
    result = await session.execute(select(Product).where(Product.id == product.id))
    assert result.scalar_one().name == "Widget"


async def test_create_two_products_get_distinct_ids(session):
    p1 = Product(id=uuid.uuid4().hex, name="Widget A", category="Electronics", price=9.99)
    p2 = Product(id=uuid.uuid4().hex, name="Widget B", category="Electronics", price=19.99)
    session.add_all([p1, p2])
    await session.commit()

    assert p1.id != p2.id


async def test_partial_update_only_touches_specified_fields(session):
    """Mirrors products.py's update_product: only fields present in
    model_dump(exclude_unset=True) get applied via setattr."""
    product = Product(
        id=uuid.uuid4().hex,
        name="Widget",
        description="Original description",
        category="Electronics",
        price=9.99,
        rating=4.0,
    )
    session.add(product)
    await session.commit()

    # simulate ProductUpdateRequest(price=19.99).model_dump(exclude_unset=True)
    update_fields = {"price": 19.99}
    for field, value in update_fields.items():
        setattr(product, field, value)
    await session.commit()

    result = await session.execute(select(Product).where(Product.id == product.id))
    updated = result.scalar_one()
    assert updated.price == 19.99
    assert updated.name == "Widget"  # untouched
    assert updated.description == "Original description"  # untouched
    assert updated.rating == 4.0  # untouched


async def test_partial_update_can_explicitly_set_field_to_null(session):
    """exclude_unset=True (not exclude_none=True) means an explicit null
    in the request body IS applied — distinguishing "field omitted" from
    "field explicitly cleared"."""
    product = Product(
        id=uuid.uuid4().hex,
        name="Widget",
        description="Will be cleared",
        category="Electronics",
        price=9.99,
    )
    session.add(product)
    await session.commit()

    # simulate ProductUpdateRequest(description=None).model_dump(exclude_unset=True)
    # — description IS in the dict (with value None) because it was
    # explicitly set on the request, unlike a field that was never sent.
    update_fields = {"description": None}
    for field, value in update_fields.items():
        setattr(product, field, value)
    await session.commit()

    result = await session.execute(select(Product).where(Product.id == product.id))
    assert result.scalar_one().description is None


async def test_delete_product_cascades_to_images_and_reviews(session):
    product = Product(id=uuid.uuid4().hex, name="Widget", category="Electronics", price=9.99)
    session.add(product)
    await session.commit()

    session.add(ProductImage(product_id=product.id, image_url="https://example.com/1.jpg"))
    session.add(Review(product_id=product.id, rating=4.5, text="Great"))
    await session.commit()

    await session.delete(product)
    await session.commit()

    images = (await session.execute(select(ProductImage))).scalars().all()
    reviews = (await session.execute(select(Review))).scalars().all()
    assert images == []
    assert reviews == []
