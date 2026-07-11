"""Structural ORM tests for models_db.py — columns, relationships, cascades.

Uses an in-memory async SQLite engine rather than a real Postgres connection.
This is an acceptable stand-in specifically because these are structural
ORM tests (do the FKs/relationships/cascades behave as declared), not tests
of Postgres-specific SQL (no JSONB operators, no Postgres-only functions
used in Phase 1's simple CRUD). 
"""

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from db import Base
from models_db import Address, Product, ProductImage, Review, User


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
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
