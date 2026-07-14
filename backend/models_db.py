import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from db import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True)  # parent_asin — matches the ES doc _id already in use
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String, nullable=False, index=True)
    price = Column(Float, nullable=False)
    original_price = Column(Float, nullable=True)
    rating = Column(Float, nullable=True)
    reviews = Column(Integer, nullable=True, default=0)
    image = Column(Text, nullable=True)
    content_hash = Column(String, nullable=True)  # populated later by a CDC pipeline (#40); not set in Phase 1 seed
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    images = relationship("ProductImage", back_populates="product", cascade="all, delete-orphan")
    reviews_rel = relationship("Review", back_populates="product", cascade="all, delete-orphan")


class ProductImage(Base):
    __tablename__ = "product_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    image_url = Column(Text, nullable=False)
    position = Column(Integer, nullable=False, default=0)

    product = relationship("Product", back_populates="images")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    addresses = relationship("Address", back_populates="user", cascade="all, delete-orphan")


class Address(Base):
    __tablename__ = "addresses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    street = Column(String, nullable=False)
    city = Column(String, nullable=False)
    state = Column(String, nullable=True)
    zip_code = Column(String, nullable=False)
    country = Column(String, nullable=False)

    user = relationship("User", back_populates="addresses")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    rating = Column(Float, nullable=False)
    title = Column(Text, nullable=True)
    text = Column(Text, nullable=True)
    verified_purchase = Column(Boolean, nullable=False, default=False)
    helpful_vote = Column(Integer, nullable=False, default=0)
    timestamp = Column(BigInteger, nullable=True)  # raw epoch-ms from source dataset, kept as-is

    product = relationship("Product", back_populates="reviews_rel")


class Cart(Base):
    __tablename__ = "cart"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    items = relationship("CartItem", back_populates="cart", cascade="all, delete-orphan")


class CartItem(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cart_id = Column(Integer, ForeignKey("cart.id", ondelete="CASCADE"), nullable=False, index=True)
    product_id = Column(String, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    quantity = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # No frozen price here — cart_items always reflects the current products.price
    # at read/checkout time. Freezing happens only in OrderItem.unit_price below.
    cart = relationship("Cart", back_populates="items")
    product = relationship("Product")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    # RESTRICT (not CASCADE, unlike every other FK in this file) — orders are
    # append-only financial history; deleting an address with order history
    # should fail loudly, not silently erase the record of what was ordered.
    address_id = Column(Integer, ForeignKey("addresses.id", ondelete="RESTRICT"), nullable=False, index=True)
    status = Column(String, nullable=False, default="paid", index=True)  # pending/paid/shipped/delivered/cancelled
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")
    payment = relationship("Payment", back_populates="order", uselist=False, cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True)
    # RESTRICT, same reasoning as Order.address_id above.
    product_id = Column(String, ForeignKey("products.id", ondelete="RESTRICT"), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)  # frozen at checkout — never recomputed after

    order = relationship("Order", back_populates="items")
    product = relationship("Product")


class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    amount = Column(Float, nullable=False)
    status = Column(String, nullable=False, default="succeeded")  # mocked — always succeeds in Phase 2, no real provider
    provider_reference = Column(String, nullable=False)  # fake reference string, e.g. f"mock_{uuid4().hex}"
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    order = relationship("Order", back_populates="payment")
