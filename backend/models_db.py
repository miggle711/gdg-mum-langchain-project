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
