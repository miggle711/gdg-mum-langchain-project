import logging
import sys
import os
import uuid

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException

from db import get_session
from models_db import Product
from search import build_product_embedding, es_upsert_document, es_delete_document
from app.models import ProductCreateRequest, ProductUpdateRequest, ProductResponse

router = APIRouter()


def _to_response(product: Product) -> ProductResponse:
    return ProductResponse(
        id=product.id,
        name=product.name,
        description=product.description,
        category=product.category,
        price=product.price,
        original_price=product.original_price,
        rating=product.rating,
        reviews=product.reviews or 0,
        image=product.image,
    )


def _to_es_doc(product: Product, embedding: list[float]) -> dict:
    return {
        "id": product.id,
        "name": product.name,
        "description": product.description or "",
        "category": product.category,
        "price": product.price,
        "original_price": product.original_price,
        "rating": product.rating,
        "reviews": product.reviews or 0,
        "image": product.image or "",
        "embedding": embedding,
    }


@router.post("/products", status_code=201)
async def create_product(body: ProductCreateRequest) -> ProductResponse:
    try:
        product = Product(
            id=uuid.uuid4().hex,
            name=body.name,
            description=body.description,
            category=body.category,
            price=body.price,
            original_price=body.original_price,
            rating=body.rating,
            reviews=body.reviews,
            image=body.image,
        )
        async with get_session() as session:
            session.add(product)
            await session.commit()

        # Postgres write is committed: this request is a success from
        # here on, regardless of what happens to the ES sync below.
        response = _to_response(product)
        try:
            embedding = build_product_embedding(product.name, product.description)
            es_upsert_document(_to_es_doc(product, embedding))
        except Exception:
            logger.exception(
                "ES sync failed after Postgres commit for new product %s — "
                "product is persisted but not yet searchable until retried.",
                product.id,
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in create_product: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/products/{product_id}")
async def update_product(product_id: str, body: ProductUpdateRequest) -> ProductResponse:
    try:
        async with get_session() as session:
            product = await session.get(Product, product_id)
            if product is None:
                raise HTTPException(status_code=404, detail="Product not found")

            # Unconditional re-embed on every update, not just when
            # name/description change — content_hash exists on Product but
            # isn't populated anywhere, so there's no cheap way to detect
            # "only price changed" without fetching+diffing every field.
            # Worth revisiting once content_hash is actually load-bearing.
            for field, value in body.model_dump(exclude_unset=True).items():
                setattr(product, field, value)

            await session.commit()
            response = _to_response(product)

        try:
            embedding = build_product_embedding(product.name, product.description)
            es_upsert_document(_to_es_doc(product, embedding))
        except Exception:
            logger.exception(
                "ES sync failed after Postgres commit for updated product %s — "
                "Postgres has the new values but ES is stale until retried.",
                product_id,
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in update_product: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/products/{product_id}")
async def delete_product(product_id: str) -> dict[str, str]:
    try:
        async with get_session() as session:
            product = await session.get(Product, product_id)
            if product is None:
                raise HTTPException(status_code=404, detail="Product not found")
            await session.delete(product)
            await session.commit()

        try:
            es_delete_document(product_id)
        except Exception:
            logger.exception(
                "ES sync failed after Postgres delete for product %s — "
                "product is gone from Postgres but may still appear in search until retried.",
                product_id,
            )
        return {"message": "Product deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in delete_product: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
