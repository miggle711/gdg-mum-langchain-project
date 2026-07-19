import json
from langfuse import Langfuse, get_client
from app.config import settings

Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
)

langfuse = get_client()

with open("golden_dataset.json") as f:
    golden_items = json.load(f)

DATASET_NAME = "ecommerce-golden-set"

langfuse.create_dataset(
    name=DATASET_NAME,
    description="Golden queries for evaluating the ecommerce agent, grounded in the live product catalog",
)

for item in golden_items:
    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME,
        input=item["input"],
        expected_output=item["reference_answer"],
        metadata={
            "id": item["id"],
            "type": item["type"],
            "expected_tool": item.get("expected_tool"),
            "expected_filters": item.get("expected_filters"),
            "anchor_products": item.get("anchor_products"),
        },
    )

langfuse.flush()
print(f"Pushed {len(golden_items)} items to dataset '{DATASET_NAME}'")