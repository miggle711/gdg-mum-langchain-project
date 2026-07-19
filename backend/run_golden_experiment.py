import asyncio
import json
import os
from langfuse import Langfuse, get_client
from app.config import settings
from app.graph import chat_graph

Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
)

langfuse = get_client()
DATASET_NAME = "ecommerce-golden-set"
RUN_NAME = "baseline-run-2"
PROGRESS_FILE = "golden_run_progress.json"


def load_completed():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_completed(completed):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(completed), f)


async def main():
    dataset = langfuse.get_dataset(DATASET_NAME)
    completed = load_completed()

    for item in dataset.items:
        if item.id in completed:
            print(f"Skipping already-completed: {item.id}")
            continue

        try:
            with item.run(
                run_name=RUN_NAME,
                run_description="Second baseline run, after fixing intent classification quota fallback",
            ) as root_span:
                result = await chat_graph.ainvoke({
                    "input": item.input,
                    "chat_history": [],
                    "session_id": f"golden-{item.id}",
                })
                response = result.get("response", "")
                root_span.update(output={"response": response})

            completed.add(item.id)
            save_completed(completed)
            print(f"Done: {item.id}")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Failed on {item.id}: {e}")
            save_completed(completed)
            break

    langfuse.flush()
    print(f"Experiment run complete or stopped early. {len(completed)}/{len(dataset.items)} items done.")


asyncio.run(main())