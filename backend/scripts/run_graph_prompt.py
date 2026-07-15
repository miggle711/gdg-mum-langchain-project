import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph import chat_graph


prompt = sys.argv[1] if len(sys.argv) > 1 else "Do you have any shoes under $300?"


async def main():
    result = await chat_graph.ainvoke(
        {
            "input": prompt,
            "chat_history": [],
            "session_id": "run-graph-prompt-script",
        }
    )

    print("PROMPT:", prompt)
    print("FULL RESULT:", result)
    print("INTENT:", result.get("intent"))
    print("RESPONSE:")
    print(result.get("response"))


asyncio.run(main())
