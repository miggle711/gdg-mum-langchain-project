from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import uuid
from typing import Optional, TypedDict
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db, seed_db, save_messages, load_messages, _get_redis
from tools import PRODUCT_TOOLS

load_dotenv()

# initialize the database and seed it with sample data on startup
# TODO: remove this when deploying to production and use proper database migrations instead. This is just for demo purposes to ensure the database is ready to use when the app starts.
init_db()
seed_db()

app = FastAPI(title="LangChain Conversation API")

class ConversationData(TypedDict):
    history: InMemoryChatMessageHistory
    is_new: bool

ECOMMERCE_SYSTEM_PROMPT = """You are a helpful and professional ecommerce customer service assistant for our online store.

Your responsibilities:
- Help customers find products by answering questions about our catalog
- Use the available tools to search and filter products when customers ask
- Provide information about product specifications, pricing, and availability

Guidelines:
- Always be polite, professional, and empathetic
- When describing products, include price, rating, and number of reviews
- If you don't know something about our specific products or policies, suggest they contact support"""


# middleware is used to hanlde CORS (Cross-Origin Resource Sharing) which allows our frontend 
# (which may be served from a different origin) to make requests to this backend API without 
# being blocked by the browser's same-origin poli
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Shared singletons — created once at startup and reused across all conversations.
# The LLM client, prompt, and AgentExecutor are stateless between calls;
# per-conversation state lives entirely in the chat_history we pass in each invocation.
_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", ECOMMERCE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

_agent_executor = AgentExecutor(
    agent=create_tool_calling_agent(_llm, PRODUCT_TOOLS, _prompt),
    tools=PRODUCT_TOOLS,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)

class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    message_count: int


def get_or_create_conversation(conversation_id: str) -> ConversationData:
    messages = load_messages(conversation_id)
    history = InMemoryChatMessageHistory()
    history.add_messages(messages)
    return {
        "history": history,
        "is_new": len(messages) == 0,
    }


@app.get("/")
def root():
    return {"message": "LangChain Conversation Backend is running"}


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        conversation = get_or_create_conversation(request.conversation_id)
        history = conversation["history"]

        result = _agent_executor.invoke({
            "input": request.message,
            "chat_history": history.messages,
        })
        response_text = result["output"]

        # Update history after successful invocation only
        history.add_user_message(request.message)
        history.add_ai_message(response_text)
        save_messages(request.conversation_id, history.messages)

        return ChatResponse(
            conversation_id=request.conversation_id,
            response=response_text or "I apologize, but I'm having trouble generating a response at the moment.",
            message_count=len(history.messages) // 2,
        )
    except Exception as e:
        print(f"Exception in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/start")
def start_conversation() -> dict[str, str]:
    conversation_id = str(uuid.uuid4())
    get_or_create_conversation(conversation_id)
    return {
        "conversation_id": conversation_id,
        "message": "Welcome to our store! How can I help you find the perfect product today?"
    }


@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    messages = load_messages(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages_out = []
    for msg in messages:
        role = "human" if isinstance(msg, HumanMessage) else "ai"
        messages_out.append({"role": role, "content": msg.content})

    return {
        "conversation_id": conversation_id,
        "history": messages_out,
        "message_count": len(messages) // 2,
    }


@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    r = _get_redis()
    deleted = r.delete(f"conversation:{conversation_id}")
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}


@app.get("/conversations")
def list_conversations():
    r = _get_redis()
    keys = r.keys("conversation:*")
    return {
        "conversations": [
            {
                "conversation_id": key.split(":", 1)[1],
                "message_count": len(load_messages(key.split(":", 1)[1])) // 2,
            }
            for key in keys
        ]
    }
