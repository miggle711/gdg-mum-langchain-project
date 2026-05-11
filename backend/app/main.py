from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import uuid
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LangChain Conversation API")

# System prompt for ecommerce customer service agent
ECOMMERCE_SYSTEM_PROMPT = """You are a helpful and professional ecommerce customer service assistant for our online store.

Your responsibilities:
- Help customers find products by answering questions about our catalog
- Provide information about product specifications, pricing, and availability
- Assist with order tracking and delivery inquiries
- Handle returns, refunds, and exchange requests
- Answer questions about payment methods and shipping policies
- Provide recommendations based on customer needs

Guidelines:
- Always be polite, professional, and empathetic
- If you don't know something about our specific products or policies, suggest they contact support
- Keep responses concise and helpful
- Ask clarifying questions if needed to better assist the customer
- For sensitive issues (refunds, returns), be accommodating and helpful
- Always maintain a friendly tone

Current conversation:"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # we allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"], # we allow all HTTP methods (GET, POST, DELETE, etc.)
    allow_headers=["*"], # we allow all headers (like Content-Type, Authorization, etc.
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

conversations = {}


### Pydantic models for request and response validation ###
class Message(BaseModel):
    content: str

class ChatRequest(BaseModel): # Model for requests to the /chat endpoint
    conversation_id: str
    message: str
    system_prompt: Optional[str] = None  # Optional custom system prompt

class ChatResponse(BaseModel): # Model for responses from the /chat endpoint
    conversation_id: str
    response: str
    message_count: int


def get_or_create_conversation(conversation_id: str):
    """
    Get or create a conversation by its ID.
    """
    if conversation_id not in conversations:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
        )
        memory = ConversationBufferMemory() # store conversation history in memory

        # Custom prompt template with system context
        prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template=ECOMMERCE_SYSTEM_PROMPT + "\n{history}\nHuman: {input}\nAssistant:"
        )

        chain = ConversationChain( # interface between the LLM and the conversation memory
            llm=llm,
            memory=memory,
            prompt=prompt_template,
            verbose=False,
        )
        conversations[conversation_id] = {
            "chain": chain,
            "memory": memory,
        }
    return conversations[conversation_id]


@app.get("/")
def root(): # Simple health check endpoint
    return {"message": "LangChain Conversation Backend is running"}


@app.post("/chat/start") # Start a new conversation
def start_conversation():
    conversation_id = str(uuid.uuid4())
    get_or_create_conversation(conversation_id)
    return {
        "conversation_id": conversation_id,
        "message": "Welcome to our store! How can I help you find the perfect product today?"
    }


@app.post("/chat") # Endpoint to handle chat messages
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        conversation = get_or_create_conversation(request.conversation_id)
        chain = conversation["chain"]

        response = chain.predict(input=request.message)

        print(f"DEBUG: Response from chain: '{response}'")
        print(f"DEBUG: Response type: {type(response)}")
        print(f"DEBUG: Response length: {len(str(response))}")

        return ChatResponse(
            conversation_id=request.conversation_id,
            response=response if response else "I apologize, but I'm having trouble generating a response at the moment.",
            message_count=len(conversation["memory"].buffer.split("\n")) // 2, # Each message is stored as "Human: ...\nAI: ...\n", so we divide by 2 to get the number of exchanges
        )
    except Exception as e:
        print(f"DEBUG: Exception in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    if conversation_id not in conversations:

        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation = conversations[conversation_id]
    return {
        "conversation_id": conversation_id,
        "history": conversation["memory"].buffer,
        "message_count": len(conversation["memory"].buffer.split("\n")) // 2,
    }


@app.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    del conversations[conversation_id]
    return {"message": "Conversation deleted"}


@app.get("/conversations")
def list_conversations():
    return {
        "conversations": [
            {
                "conversation_id": cid,
                "message_count": len(conv["memory"].buffer.split("\n")) // 2,
            }
            for cid, conv in conversations.items()
        ]
    }
