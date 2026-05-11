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
from typing import TypedDict

load_dotenv() # load the .env file to get environment variables like GOOGLE_API_KEY

# define FastAPI application (the backend server for our conversation API)
app = FastAPI(title="LangChain Conversation API")

class ConversationData(TypedDict):
    # we define a typed dictionary to store the conversation chain and memory for each conversation ID
    chain: ConversationChain 
    memory: ConversationBufferMemory

# System prompt for ecommerce customer service agent
# this is prepended to the conversation history to provide context to the LLM about its role and responsibilities
ECOMMERCE_SYSTEM_PROMPT = """You are a helpful and professional ecommerce customer service assistant for our online store.

Your responsibilities:
- Help customers find products by answering questions about our catalog
- Provide information about product specifications, pricing, and availability

Guidelines:
- Always be polite, professional, and empathetic
- If you don't know something about our specific products or policies, suggest they contact support

Current conversation:"""


# middleware is used to hanlde CORS (Cross-Origin Resource Sharing) which allows our frontend 
# (which may be served from a different origin) to make requests to this backend API without 
# being blocked by the browser's same-origin policy
app.add_middleware(
    CORSMiddleware,
    # we allow all origins for simplicity, but when we deploy this, we will want to restrict this to only the frontend's origin for security reasons
    allow_origins=["*"], 
    # we allow credentials (like cookies, authorization headers, etc.) to be sent in cross-origin requests
    allow_credentials=True,
    # # we allow all HTTP methods (GET, POST, DELETE, etc.)
    allow_methods=["*"], 
    allow_headers=["*"], # we allow all headers (like Content-Type, Authorization, etc.
)

# we load the Google API key from an environment variable for security reasons (so we don't hardcode it in our codebase)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# we will store conversations in memory using a dictionary where the keys are conversation IDs and the values are the conversation chains and memory objects
# this is a simple in-memory store for demonstration purposes, but in a production application, we would want to use a more robust storage solution (like a database)
#  to persist conversations across server restarts and scale better
conversations = {}


### Pydantic models for request and response validation ###
# these models define the structure of the data we expect to receive in requests and send in responses
# helps to validate incoming data and ensure our API is consistent and well-documented ;)
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


def get_or_create_conversation(conversation_id: str) -> ConversationData:
    """
    Helper function to get an existing conversation chain and memory by conversation ID, or create a new one if it doesn't exist.
    
    args:
        conversation_id (str): The unique identifier for the conversation.
    returns:
        ConversationData: A dictionary containing the conversation chain and memory objects for the given conversation ID.
    """
    if conversation_id not in conversations:
        # we create a new conversation chain and memory for this conversation ID if it doesn't exist
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7, # controls the randomness of the LLM's responses 
        )
        memory = ConversationBufferMemory() # store conversation history in memory

        # Custom prompt template with adding the system prompt 
        prompt_template = PromptTemplate(
            # add the history (conversation history) and user input (latest user message)
            input_variables=["history", "input"],
            template=ECOMMERCE_SYSTEM_PROMPT + "\n{history}\nHuman: {input}\nAssistant:"
        )

        chain = ConversationChain( # interface between the LLM and the conversation memory
            llm=llm,
            memory=memory, # the memory object that will store the conversation history and provide it as context to the LLM for generating responses
            prompt=prompt_template,
            verbose=False, 
        )
        # Store the conversation data in the dictionary with the conversation ID as the key
        conversations[conversation_id] = {
            "chain": chain, 
            "memory": memory,
        }
    return conversations[conversation_id]


@app.get("/")
def root(): 
    # Simple health check endpoint, we can use this to verify that the backend server is running and responding to requests
    return {"message": "LangChain Conversation Backend is running"}


# async because we don't want to block the server while waiting for the LLM to generate a response, allowing it to handle multiple requests concurrently
@app.post("/chat") # Endpoint to handle chat messages
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle chat messages for a specific conversation.
    """
    try:
        conversation = get_or_create_conversation(request.conversation_id)
        chain = conversation["chain"]

        # generate a response from the conversation chain using the input message from the request
        response = chain.predict(input=request.message)

        print(f"DEBUG: Response from chain: '{response}'")
        print(f"DEBUG: Response type: {type(response)}")
        print(f"DEBUG: Response length: {len(str(response))}")

        # using pydantic model to validate and structure the response we send back to the frontend
        return ChatResponse(
            conversation_id=request.conversation_id,
            response=response if response else "I apologize, but I'm having trouble generating a response at the moment.",
            message_count=len(conversation["memory"].buffer.split("\n")) // 2, # Each message is stored as "Human: ...\nAI: ...\n", so we divide by 2 to get the number of exchanges
        )
    except Exception as e:
        print(f"DEBUG: Exception in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat/start") # Start a new conversation
def start_conversation() -> dict[str, str]:
    # we generate a new unique conversation ID using uuid4 and create a new conversation chain and memory for that ID
    conversation_id = str(uuid.uuid4())
    get_or_create_conversation(conversation_id) # this will create a new conversation chain and memory for the new conversation ID
    return {
        "conversation_id": conversation_id,
        "message": "Welcome to our store! How can I help you find the perfect product today?"
    }


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
