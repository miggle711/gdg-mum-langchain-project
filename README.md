# gdg-mum-langchain-project

A LangChain-powered ecommerce customer service chatbot with Angular frontend and FastAPI backend, deployed with Docker.

## Overview

This project demonstrates stateful multi-turn conversations using LangChain and Google's Gemini API. The backend generates unique conversation IDs and maintains conversation memory, while the Angular frontend provides a clean chat UI. Conversations are preserved server-side, allowing the AI to maintain context across multiple messages.

## Tech Stack

**Backend:**

- FastAPI (Python)
- LangChain with ConversationChain
- Google Gemini API (gemini-2.5-flash)
- ConversationBufferMemory for conversation state

**Frontend:**

- Angular 19+
- TypeScript
- RxJS Observables
- Angular Material

**Deployment:**

- Docker & Docker Compose
- nginx (reverse proxy + API routing)

## Architecture

1. **Frontend** (Angular) makes HTTP request to `/api/chat/start` on component load
2. **Backend** (FastAPI) generates a unique conversation ID (UUID) and initializes a LangChain conversation chain
3. **Frontend** stores the ID and sends user messages to `/api/chat` with the ID
4. **Backend** maintains conversation history in memory and uses LangChain to generate contextual responses
5. System prompt provides ecommerce customer service instructions to the AI

### Conversation Flow

![Chat Service Sequence Diagram](docs/Chat%20Service%20Conversation-2026-05-11-155722.png)

## Features

- Stateful conversations with server-side memory management
- Real-time message display with auto-scrolling
- Ecommerce customer service system prompt
- Conversation history tracking
- Error handling and user feedback
- Full Docker deployment

## Getting Started

### Prerequisites
- Docker Desktop
- `.env` file in project root with `GOOGLE_API_KEY` set

### Run Locally

```bash
docker compose up --build
```

Then open `http://localhost` in your browser.

### Setup Guides

- **[DOCKER_SETUP.md](DOCKER_SETUP.md)** — Docker prerequisites, environment setup, troubleshooting
- **[LANGCHAIN_SETUP.md](LANGCHAIN_SETUP.md)** — Backend setup, API keys, testing


## Key Files

- **backend/app/main.py**: REST API endpoints and LangChain conversation logic
- **frontend/chatbot-ui/src/app/services/chat.ts**: Service layer for HTTP requests
- **frontend/chatbot-ui/src/app/components/chat-panel/chat-panel.ts**: Chat UI with message handling
- **docker-compose.yml**: Orchestrates backend and frontend containers

## API Endpoints

- `POST /chat/start` - Initialize conversation, returns `conversation_id` and welcome message
- `POST /chat` - Send message, returns AI response
- `GET /conversation/{id}` - Retrieve conversation history
- `DELETE /conversation/{id}` - Delete conversation
- `GET /conversations` - List all conversations

## Development Notes

- Conversations are stored in-memory; they reset on server restart
- The system prompt is configured for ecommerce customer service
