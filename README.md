# gdg-mum-langchain-project

A minimal frontend chatbot project for the GDG Monash Malaysia Tech AI Team.

## Overview

This repository contains the frontend UI for a chatbot experience built with Angular. The app is packaged with Docker and served through nginx so it can run as a simple static web app locally.

## Tech Stack

backend:

- To be filled out

frontend:

- Angular
- TypeScript
- Docker
- nginx

## Features

- Floating chat button
- Chat panel UI
- Message input and display
- Dockerized frontend build
- nginx serving the compiled app

## Running with Docker

1. Make sure Docker Desktop is running.
2. From the project root, run:

```bash
docker compose up --build
```

3. Open:

```text
http://localhost
```

## Project Structure

```text
frontend/chatbot-ui/
README.md
```

## nginx

The frontend container uses nginx to:

- serve the built Angular app
- handle single-page app routing
- proxy API requests under /api to the backend service
- keep the deployment lightweight

## Notes

This repository is currently focused on the frontend UI and Docker setup.
