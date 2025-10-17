🤖 Gemini Task Receiver & Automation Service

This project implements a secure, scalable web service using FastAPI to receive task payloads from an evaluation server, verify the request integrity, trigger a file generation and deployment workflow in the background, and provide real-time status updates.

It is designed to be deployed as an always-on service to platforms like Render or Hugging Face Spaces.

🌟 Summary

This service acts as the central command and control application for handling automated tasks.

Receive Task: An external evaluation server sends a POST request to the /ready endpoint with a task payload (including the brief, attachments, evaluation_url, and a secret).

Verify Security: The service instantly verifies the student's provided secret against an environment variable (STUDENT_SECRET).

Process Async: Upon successful verification, the heavy-lifting (LLM API calls, file generation, Git operations) is started as a non-blocking background task using asyncio.create_task.

Instant Response: A 200 OK response is immediately returned to the caller, preventing timeouts.

Deployment & Notification: The background task generates the required files, commits them to the specified GitHub repository, pushes them, and finally notifies the evaluation server via the evaluation_url.

⚙️ Setup and Installation

Prerequisites

Python 3.10+

A Google Gemini API Key

A GitHub Personal Access Token (PAT) with repo scope permissions.

Local Setup

Clone the repository:

git clone [your-repo-url]
cd gemini-project


Create a virtual environment and install dependencies:

python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
pip install -r requirements.txt


Configure Environment Variables

Create a file named .env in the root directory to store your secrets. (IMPORTANT: This file is ignored by Git and should never be committed!)

# .env file (DO NOT COMMIT!)
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
GITHUB_TOKEN="YOUR_GITHUB_PAT_HERE"
STUDENT_SECRET="YOUR_SECRET_STRING_FOR_VERIFICATION"
GITHUB_USERNAME="your_github_username"


Run the service:

uvicorn main:app --reload


The application will be running at http://127.0.0.1:8000.

🚀 Usage

The service exposes three primary endpoints:

Endpoint

Method

Description

/

GET

Root endpoint. Returns a simple "Service is running" message.

/status

GET

Displays the data from the last successfully received task payload.

/ready

POST

Main operational endpoint. Receives the TaskRequest payload from the evaluation server to kick off the generation and deployment process.

Submitting a Task (via /ready)

To submit a task, an external system must send a JSON payload matching the TaskRequest schema (defined in models.py).

Example Payload Structure:

{
  "email": "user@example.com",
  "secret": "YOUR_SECRET_STRING_FOR_VERIFICATION",
  "task": "my-first-task-001",
  "round": 1,
  "nonce": "a1b2c3d4e5f6",
  "brief": "Write a python script to calculate the Nth Fibonacci number.",
  "checks": [
    "Repo has MIT license",
    "README.md is professional"
  ],
  "evaluation_url": "[https://api.evalserver.com/notify/task/001](https://api.evalserver.com/notify/task/001)",
  "attachments": []
}


🧠 Code Explanation

The core logic is distributed across four Python files:

File

Purpose

Key Features

config.py

Configuration Management

Uses pydantic-settings to securely load environment variables (.env or system variables) into a strongly typed Settings object. Uses @lru_cache to ensure settings are loaded only once.

models.py

Data Validation

Defines the TaskRequest Pydantic model, ensuring all incoming POST requests to /ready are validated against the expected schema, including EmailStr checks.

main.py

FastAPI Application & Core Logic

Initializes the FastAPI app. Contains the /ready endpoint, which performs secret verification, immediately returns an HTTP 200, and then calls asyncio.create_task(generate_files_and_deploy) to handle all time-consuming LLM and Git operations asynchronously.

Dockerfile

Containerization

Provides instructions to build a lightweight Docker image, installing dependencies and running the service with Uvicorn, ready for cloud deployment.

Asynchronous Background Task (generate_files_and_deploy):
The most critical function is the asynchronous one. It orchestrates the full pipeline:

LLM Call: Calls the Gemini API to generate the required code/files.

File Parsing: Uses re.findall to safely extract content from the generated file blocks (e.g., \``python:Title:file.py\n...\n```eof`).

File Creation: Writes the extracted content to the filesystem.

Git Automation: Uses GitPython to perform git add, git commit, and git push with the GITHUB_TOKEN.

Notification: Sends a final POST request to the provided evaluation_url to signal completion and provide the GitHub Pages URL.

📄 License

This project is released under the MIT License. For more details, see the accompanying LICENSE file
