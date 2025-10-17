from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from models import TaskRequest # Ensure models.py is available
from config import get_settings
import asyncio
import httpx # Used for making the HTTP notification call
import json # For parsing the structured JSON response from the LLM
import os # For configuration and file system operations
import base64
import re
import git  # For local Git operations
import time
import shutil
import stat # For robust cleanup on Windows

# Assuming this model is defined elsewhere
# --- Configuration and Setup ---
settings = get_settings()

# --- Helper Function for Security ---
def verify_secret(secret_from_request: str) -> bool:
    """Checks if the provided secret matches the expected student secret."""
    return secret_from_request == settings.STUDENT_SECRET

# --- GITHUB CONSTANTS ---
GITHUB_API_BASE = "https://api.github.com"
# Pages URL is constructed dynamically using the username from settings
GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"
# --------------------------

# LLM Configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
# NOTE: API key is left empty (or read from environment) as per instructions; 
# the execution environment is assumed to handle the required authentication.
GEMINI_API_KEY = settings.GEMINI_API_KEY 
# Initialize the FastAPI application
app = FastAPI(
    title="Automated Task Receiver & Processor",
    description="Endpoint for receiving task assignments and triggering AI code generation/deployment."
)

# Global storage for the last received task (for demonstration purposes)
received_task_data = {}

# --- REFACTORING: SPLIT deploy_to_github ---

async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int) -> git.Repo:
    """Handles creating the remote repo (R1) or cloning the existing one (R2+) into an EMPTY directory."""
    
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            # 1. CREATE or INITIALIZE REPO / CLONE EXISTING REPO
            if round_index == 1:
                print(f"   -> R1: Creating remote repository '{repo_name}'...")
                payload = {"name": repo_name, "private": False, "auto_init": True}
                response = await client.post(f"{GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                response.raise_for_status()

                # Initialize local git repo in the EMPTY path
                repo = git.Repo.init(local_path)
                repo.create_remote('origin', repo_url_auth)
                print("   -> R1: Local git repository initialized.")
            
            elif round_index >= 2:
                # Crucial part for Round 2: Cloning the existing work into the EMPTY local_path
                print(f"   -> R{round_index}: Cloning existing repository from {repo_url_http}...")
                # local_path is guaranteed to be empty due to the cleanup and directory creation in the main function
                repo = git.Repo.clone_from(repo_url_auth, local_path)
                print(f"   -> R{round_index}: Repository cloned and ready for update.")
            
            return repo

        except httpx.HTTPStatusError as e:
            print(f"--- [API ERROR] GitHub API call failed with status {e.response.status_code}: {e.response.text} ---")
            raise Exception("GitHub API call failed during repository setup.")
        except git.GitCommandError as e:
            print(f"--- [GIT ERROR] Failed to perform git operation: {e} ---")
            raise Exception("Git operation failed during repository setup.")


async def commit_and_publish(repo: git.Repo, task_id: str, round_index: int, repo_name: str) -> dict:
    """Handles adding, committing, pushing, and configuring GitHub Pages after files are saved."""

    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    repo_url_http = f"https://github.com/{github_username}/{repo_name}"

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            # 1. ADD, COMMIT, AND PUSH FILES
            # The new files (generated and attachments) are now in the local_path.
            repo.git.add(A=True)
            commit_message = f"Task {task_id} - Round {round_index}: LLM-generated app update/creation"
            repo.index.commit(commit_message)
            commit_sha = repo.head.object.hexsha
            print(f"   -> Files committed. SHA: {commit_sha}")

            # Ensure main branch consistency and push
            repo.git.branch('-M', 'main')
            print("   -> Branch renamed to 'main'.")
            repo.git.push('--set-upstream', 'origin', 'main', force=True)
            print("   -> Changes pushed to remote 'main' branch.")

            # Wait for GitHub to register the branch
            print("   -> Waiting 10 seconds for GitHub to register the main branch...")
            await asyncio.sleep(10)

            # 2. ENABLE GITHUB PAGES WITH ROBUST RETRIES
            print("   -> Enabling GitHub Pages with robust retries...")
            pages_api_url = f"{GITHUB_API_BASE}/repos/{github_username}/{repo_name}/pages"
            pages_payload = {"source": {"branch": "main", "path": "/"}}
            pages_max_retries = 5
            pages_base_delay = 3

            for retry_attempt in range(pages_max_retries):
                try:
                    pages_response = await client.get(pages_api_url, headers=headers)
                    is_configured = (pages_response.status_code == 200)

                    if is_configured:
                        print(f"   -> Pages exists. Updating configuration (Attempt {retry_attempt + 1}).")
                        (await client.put(pages_api_url, json=pages_payload, headers=headers)).raise_for_status()
                    else:
                        print(f"   -> Creating Pages configuration (Attempt {retry_attempt + 1}).")
                        (await client.post(pages_api_url, json=pages_payload, headers=headers)).raise_for_status()

                    print("   -> Pages configuration successful.")
                    break
                
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 422 and "main branch must exist" in e.response.text and retry_attempt < pages_max_retries - 1:
                        delay = pages_base_delay * (2 ** retry_attempt)
                        print(f"   -> [Timing Issue] Branch not recognized. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        raise
            else:
                raise Exception("Failed to configure GitHub Pages after multiple retries due to branch existence.")

            # 3. CONSTRUCT RETURN VALUES
            print("   -> Waiting 5 seconds for GitHub Pages deployment...")
            await asyncio.sleep(5)

            pages_url = f"{GITHUB_PAGES_BASE}/{repo_name}/"

            return {
                "repo_url": repo_url_http,
                "commit_sha": commit_sha,
                "pages_url": pages_url
            }

        except git.GitCommandError as e:
            print(f"--- [GIT ERROR] Failed to perform git operation: {e} ---")
            raise Exception("Git operation failed during deployment.")
        except httpx.HTTPStatusError as e:
            print(f"--- [API ERROR] GitHub API call failed with status {e.response.status_code}: {e.response.text} ---")
            raise Exception("GitHub API call failed during deployment.")
        except Exception as e:
            print(f"--- [CRITICAL ERROR] Deployment failed: {e} ---")
            raise

# --- REMOVED: Original deploy_to_github (replaced by setup_local_repo and commit_and_publish) ---
# The function name deploy_to_github is now DELETED.

        
def data_uri_to_gemini_part(data_uri: str) -> dict:
    """
    Extracts Base64 data and MIME type from a Data URI and formats it 
    as the 'inlineData' structure required for a Gemini API multimodal part.
    """
    if not data_uri or not data_uri.startswith("data:"):
        print("ERROR: Invalid Data URI provided.")
        return None

    try:
        # Extract MIME type and Base64 part using regex
        match = re.search(r"data:(?P<mime_type>[^;]+);base64,(?P<base64_data>.*)", data_uri, re.IGNORECASE)
        if not match:
            print("ERROR: Could not parse MIME type or base64 data from URI.")
            return None

        mime_type = match.group('mime_type')
        base64_data = match.group('base64_data')

        # Check if it's a known image type to ensure we only send images to the LLM
        if not mime_type.startswith("image/"):
            print(f"Skipping attachment with non-image MIME type: {mime_type}")
            return None
        
        return {
            "inlineData": {
                "data": base64_data,  # The Base64 string itself
                "mimeType": mime_type
            }
        }
    except Exception as e:
        print(f"ERROR creating Gemini Part from URI: {e}")
        return None

def is_image_data_uri(data_uri: str) -> bool:
    """Checks if the data URI refers to an image based on the MIME type."""
    if not data_uri.startswith("data:"):
        return False
    # Check for "image/" prefix in the MIME type part of the URI
    return re.search(r"data:image/[^;]+;base64,", data_uri, re.IGNORECASE) is not None
# --- Helper Functions for File System Operations ---

async def save_generated_files_locally(task_id: str, files: dict) -> str:
    """
    Saves the generated files (index.html, README.md, LICENSE) into a local 
    directory named after the task_id within the 'generated_tasks' folder.
    """
    base_dir = os.path.join(os.getcwd(), "generated_tasks")
    task_dir = os.path.join(base_dir, task_id)
    
    # Ensure the task-specific directory exists
    # NOTE: This directory is created earlier in the main orchestration function
    os.makedirs(task_dir, exist_ok=True)
    
    print(f"--- [LOCAL_SAVE] Saving files to: {task_dir} ---")
    
    # Write each file from the generated dictionary to the local file system
    for filename, content in files.items():
        file_path = os.path.join(task_dir, filename)
        try:
            # Write the content to the file. Assuming content is a string (text files).
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"   -> Saved: {filename} (Size: {len(content)} bytes)")
        except Exception as e:
            print(f"   -> ERROR saving {filename}: {e}")
            # If saving fails, we treat this as a critical error for the task
            raise Exception(f"Failed to save file {filename} locally.")

    return task_dir


# --- Helper Functions for External Services ---

async def call_llm_for_code(prompt: str, task_id: str, image_parts: list) -> dict:
    """
    Calls the Gemini API to generate the web application code and structured
    metadata (README and LICENSE), now supporting image inputs.
    The response is strictly validated against a JSON schema.
    """
    print(f"--- [LLM_CALL] Attempting to generate code for Task: {task_id} using Gemini API ---")
      # --- Improve vague user prompts automatically ---
    normalized_prompt = prompt.lower().strip()

    # If the user gave a vague short task like "create a captcha solver..."
    # auto-extend it into a detailed instruction so Gemini knows what to do
    if "captcha solver" in normalized_prompt and "responsive" not in normalized_prompt:
        print("--- [LLM_CALL] Detected vague captcha solver prompt — auto-expanding ---")
        prompt += (
            "\n\n"
            "Ensure the web app is a single, complete, fully responsive HTML file using Tailwind CSS. "
            "It must fetch an image from the query parameter '?url=https://.../image.png', display it, "
            "and perform OCR using Tesseract.js via CDN. "
            "If the URL parameter is missing, use the attached sample image by default. "
            "Show the recognized text and any errors clearly in the UI. "
            "Return output strictly as a JSON object with keys: 'index.html', 'README.md', and 'LICENSE'."
        )
    # Define system instruction for the model (UNCHANGED)
    system_prompt = (
        "You are an expert full-stack engineer and technical writer. Your task is to generate "
        "three files in a single structured JSON response: 'index.html', 'README.md', and 'LICENSE'. "
        "The 'index.html' must be a single, complete, fully responsive HTML file using Tailwind CSS "
        "for styling and must implement the requested application logic. The 'README.md' must be "
        "professional. The 'LICENSE' must contain the full text of the MIT license."
    )
    
    # Define the JSON response structure (UNCHANGED)
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "index.html": {"type": "STRING", "description": "The complete, single-file HTML content with inline CSS and JS, using Tailwind."},
            "README.md": {"type": "STRING", "description": "The professional Markdown content for the project README."},
            "LICENSE": {"type": "STRING", "description": "The full text of the MIT license."}
        },
        "required": ["index.html", "README.md", "LICENSE"]
    }

    # --- CONSTRUCT THE CONTENTS FIELD ---
    contents = []
    
    if image_parts:
        # Combine image parts and the text prompt.
        all_parts = image_parts + [
            { "text": prompt } 
        ]
        contents.append({ "parts": all_parts })
    else:
        # If no images, use the original structure with only the text prompt
        contents.append({ "parts": [{ "text": prompt }] })

    # Construct the final API payload
    payload = {
        "contents": contents,  
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }
    
    # Use exponential backoff for the API call 
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Construct the URL with the API key
            url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    url, 
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status() # Raises an exception for 4xx/5xx status codes
                
                # Parse the response to get the structured JSON text
                result = response.json()
                
                # Extract the generated JSON string from the result
                json_text = result['candidates'][0]['content']['parts'][0]['text']
                
                # The LLM output is a JSON string, so we need to parse it into a Python dict
                generated_files = json.loads(json_text)
                
                print(f"--- [LLM_CALL] Successfully generated files on attempt {attempt + 1}. ---")
                return generated_files

        except httpx.HTTPStatusError as e:
            print(f"--- [LLM_CALL] HTTP Error on attempt {attempt + 1}: {e}. ---")
        except (httpx.RequestError, KeyError, json.JSONDecodeError) as e:
            # Catches network errors, missing structure in the result, or invalid JSON output
            print(f"--- [LLM_CALL] Processing Error on attempt {attempt + 1}: {e}. ---")
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            print(f"--- [LLM_CALL] Retrying LLM call in {delay} seconds... ---")
            await asyncio.sleep(delay)

    # If all retries fail, we raise an exception which is caught downstream
    print("--- [LLM_CALL] Failed to generate code after multiple retries. ---")
    raise Exception("LLM Code Generation Failure")


async def notify_evaluation_server(
    evaluation_url: str, 
    email: str,          
    task_id: str, 
    round_index: int,    
    nonce: str, 
    repo_url: str,
    commit_sha: str,     
    pages_url: str       
) -> bool:
    """
    Calls the evaluation_url to notify the server that the code has been deployed.
    """
    payload = {
        "email": email,
        "task": task_id,
        "round": round_index,
        "nonce": nonce,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url  
    }
    
    max_retries = 3
    base_delay = 1
    
    print(f"--- [NOTIFICATION] Attempting to notify server at {evaluation_url} ---")
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(evaluation_url, json=payload)
                response.raise_for_status() # Raises an exception for 4xx/5xx status codes
                
                print(f"--- [NOTIFICATION] Successfully notified server. Response: {response.status_code} ---")
                return True
        except httpx.HTTPStatusError as e:
            print(f"--- [NOTIFICATION] HTTP Error on attempt {attempt + 1}: {e}. ---")
        except httpx.RequestError as e:
            print(f"--- [NOTIFICATION] Request Error on attempt {attempt + 1}: {e}. ---")
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            print(f"--- [NOTIFICATION] Retrying in {delay} seconds... ---")
            await asyncio.sleep(delay)
            
    print(f"--- [NOTIFICATION] Failed to notify evaluation server after {max_retries} attempts. ---")
    return False


async def save_attachments_locally(task_dir: str, attachments: list) -> list:
    """
    Decodes and saves attachments (provided as Base64 Data URIs) into the task directory.
    Returns a list of saved filenames.
    """
    saved_files = []
    print(f"--- [ATTACHMENTS] Processing {len(attachments)} attachments for: {task_dir} ---")
    
    for attachment in attachments:
        filename = attachment.name 
        data_uri = attachment.url
        
        if not filename or not data_uri or not data_uri.startswith("data:"):
            print(f"    -> WARNING: Skipping invalid attachment entry: {filename}")
            continue

        # Use regex to extract the Base64 part of the URI (after base64,)
        match = re.search(r"base64,(.*)", data_uri, re.IGNORECASE)
        if not match:
            print(f"    -> ERROR: Could not find base64 data in URI for {filename}")
            continue

        base64_data = match.group(1)
        file_path = os.path.join(task_dir, filename)

        try:
            # Decode the base64 string
            file_bytes = base64.b64decode(base64_data)
            
            # Write the raw bytes to the file
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            print(f"    -> Saved Attachment: {filename} (Size: {len(file_bytes)} bytes)")
            saved_files.append(filename)
            
        except Exception as e:
            print(f"    -> CRITICAL ERROR saving attachment {filename}: {e}")
            raise Exception(f"Failed to save attachment {filename} locally.")

    return saved_files
# --- Main Orchestration Logic ---


async def generate_files_and_deploy(task_data: TaskRequest):
    """
    The asynchronous background process that executes the main project workflow.
    It adapts the LLM prompt for multi-round tasks and fixes the cloning order.
    """
    task_id = task_data.task
    email = task_data.email         
    round_index = task_data.round 
    brief = task_data.brief
    evaluation_url = task_data.evaluation_url
    nonce = task_data.nonce
    attachments = task_data.attachments
    
    
    print(f"\n--- [PROCESS START] Starting background task for {task_id}, Round {round_index} ---")
    
    # Deployment configuration
    repo_name = task_id.replace(' ', '-').lower()
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    repo_url_auth = f"https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git"
    repo_url_http = f"https://github.com/{github_username}/{repo_name}"
    
    try:
        # 0. Setup local directory
        base_dir = os.path.join(os.getcwd(), "generated_tasks")
        local_path = os.path.join(base_dir, task_id)

        # --- ROBUST CLEANUP LOGIC ---
        # Crucial: Cleans up local directory before cloning or creating a new repo.
        if os.path.exists(local_path):
            print(f"--- [CLEANUP] Deleting existing local directory: {local_path} ---")
            
            def onerror(func, path, exc_info):
                """Error handler for shutil.rmtree to handle permission issues."""
                if exc_info[0] is PermissionError or 'WinError 5' in str(exc_info[1]):
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
                else:
                    raise

            try:
                shutil.rmtree(local_path, onerror=onerror)
                print("--- [CLEANUP] Directory deleted successfully. ---")
            except Exception as e:
                print(f"!!! CRITICAL: Failed to clean up directory. Error: {e}")
                raise Exception(f"Failed to perform local cleanup: {e}")
        
        # Create the fresh, EMPTY directory (ready for clone or init)
        os.makedirs(local_path, exist_ok=True)
        # --- END ROBUST CLEANUP ---
        
        # 1. SETUP REPO (Clone or Init)
        # MUST run before any files are saved to local_path.
        print(f"--- [DEPLOYMENT] Setting up local Git repository for Round {round_index}... ---")
        repo = await setup_local_repo(
            local_path=local_path, 
            repo_name=repo_name, 
            repo_url_auth=repo_url_auth, 
            repo_url_http=repo_url_http, 
            round_index=round_index
        ) 
        
        # 2. Process Attachments for LLM Input
        image_parts = []
        attachment_list_for_llm_prompt = []

        for attachment in attachments:
            # Check for image parts for LLM input
            if is_image_data_uri(attachment.url):
                gemini_part = data_uri_to_gemini_part(attachment.url)
                if gemini_part:
                    image_parts.append(gemini_part)
            
            # List all attachment names for the prompt
            attachment_list_for_llm_prompt.append(attachment.name)

        print(f"--- [LLM_INPUT] Found {len(image_parts)} image(s) to pass to LLM. ---")
        
        attachment_list_str = ", ".join(attachment_list_for_llm_prompt)
        
        # 3. AI Code Generation - Adapt Prompt for Round 2
        
        # --- MODIFICATION START: Adapting the LLM Prompt ---
        if round_index > 1:
            # For Round 2+, tell the LLM it's modifying existing work
            llm_prompt = (
                f"UPDATE INSTRUCTION (ROUND {round_index}): You must modify the existing project files "
                f"(index.html, README.md, LICENSE) based on this new brief: '{brief}'. "
                "You must replace all content in 'index.html', 'README.md', and 'LICENSE' with new, complete versions "
                "that implement the requested modifications. The 'index.html' must remain a single, complete, "
                "fully responsive HTML file using Tailwind CSS."
            )
        else:
            # For Round 1, generate a new application
            llm_prompt = (
                f"Generate a complete, single-file HTML web application to achieve the following: {brief}. "
                "Ensure your code is fully responsive, and uses Tailwind CSS. "
                "Provide the code for the main web app, a README.md, and an MIT LICENSE."
            )
        
        # Add attachment context if files were provided, regardless of round.
        if attachment_list_str:
             llm_prompt += f"\nAdditional context: The following files are available in the project root: {attachment_list_str}. "
             llm_prompt += f"Ensure your code references these files correctly (if applicable)."
        # --- MODIFICATION END ---
        
        # Call LLM
        generated_files = await call_llm_for_code(llm_prompt, task_id, image_parts)
        
        # 4. Save Generated Code Locally
        # This overwrites the cloned files (index.html, README.md, LICENSE)
        await save_generated_files_locally(task_id, generated_files)
        
        # 5. Save Attachments Locally
        # This adds attachments (like data.csv) to the local directory
        # The attachment saving now happens *after* the clone/init, resolving the Round 2 error.
        await save_attachments_locally(local_path, attachments)

        # 6. COMMIT AND PUBLISH
        print(f"--- [DEPLOYMENT] Committing and Publishing task {task_id}, Round {round_index} to GitHub... ---")
        
        deployment_info = await commit_and_publish(
            repo=repo, 
            task_id=task_id,
            round_index=round_index,
            repo_name=repo_name
        )
        
        repo_url = deployment_info["repo_url"]
        commit_sha = deployment_info["commit_sha"]
        pages_url = deployment_info["pages_url"] 
        
        print(f"--- [DEPLOYMENT] Success! Repo: {repo_url}, Pages: {pages_url} ---")
        
        # 7. Notify the Evaluation Server
        await notify_evaluation_server(
            evaluation_url=evaluation_url, 
            email=email,
            task_id=task_id, 
            round_index=round_index,
            nonce=nonce, 
            repo_url=repo_url,
            commit_sha=commit_sha,
            pages_url=pages_url
        )

    except Exception as e:
        print(f"--- [CRITICAL FAILURE] Task {task_id} failed during processing: {e} ---")
        
    print(f"--- [PROCESS END] Background task for {task_id} completed. ---")


# --- FastAPI Endpoint ---

@app.post("/ready", status_code=200)
async def receive_task(task_data: TaskRequest):
    """
    API endpoint that receives the task payload. 
    It verifies the secret and starts the generation/deployment process in the background.
    """
    global received_task_data
    
    # 1. SECRET VERIFICATION (CRITICAL PROJECT REQUIREMENT)
    if not verify_secret(task_data.secret):
        print(f"--- FAILED SECRET VERIFICATION for task {task_data.task} ---")
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: Secret does not match configured student secret."
        )

    # Store data and print initial confirmation
    received_task_data = task_data.dict()
    
    print("--- TASK RECEIVED SUCCESSFULLY ---")
    print(f"Task ID: {received_task_data['task']}, Round: {received_task_data['round']}")
    
    # Start the processing function in the background 
    asyncio.create_task(generate_files_and_deploy(task_data))

    # Respond immediately with 200 OK to the evaluation server
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "message": f"Task {task_data.task} received and processing started."}
    )

@app.get("/")
async def root():
    return {"message": "Task Receiver Service is running. Post to /ready to submit a task."}

@app.get("/status")
async def get_status():
    global received_task_data
    if received_task_data:
        # Note: This status only shows the last received request, not the live status of the background task.
        return {"last_received_task": received_task_data}
    else:
        return {"message": "Awaiting first task submission to /ready"}