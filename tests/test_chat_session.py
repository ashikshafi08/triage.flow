import requests
import time

# Configuration
BASE_URL = "http://localhost:8000"
ISSUE_URL = "https://github.com/huggingface/trl/issues/3466"
PROMPT_TYPE = "explain"

def test_chat_session():
    # Step 1: Create session
    session_res = requests.post(
        f"{BASE_URL}/sessions",
        json={
            "issue_url": ISSUE_URL,
            "prompt_type": PROMPT_TYPE,
            "llm_config": {
                "provider": "openrouter",
                "name": "openai/o4-mini-high"
            }
        }
    )
    if session_res.status_code != 200:
        print(f"Error creating session: {session_res.text}")
        return
        
    session_data = session_res.json()
    session_id = session_data.get("session_id")
    initial_message = session_data.get("initial_message")
    
    if not session_id or not initial_message:
        print(f"Invalid response: {session_data}")
        return
        
    print(f"Created session: {session_id}")
    print(f"Initial message: {initial_message[:100]}...")
    
    # Wait for context initialization
    time.sleep(2)
    
    # Step 2: Send follow-up message
    message_res = requests.post(
        f"{BASE_URL}/sessions/{session_id}/messages",
        json={"role": "user", "content": "Show me the relevant code"}
    )
    message_data = message_res.json()
    print(f"Assistant response: {message_data['content'][:100]}...")
    
    # Step 3: Send another message
    message_res = requests.post(
        f"{BASE_URL}/sessions/{session_id}/messages",
        json={"role": "user", "content": "How should I fix this?"}
    )
    message_data = message_res.json()
    print(f"Assistant response: {message_data['content'][:100]}...")

if __name__ == "__main__":
    test_chat_session()
