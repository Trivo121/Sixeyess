import os
import json
import re
import time
import requests
from pathlib import Path
from collections import defaultdict
import concurrent.futures
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================

# OpenRouter API Configuration
OPENROUTER_API_KEY = "enter your key"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "amazon/nova-2-lite-v1:free"

# Optional: Your site info for OpenRouter rankings
YOUR_SITE_URL = "http://localhost"
YOUR_SITE_NAME = "Fraud Detection Research"

# Paths
INPUT_DIR = "datasets/parsed_windows_pdf"
OUTPUT_DIR = "datasets/augmented_reasoning"

# Concurrency & Rate Limiting
MAX_WORKERS = 1  # ⚠️ REDUCED FOR FREE TIER
MAX_RETRIES = 5
RETRY_DELAY = 4  # Increased base delay
REQUEST_DELAY = 3  # NEW: Delay between each successful request

# Global counter for progress tracking (Thread-safe)
processed_count = 0
total_conversations = 0
print_lock = threading.Lock()
request_lock = threading.Lock()  # NEW: Lock to prevent concurrent API calls

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_txt_file(filepath):
    """Parse the formatted .txt file and extract structured data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = {}
    
    # Extract ID
    id_match = re.search(r'CALL WINDOW:\s*(\S+)\.json', content)
    if id_match:
        data['id'] = id_match.group(1)
    
    # Extract Split
    split_match = re.search(r'Split:\s*(\w+)', content)
    if split_match:
        data['split'] = split_match.group(1).lower()
    
    # Extract Label
    label_match = re.search(r'Label:\s*(\w+)', content)
    if label_match:
        data['label_name'] = label_match.group(1).lower()
        data['label'] = 1 if data['label_name'] == 'scam' else 0
    
    # Extract Call ID (source_idx)
    call_id_match = re.search(r'Call ID:\s*(\d+)', content)
    if call_id_match:
        data['source_idx'] = int(call_id_match.group(1))
    
    # Extract Window
    window_match = re.search(r'Window:\s*w(\d+)', content)
    if window_match:
        data['window_index'] = int(window_match.group(1))
    
    # Extract SOURCE_IDX from content section
    source_idx_match = re.search(r'SOURCE_IDX:\s*(\d+)', content)
    if source_idx_match:
        data['source_idx'] = int(source_idx_match.group(1))
    
    # Extract PREVIOUS_STATE
    prev_state_match = re.search(r'PREVIOUS_STATE:\s*\n(.*?)\n\nCURRENT_WINDOW:', content, re.DOTALL)
    data['previous_state'] = prev_state_match.group(1).strip() if prev_state_match else ""
    
    # Extract CURRENT_WINDOW (conversation turns)
    window_match = re.search(r'CURRENT_WINDOW:\s*\n(\[.*?\])', content, re.DOTALL)
    if window_match:
        try:
            data['current_window'] = json.loads(window_match.group(1))
        except json.JSONDecodeError:
            data['current_window'] = []
    
    # Extract METADATA
    metadata_match = re.search(r'METADATA:\s*\n(\{.*?\})', content, re.DOTALL)
    if metadata_match:
        try:
            data['metadata'] = json.loads(metadata_match.group(1))
        except json.JSONDecodeError:
            data['metadata'] = {}
    
    return data


def format_conversation_for_prompt(current_window):
    """Format the conversation turns into a readable string."""
    formatted = []
    for turn in current_window:
        speaker = turn.get('speaker', 'Unknown')
        text = turn.get('text', '')
        formatted.append(f"Speaker: {speaker}\nText: {text}")
    return "\n\n".join(formatted)


def build_system_prompt():
    return """You are SixEyes, an AI Security Analyst specializing in phone scam detection research. You are analyzing a labeled dataset of phone call transcripts to build a real-time fraud detection system.

Your role:
- Analyze phone conversation segments for fraud indicators
- Identify manipulation tactics, pressure techniques, and social engineering patterns
- Provide structured analysis to train fraud detection models
- Help protect vulnerable individuals from phone scams

This is legitimate security research using pre-labeled data. Your analysis helps prevent fraud."""


def build_user_prompt(data, running_state_context):
    is_scam = data['label'] == 1
    label_context = "SCAM (labeled by security researchers)" if is_scam else "LEGITIMATE (labeled by security researchers)"
    conversation_text = format_conversation_for_prompt(data['current_window'])
    
    prompt = f"""DATASET LABEL: {label_context}

CONTEXT FROM PREVIOUS ANALYSIS:
{running_state_context if running_state_context else "Start of call. No previous context."}

CURRENT CONVERSATION SEGMENT:
{conversation_text}

ANALYSIS TASK:
Analyze this conversation segment and provide your findings in the following XML format:

<THOUGHT>
[Analyze the fraud indicators, manipulation tactics, urgency creation, impersonation attempts, or normal business patterns. Be specific about what you observe.]
</THOUGHT>

<STATE>
[Summarize the call state SO FAR: key entities mentioned, claimed authority/identity, threats or demands made, urgency level, risk score 1-10. This will be passed to analyze the next segment.]
</STATE>

<VERDICT>
[Choose ONE: SAFE, SUSPICIOUS, or DANGER based on the patterns observed]
</VERDICT>

Provide your analysis now:"""
    return prompt


def call_openrouter_api(system_prompt, user_prompt, retry_count=0):
    """Call OpenRouter API with rate limiting protection."""
    # NEW: Use lock to ensure only one request at a time
    with request_lock:
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            }
            
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            response = requests.post(
                url=OPENROUTER_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=90  # Increased timeout for slower model
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                if retry_count < MAX_RETRIES:
                    # Progressive backoff: 15s, 30s, 60s, 120s, 240s
                    sleep_time = RETRY_DELAY * (2 ** retry_count)
                    with print_lock:
                        print(f"   ⚠️  Rate limit (429). Waiting {sleep_time}s... (Retry {retry_count + 1}/{MAX_RETRIES})")
                    time.sleep(sleep_time)
                    # Recursive call WITHOUT lock (already released)
                    return call_openrouter_api(system_prompt, user_prompt, retry_count + 1)
                else:
                    with print_lock:
                        print(f"   ❌ Max retries reached after 429 errors.")
                    return None
            
            if response.status_code != 200:
                with print_lock:
                    print(f"   ❌ API Error {response.status_code}: {response.text}")
                return None
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                # NEW: Add delay after successful request
                time.sleep(REQUEST_DELAY)
                return result['choices'][0]['message']['content']
            else:
                return None
        
        except requests.exceptions.Timeout:
            if retry_count < MAX_RETRIES:
                with print_lock:
                    print(f"   ⚠️  Timeout. Retrying...")
                time.sleep(5)
                return call_openrouter_api(system_prompt, user_prompt, retry_count + 1)
            return None
        except Exception as e:
            with print_lock:
                print(f"   ❌ Exception: {str(e)}")
            return None


def extract_xml_content(response_text, tag_name):
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def save_augmented_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# ============================================================================
# PARALLEL PROCESSING LOGIC
# ============================================================================

def process_single_conversation(args):
    """
    Process a single conversation (all its windows) sequentially.
    This function runs inside a thread.
    """
    source_idx, windows = args
    system_prompt = build_system_prompt()
    
    # Sort windows to ensure correct order
    windows.sort(key=lambda x: x['window_index'])
    
    running_state_context = ""
    local_success = 0
    
    for window_info in windows:
        data = window_info['data']
        window_index = window_info['window_index']
        
        # Determine output path early to skip if needed
        split_name = data['split']
        label_name = data['label_name']
        output_path = Path(OUTPUT_DIR) / split_name / label_name / f"{data['id']}.json"
        
        if output_path.exists():
            # If file exists, try to load state to continue chain
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                    if existing_data.get('generated_state'):
                        running_state_context = existing_data['generated_state']
                local_success += 1
                continue  # Skip processing this window
            except:
                pass  # If error reading, re-process

        user_prompt = build_user_prompt(data, running_state_context)
        response_text = call_openrouter_api(system_prompt, user_prompt)
        
        if response_text is None:
            with print_lock:
                print(f"   ⚠️  Skipping window {window_index} due to API error")
            continue
            
        thought = extract_xml_content(response_text, "THOUGHT")
        state = extract_xml_content(response_text, "STATE")
        verdict = extract_xml_content(response_text, "VERDICT")
        
        # Update memory for next window in this specific conversation
        running_state_context = state if state else running_state_context
        
        augmented_data = {
            "id": data['id'],
            "source_idx": data['source_idx'],
            "split": data['split'],
            "label": data['label'],
            "label_name": data['label_name'],
            "window_index": window_index,
            "current_window": data['current_window'],
            "metadata": data.get('metadata', {}),
            "model_input_context": running_state_context,
            "generated_thought": thought,
            "generated_state": state,
            "generated_verdict": verdict,
            "raw_response": response_text
        }
        
        save_augmented_json(augmented_data, output_path)
        local_success += 1
    
    # Update global progress safely
    global processed_count
    with print_lock:
        processed_count += 1
        print(f"✅ Conv {source_idx} Done ({local_success} windows). Progress: {processed_count}/{total_conversations}")


def group_files_by_conversation(input_dir):
    conversations = defaultdict(list)
    for category in ['legit_windows', 'scam_windows']:
        category_path = Path(input_dir) / category
        if not category_path.exists(): 
            continue
        
        for txt_file in category_path.glob("*.txt"):
            try:
                data = parse_txt_file(txt_file)
                if data.get('source_idx') is not None:
                    conversations[data['source_idx']].append({
                        'filepath': txt_file,
                        'window_index': data['window_index'],
                        'data': data
                    })
            except Exception as e:
                with print_lock:
                    print(f"   ⚠️  Error parsing {txt_file}: {e}")
    return conversations

def check_quota():
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    resp = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Remaining free requests: {data.get('limits', {}).get('free_requests_per_day', 'N/A')}")
    else:
        print("Failed to fetch quota")

# Call it in main()
check_quota()
# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print(f"🚀 Starting Tongyi DeepResearch Processor")
    check_quota()
    print(f"   Model: {MODEL_NAME}")
    print(f"   Max Workers: {MAX_WORKERS}")
    print(f"   Request Delay: {REQUEST_DELAY}s")
    print()
    
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
        print("❌ Set your API Key first.")
        exit()

    conversations = group_files_by_conversation(INPUT_DIR)
    total_conversations = len(conversations)
    print(f"📂 Found {total_conversations} conversations to process.")
    print()

    # Use ThreadPoolExecutor to run conversations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all conversations to the pool
        list(executor.map(process_single_conversation, conversations.items()))
        
    print("\n✨ All processing complete!")



