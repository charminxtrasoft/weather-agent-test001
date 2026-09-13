import os
import sys
from groq import Groq, GroqError

# Retrieve API key from environment variables
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    print("Error: GROQ_API_KEY environment variable is not set.")
    print("Please set it in your environment: export GROQ_API_KEY='your_key_here'")
    sys.exit(1)

# Initialize the Groq SDK client
client = Groq(api_key=API_KEY)


def fetch_accessible_models() -> list[str]:
    """Retrieves all model IDs currently active and accessible with your API key."""
    try:
        models_data = client.models.list()
        # Sort retrieved model IDs for consistent ordering
        available_ids = sorted([model.id for model in models_data.data])
        return available_ids
    except GroqError as err:
        print(f"Failed to query model list from Groq API: {err}")
        return []


def run_chat_completion(model_id: str, prompt: str) -> str | None:
    """Executes a chat completion request for a specified model ID."""
    print(f" -> Testing model: '{model_id}'...")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise, helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_completion_tokens=256,
        )
        return response.choices[0].message.content
    except GroqError as err:
        print(f"    [X] Model '{model_id}' failed: {err}")
        return None


def main():
    test_prompt = "Explain why an API returns a 404 status code in two concise sentences."

    print("=== Step 1: Discovering Available Groq Models ===")
    accessible_models = fetch_accessible_models()

    if not accessible_models:
        print("No accessible models found for your API key. Exiting.")
        return

    print(f"Retrieved {len(accessible_models)} active model(s):")
    for model_name in accessible_models:
        print(f"  • {model_name}")

    print("\n=== Step 2: Executing Completion with Automatic Fallback ===")

    # Priority queue of production models
    preferred_models = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ]

    # Intersect preferred models with active account models to avoid 404 errors
    execution_queue = [m for m in preferred_models if m in accessible_models]

    # Append any remaining available account models as secondary fallbacks
    for m in accessible_models:
        if m not in execution_queue:
            execution_queue.append(m)

    # Attempt execution down the queue until one succeeds
    completion_result = None
    successful_model = None

    for model_id in execution_queue:
        completion_result = run_chat_completion(model_id, test_prompt)
        if completion_result:
            successful_model = model_id
            break

    # Print results
    print("\n=== Step 3: Execution Result ===")
    if completion_result and successful_model:
        print(f"Successfully generated response using: [{successful_model}]\n")
        print("Response:")
        print(completion_result)
    else:
        print("Could not complete the request with any available model.")


if __name__ == "__main__":
    main()
