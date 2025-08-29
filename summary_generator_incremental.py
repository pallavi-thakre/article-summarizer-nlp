import os
import json
import time
import requests
import argparse

# --- Configuration ---
# It's recommended to set your API key as an environment variable
# for better security, rather than hardcoding it in the script.
# You can set it in your terminal like this:
# export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

def generate_summary(content: str, retries: int = 3, delay: int = 5) -> str:
    """
    Calls the Gemini API to generate a summary for the given text,
    with exponential backoff for retries.

    Args:
        content: The text content to be summarized.
        retries: The number of times to retry the API call on failure.
        delay: The initial delay (in seconds) between retries.

    Returns:
        The generated summary string, or a descriptive error message if it fails.
    """
    # Truncate content if it's too long to avoid overly large API requests
    max_content_length = 10000 
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."

    prompt = (
        "Please provide a one or two sentence, high-quality summary of the "
        "following text. The summary should be neutral and informative.\n\n"
        f'TEXT:\n"""\n{content}\n"""\n\nSUMMARY:'
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2000}
    }

    current_delay = delay
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Safely get the first candidate, or an empty dictionary if none exists.
            candidate = result.get('candidates', [{}])[0]
            
            # Check if the generation was stopped for a reason other than finishing normally.
            finish_reason = candidate.get('finishReason')
            if finish_reason and finish_reason != "STOP":
                return f"Error: Generation stopped due to {finish_reason}."

            # Check if the expected content structure exists and contains text.
            if (candidate.get('content') and 
                candidate['content'].get('parts') and
                candidate['content']['parts'][0].get('text')):
                
                summary_text = candidate['content']['parts'][0]['text'].strip()
                
                # Check if the summary is not just an empty or whitespace string.
                if summary_text:
                    return summary_text
                else:
                    return "Error: API returned an empty summary."
            else:
                # This handles cases where the response structure is unexpected.
                return "Error: Invalid response structure (e.g., missing text part)."

        except requests.exceptions.RequestException as e:
            print(f"Error making API call: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff
            else:
                return f"Error: Could not connect to API after {retries} attempts."
    
    return "Error: Failed to generate summary after all retries."


def process_json_file(input_path: str, output_path: str):
    """
    Reads a JSON file, generates summaries for each item, and writes the
    results to a new file, resuming from where it left off if the output
    file already exists.

    Args:
        input_path: Path to the input JSON file.
        output_path: Path to save the output JSON file with summaries.
    """
    # --- MODIFICATION V1: Load all input data first ---
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            all_items = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_path}'. Please check the file format.")
        return

    if not isinstance(all_items, list):
        print("Error: Input JSON must be an array of objects.")
        return

    # --- MODIFICATION V2: Check for existing output file to resume ---
    processed_items = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                # Check if the file is not empty before trying to load
                if os.path.getsize(output_path) > 0:
                    processed_items = json.load(f)
                    if not isinstance(processed_items, list):
                       print(f"Warning: Output file '{output_path}' is not a valid list. Starting fresh.")
                       processed_items = []
        except json.JSONDecodeError:
            print(f"Warning: Output file '{output_path}' is corrupted or incomplete. Starting fresh.")
            processed_items = []
        except Exception as e:
            print(f"An unexpected error occurred reading the output file: {e}. Starting fresh.")
            processed_items = []

    # --- MODIFICATION V3: Determine which items still need to be processed ---
    processed_titles = {item.get("Title") for item in processed_items if "Title" in item}
    items_to_process = [item for item in all_items if item.get("Title") not in processed_titles]

    if not items_to_process:
        print("All items have already been processed. The output file is up to date.")
        return

    total_new_items = len(items_to_process)
    print(f"Found {len(processed_items)} already processed items.")
    print(f"Starting to process {total_new_items} new items.")

    # --- MODIFICATION V4: Process new items and save progress after each one ---
    try:
        for i, item in enumerate(items_to_process):
            title = item.get("Title", "Untitled")
            content = item.get("Content")

            print(f"\nProcessing new item {i + 1}/{total_new_items}: '{title}'...")

            if not content:
                print("  -> Skipping, no 'Content' field found.")
                summary = "Error: No content provided."
            else:
                summary = generate_summary(content)
                print(f"  -> Result for summary: '{summary}'")

            new_item = item.copy()
            new_item["Summary"] = summary
            
            # Add the newly processed item to our list
            processed_items.append(new_item)
            
            # Save the entire updated list to the output file.
            # This is safer than appending and ensures the file is always valid JSON.
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(processed_items, outfile, indent=2, ensure_ascii=False)
            
            # A small delay to be respectful to the API rate limits
            time.sleep(1)

        print(f"\nSuccessfully processed all {total_new_items} new items.")
        print(f"Results saved to '{output_path}'")
        
    except IOError as e:
        print(f"\nError writing to output file '{output_path}': {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Batch generate summaries for content in a JSON file, with resume capability."
    )
    parser.add_argument(
        "input_file", 
        help="Path to the input JSON file (e.g., op2.json)."
    )
    parser.add_argument(
        "-o", "--output_file", 
        help="Path for the output JSON file. Defaults to 'output.json'.",
        default="output.json"
    )
    
    args = parser.parse_args() 

    if not API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set the environment variable before running the script.")
        return

    process_json_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
