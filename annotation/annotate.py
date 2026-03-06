"""
Runs batch annotation over a sampled dataset.
Safe to interrupt and resume.
"""

import pandas as pd
import time
from cards_client import classify_text
from utils import ensure_directory, save_incremental


INPUT_PATH = "../data/processed/pilot_sample.parquet"
OUTPUT_PATH = "../data/annotated/annotations.parquet"
TEXT_COLUMN = "ArticleText"
ID_COLUMN = "ArticleKey"

def run_annotation():

    # Ensure output directory exists
    ensure_directory("../data/annotated/")

    # Load data
    df = pd.read_parquet(INPUT_PATH)

    # If previous results exist, resume from there
    try:
        existing = pd.read_parquet(OUTPUT_PATH)
        processed_ids = set(existing[ID_COLUMN])
        results = existing.to_dict("records")
        print(f"Resuming from {len(processed_ids)} already processed articles.")
    except FileNotFoundError:
        processed_ids = set()
        results = []
        print("Starting annotation")

    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, row in df.iterrows():

        article_id = row[ID_COLUMN]

        # Skip already processed articles
        if article_id in processed_ids:
            continue

        print(f"Processing article {i+1}/{len(df)}")

        try:
            output = classify_text(row[TEXT_COLUMN])

            category_numbers = [c.category_number for c in output["categories"]]

            total_prompt_tokens += output["prompt_tokens"]
            total_completion_tokens += output["completion_tokens"]

            results.append({
                ID_COLUMN: article_id,
                "categories": category_numbers,
                "prompt_tokens": output["prompt_tokens"],
                "completion_tokens": output["completion_tokens"]
            })

            # Save after each article (important!)
            save_incremental(results, OUTPUT_PATH)

            # Small pause to stay safe under rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"Error processing article {article_id}: {e}")

    print("Annotation complete.")
    print("Total prompt tokens:", total_prompt_tokens)
    print("Total completion tokens:", total_completion_tokens)


if __name__ == "__main__":
    run_annotation()