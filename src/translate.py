"""
HinglishDB — Translation Pipeline
Translates BIRD English questions to Hinglish
using Llama-3.1-8B-Instant via Groq API.
"""

import json, os, time
from groq import Groq

def translate_to_hinglish(question: str, client: Groq, max_retries: int = 5) -> str:
    """Translates a single English question to Hinglish."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Translate English database questions to Hinglish "
                            "(Roman script only, no Devanagari). Use natural Hindi "
                            "grammar with English technical terms. Return translation only.\n"
                            "Examples:\n"
                            "EN: How many students scored above 90? "
                            "-> HI: Kitne students ne 90 se zyada score kiya?\n"
                            "EN: List employees ordered by salary. "
                            "-> HI: Employees ko salary ke order mein list karo."
                        )
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = (2 ** attempt) + 1
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  All retries failed for: {question[:50]}")
                return ""

def translate_split(data: list, split_name: str, output_dir: str,
                    client: Groq, save_every: int = 50) -> list:
    """Translates a full split with checkpointing."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = f"{output_dir}/{split_name}_checkpoint.json"
    final_path = f"{output_dir}/{split_name}_hinglish.json"

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming {split_name} from index {start_idx}/{len(data)}")
    else:
        results, start_idx = [], 0

    for i in range(start_idx, len(data)):
        row = data[i]
        hinglish_q = translate_to_hinglish(row["question"], client)
        result = dict(row)
        result["hinglish_question"] = hinglish_q
        result["question_en"] = row["question"]
        results.append(result)

        if (i - start_idx + 1) % 25 == 0:
            print(f"  {i+1}/{len(data)} translated")
        if (i - start_idx + 1) % save_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(0.1)

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"✅ {split_name}: {len(results)} pairs saved to {final_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing train_en.json, dev_en.json, test_en.json")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save translated files")
    parser.add_argument("--api_key", default=None,
                        help="Groq API key (or set GROQ_API_KEY env variable)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Provide --api_key or set GROQ_API_KEY environment variable")

    client = Groq(api_key=api_key)

    for split in ["train", "dev", "test"]:
        path = f"{args.input_dir}/{split}_en.json"
        if not os.path.exists(path):
            print(f"Skipping {split} — file not found at {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        translate_split(data, split, args.output_dir, client)
