import os
import sys
import base64
import asyncio
import argparse
from pathlib import Path
from itertools import cycle
from typing import Optional
import httpx
from dotenv import load_dotenv

#config
load_dotenv(Path(__file__).parent / '.env')

def _require_key_(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        print(f"Error: Missing API Key: '{name}' is not set in .env file.")
        sys.exit(1)
    return value

API_KEYS: list[str] = [_require_key_(f"GEMINI_KEY_{i}") for i in range(1, 7)]

GEMINI_API_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models/",
        "gemini-1.5-flash:generateContent"
)

PROMPT = (
    "Examine the following image CAREFULLY. "
    "Determine if this is an image of a galaxy (e.g elliptical, spiral, irregular). "
    "Reply with EXACTLY one of the following options (DO NOT ADD EXTRA EXPLANATION): \n"
    "GALAXY - if the image shows a galaxy/galaxies. \n"
    "NOT GALAXY - if the image does NOT show a galaxy/galaxies. \n"
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

#helpers
def load_image_b64(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp"
    }
    mime = mime_map.get(ext, "image/jpeg")
    data = base64.standard_b64decode(path.read_bytes()).decode("utf-8")
    return mime, data

def assign_keys(image_paths: list[Path]) -> list[tuple[Path, str]]:
    key_cycle = cycle(API_KEYS)
    return [(img, next(key_cycle)) for img in image_paths]

async def classify_image(
    client: httpx.AsyncClient,
    image_path: Path,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        b64_data, mime_type = load_image_b64(image_path)
        
        payload = {
            "content":[
                {
                    "parts": [
                        {"text": PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data,
                            }
                        },
                    ]
                }
            ]
        }
        
        try:
            resp = await client.post(
                GEMINI_API_URL,
                params={"key": api_key},
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            body = resp.json()
            text = (
                body["candidates"][0]["content"]["parts"][0]["text"].strip()
            )
            is_galaxy = text.upper().startswith("GALAXY")
            return{
                "file": str(image_path),
                "api_key": f"...{api_key[-6:]}",
                "is_galaxy": is_galaxy,
                "response": text,
                "error": None,
            }
        except Exception as exc:
            return {
                "file": str(image_path),
                "api_key": f"...{api_key[-6:]}",
                "is_galaxy": None,
                "response": None,
                "error": str(exc),
            }
            
async def run_batch(image_paths: list[Path], max_concurrent: int = 12) -> list[dict]:
    assignments = assign_keys(image_paths)
    semaphore = asyncio.Semaphore(max_concurrent)
    async with httpx.AsyncClient() as client:
        tasks = [
            classify_image(client, img, key, semaphore) 
            for img, key in assignments
        ]
        results = await asyncio.gather(*tasks)
        
    return list(results)

def print_results(results: list[dict]) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"{'GALAXY CLASSIFICATION RESULTS':^{width}}")
    print("=" * width)
    
    galaxies, not_galaxies, errors = [], [], []
    
    for r in results:
        fname = Path(r["file"]).name
        key_hint = r["api_key"]
        
        if r["error"]:
            errors.append(r)
            print(f"  ✗  {fname:<35} [key {key_hint}]  ERROR: {r['error']}")
        elif r["is_galaxy"]:
            galaxies.append(r)
            print(f"  ✓  {fname:<35} [key {key_hint}]  GALAXY")
            print(f"        {r['response'].splitlines()[-1] if r['response'] else ''}")
        else:
            not_galaxies.append(r)
            print(f"  ✗  {fname:<35} [key {key_hint}]  NOT GALAXY")
            print(f"        {r['response'].splitlines()[-1] if r['response'] else ''}")

    print("=" * width)
    print(f"Total Images: {len(results)}")
    print(f"Galaxies: {len(galaxies)}")
    print(f"Not Galaxies: {len(not_galaxies)}")
    print(f"Errors: {len(errors)}")
    
    from collections import Counter
    key_counts = Counter(r["api_key"] for r in results)
    print("\n Loading distribution across API keys:")
    for key, count in sorted(key_counts.items()):
        bar = "█" * count
        print(f"  Key {key} | {bar} ({count})")
    print("=" * width + "\n")

#CLI
def collect_images(inputs: list[str]) -> list[Path]:
    images: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file():
            for ext in EXTENSIONS:
                images.extend(p.glob(f"*{ext}"))
                images.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in EXTENSIONS:
            images.append(p)
        else:
            print(f"[Warning] Skipping '{item}': Not a valid file or directory.")
    return sorted(set(images))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Galaxy Image Classifier using Gemini API (6 keys, batch processing, async)"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths to image files or directories containing images.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="Maximum number of concurrent API requests (default: 12).",
    )
    args = parser.parse_args()
    
    images = collect_images(args.inputs)
    if not images:
        print("[Error] No valid images found in the provided paths.")
        sys.exit(1)
        
    print(f"Found {len(images)} images to classify. Starting batch processing...")
    results = asyncio.run(run_batch(images, max_concurrent=args.concurrency))
    print_results(results)

if __name__ == "__main__":
    main()