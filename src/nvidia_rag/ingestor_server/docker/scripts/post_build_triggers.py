import os
import sys

from transformers import AutoTokenizer

model_predownload_path = os.environ.get("MODEL_PREDOWNLOAD_PATH")
if not model_predownload_path:
    print("Warning: MODEL_PREDOWNLOAD_PATH not set, skipping tokenizer pre-download")
    sys.exit(0)

tokenizer_path = os.path.join(
    model_predownload_path, "e5-large-unsupervised/tokenizer/"
)
os.makedirs(tokenizer_path, exist_ok=True)

try:
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-unsupervised")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Successfully pre-downloaded tokenizer to {tokenizer_path}")
except Exception as e:
    print(f"Warning: Failed to pre-download tokenizer: {e}")
    sys.exit(0)  # Exit with success to allow build to continue
