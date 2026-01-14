# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from transformers import AutoTokenizer

model_predownload_path = os.environ.get("MODEL_PREDOWNLOAD_PATH")
if not model_predownload_path:
    print("Warning: MODEL_PREDOWNLOAD_PATH not set, skipping tokenizer pre-download")
    sys.exit(0)

tokenizer_name = os.environ.get("PREDOWNLOAD_TOKENIZER", "")

if not tokenizer_name:
    print("Info: PREDOWNLOAD_TOKENIZER not set, skipping tokenizer pre-download")
    print(
        "      For air-gap deployment, use: --build-arg PREDOWNLOAD_TOKENIZER=<model-name>"
    )
    sys.exit(0)

# Create a safe directory name from the tokenizer name
tokenizer_dir_name = tokenizer_name.replace("/", "-")
tokenizer_path = os.path.join(
    model_predownload_path, f"{tokenizer_dir_name}/tokenizer/"
)
os.makedirs(tokenizer_path, exist_ok=True)

try:
    print(f"Pre-downloading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(tokenizer_path)
    print(
        f"Successfully pre-downloaded tokenizer '{tokenizer_name}' to {tokenizer_path}"
    )
except Exception as e:
    print(f"Warning: Failed to pre-download tokenizer '{tokenizer_name}': {e}")
    sys.exit(0)  # Exit with success to allow build to continue
