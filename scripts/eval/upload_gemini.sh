#!/bin/bash

base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 "$base_path/cyclist/eval/utils/upload_gemini_vid.py"
