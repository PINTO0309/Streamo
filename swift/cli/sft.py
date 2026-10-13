# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from pathlib import Path

# File-based torchrun workers start with swift/cli, not the checkout root, on sys.path.
# Resolve the root from this entrypoint so sibling packages work before reinstalling.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if int(os.environ.get('UNSLOTH_PATCH_TRL', '0')) != 0:
    import unsloth

from swift.llm import sft_main

if __name__ == '__main__':
    sft_main()
