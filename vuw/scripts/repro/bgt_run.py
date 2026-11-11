#!/usr/bin/env python
"""bgt-ada's run_pipeline.py with one deviation: detection batch 8 instead of 32 (32 VOC images through their
ResNet50-FPN-v2 need > 23.5 GB, more than one RTX A5000). Batch composition only changes the zero padding inside a batch."""
import sys

sys.path.insert(0, ".")
from config import datasets                                                     # noqa: E402

datasets.DATASETS["voc"].detection_batch_size = 8
import run_pipeline                                                            # noqa: E402

sys.exit(run_pipeline.main())
