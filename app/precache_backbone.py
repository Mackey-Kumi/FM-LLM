"""
Pre-compute Llama backbone outputs for all test windows.
Run once: python -m app.precache_backbone
Takes ~10-20 minutes on CPU, then inference is instant.
"""

from .inference import build_backbone_cache

if __name__ == "__main__":
    build_backbone_cache()