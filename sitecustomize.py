"""
NutriVision AI — Environment Auto-Initialization
Ensures compatibility patches are loaded across all submodules.
"""
try:
    import src.utils.torch_patch
except Exception:
    pass
