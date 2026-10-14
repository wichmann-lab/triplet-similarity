import os
import sys

# scripts/ has no package __init__, so its modules are imported as siblings.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
