import flask
import transformers
import huggingface_hub
import torch
import pandas as pd
import PIL
import base64
import flask_cors

# Check versions
print(f"Flask version: {flask.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Huggingface Hub version: {huggingface_hub.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"PIL version: {PIL.__version__}")
print(f"Base64 module (built-in) does not have a version.")
print(f"Flask-CORS version: {flask_cors.__version__}")
