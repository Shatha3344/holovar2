# Import environment variables before any other imports
import os
# Limit OpenBLAS and TensorFlow threads to avoid resource exhaustion
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from app import app  # noqa: F401

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

