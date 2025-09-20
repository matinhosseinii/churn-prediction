import os
import sys

# Suppress TensorFlow's informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

try:
    import tensorflow as tf
except ImportError:
    print("❌ Error: TensorFlow is not installed. Please check your environment.")
    sys.exit(1) # Exit with an error

print(f"✅ TensorFlow imported successfully (Version: {tf.__version__})")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU is available: {gpus[0].name}")
else:
    print("⚠️ No GPU found. TensorFlow will use the CPU.")

# Perform a test computation
try:
    tf.reduce_sum(tf.random.normal([1000, 1000]))
    print("✅ Test computation successful.")
except Exception as e:
    print(f"❌ Test computation failed: {e}")