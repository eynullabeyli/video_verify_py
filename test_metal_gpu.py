#!/usr/bin/env python3
"""
Test script to verify Metal GPU detection and configuration.
"""

def test_metal_gpu():
    print("=== Metal GPU Test ===")
    
    # Test TensorFlow Metal import
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
        
        # Check for GPU devices (Metal support)
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"✓ Metal GPU support detected - Found {len(gpu_devices)} GPU device(s)")
        else:
            print("ℹ No GPU devices found - Metal GPU not available")
            
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    # Test GPU device detection
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        cpu_devices = tf.config.list_physical_devices('CPU')
        
        print(f"✓ Found {len(gpu_devices)} GPU device(s) and {len(cpu_devices)} CPU device(s)")
        
        for i, device in enumerate(gpu_devices):
            print(f"  GPU {i}: {device.name}")
        
        for i, device in enumerate(cpu_devices):
            print(f"  CPU {i}: {device.name}")
            
    except Exception as e:
        print(f"✗ GPU device detection failed: {e}")
        return False
    
    # Test TensorFlow version
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Test simple GPU operation
    try:
        if gpu_devices:
            print("✓ Testing GPU computation...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                result = c.numpy()
                print(f"✓ GPU computation successful: {result}")
        else:
            print("ℹ No GPU devices available for computation test")
            
    except Exception as e:
        print(f"✗ GPU computation test failed: {e}")
        return False
    
    print("=== Metal GPU Test Complete ===")
    return True

if __name__ == "__main__":
    test_metal_gpu()
