#!/usr/bin/env python3
"""
Performance test script for video verification with Metal GPU optimization.
"""

import time
import os
from app.services.video_verification_service import video_verification

def test_performance():
    print("=== Performance Test with Metal GPU Optimization ===")
    
    # Test file paths (adjust these to your actual test files)
    video_path = "/Users/yusif/Desktop/assets/testpurpose/yusif.mp4"
    image_path = "/Users/yusif/Desktop/assets/testpurpose/yusif.jpg"
    transcribe_reference = "Mən Yusif Eynullabəyli kredit almaq istəyirəm"
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    print(f"✅ Test files found")
    print(f"📹 Video: {video_path}")
    print(f"🖼️  Image: {image_path}")
    
    # Read files into memory
    print("\n📖 Reading files into memory...")
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    print(f"📊 Video size: {len(video_bytes):,} bytes")
    print(f"📊 Image size: {len(image_bytes):,} bytes")
    
    # Test video verification with timing
    print("\n🚀 Starting video verification with Metal GPU optimization...")
    start_time = time.time()
    
    try:
        result = video_verification(
            video_bytes=video_bytes,
            image_bytes=image_bytes,
            transcribe_reference=transcribe_reference,
            debug=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✅ Video verification completed!")
        print(f"⏱️  Total processing time: {total_time:.2f} seconds")
        print(f"⏱️  Service reported time: {result.get('elapsed_time', 'N/A')}")
        
        # Display results
        print(f"\n📊 Results:")
        print(f"   Similarity: {result.get('similarity', 'N/A')}")
        print(f"   Verified: {result.get('verified', 'N/A')}")
        print(f"   Liveness: {result.get('liveness', 'N/A')}")
        print(f"   Transcription: {result.get('transcription', 'N/A')}")
        print(f"   Transcription Similarity: {result.get('transcription_similarity', 'N/A')}")
        
        # Performance analysis
        print(f"\n📈 Performance Analysis:")
        if 'elapsed_time' in result:
            service_time_str = result['elapsed_time']
            if 'seconds' in service_time_str:
                service_time = float(service_time_str.replace(' seconds', ''))
                print(f"   Service processing time: {service_time:.2f} seconds")
                print(f"   Total overhead: {total_time - service_time:.2f} seconds")
        
        # Expected improvements
        print(f"\n🚀 Expected Improvements with Metal GPU:")
        print(f"   • DeepFace operations: 2-5x faster")
        print(f"   • Whisper transcription: 3-8x faster")
        print(f"   • Parallel processing: Reduced frame sampling")
        print(f"   • Model caching: Faster subsequent runs")
        
    except Exception as e:
        print(f"❌ Error during video verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_performance()
