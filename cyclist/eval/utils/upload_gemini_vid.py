"""
Standalone script to upload videos to Gemini using the convenience function.
"""
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from gemini_upload_utils import upload_videos_from_directory
from load_env_vars import load_env

# Load environment variables (API keys, etc.)
load_env()

def main():
    # Configuration - List of video directories to process
    video_directories = [
        "Cyclist/output/videos/unicycle/test",
        "Cyclist/output/videos/bicycle/test",
        "Cyclist/output/videos/tricycle/test",
        "Cyclist/output/videos/unicycle_cluttered/test",
        "Cyclist/output/videos/nightrider/test",


        # Add more directories here as needed
    ]
    mapping_file = "gemini_video_mapping.json"
    
    print(f"Starting video upload from {len(video_directories)} director{'y' if len(video_directories) == 1 else 'ies'}")
    print(f"Mapping file: {mapping_file}")
    print("=" * 60)
    
    # Track overall statistics
    total_successful = 0
    total_failed = 0
    total_processed = 0
    
    # Process each directory
    for idx, video_directory in enumerate(video_directories, 1):
        print(f"\n[{idx}/{len(video_directories)}] Processing directory: {video_directory}")
        print("-" * 60)
        
        if not os.path.exists(video_directory):
            print(f"Warning: Directory does not exist, skipping...")
            continue
        
        if not os.path.isdir(video_directory):
            print(f"Warning: Path is not a directory, skipping...")
            continue
        
        # Upload all videos from the directory
        results = upload_videos_from_directory(
            directory=video_directory,
            mapping_file=mapping_file,
            skip_existing=True  # Skip videos already uploaded
        )
        
        # Print results for this directory
        print(f"\nDirectory upload completed: {len(results)} videos processed")
        
        successful = 0
        failed = 0
        
        for video_path, gemini_name in results.items():
            if gemini_name:
                print(f"✓ {os.path.basename(video_path)} -> {gemini_name}")
                successful += 1
            else:
                print(f"✗ Failed to upload {os.path.basename(video_path)}")
                failed += 1
        
        print(f"Directory summary: {successful} successful, {failed} failed")
        
        total_successful += successful
        total_failed += failed
        total_processed += len(results)
    
    # Print overall summary
    print("\n" + "=" * 60)
    print(f"OVERALL SUMMARY:")
    print(f"  Directories processed: {len(video_directories)}")
    print(f"  Total videos processed: {total_processed}")
    print(f"  Total successful: {total_successful}")
    print(f"  Total failed: {total_failed}")
    print(f"  Mapping saved to: {mapping_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()