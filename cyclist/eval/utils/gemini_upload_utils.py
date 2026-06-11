"""
Utility functions for uploading videos to Gemini and managing video mappings.
This module can be imported in notebooks or other scripts for video upload functionality.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import google.genai as genai


class GeminiVideoUploader:
    """
    A class to handle video uploads to Gemini and manage mappings.
    """
    
    def __init__(self, mapping_file: str = "gemini_video_mapping.json"):
        """
        Initialize the uploader.
        
        Args:
            mapping_file: Path to the mapping file
        """
        self.client = genai.Client()
        self.mapping_file = mapping_file
        self.mapping = self.load_mapping()
    
    def load_mapping(self) -> Dict[str, str]:
        """Load existing mapping file if it exists."""
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load mapping file {self.mapping_file}: {e}")
        return {}
    
    def save_mapping(self):
        """Save the current mapping to file."""
        with open(self.mapping_file, 'w') as f:
            json.dump(self.mapping, f, indent=2)
    
    def get_video_files(self, directory: str, extensions: List[str] = None) -> List[str]:
        """
        Get all video files from a directory.
        
        Args:
            directory: Path to the directory containing videos
            extensions: List of video file extensions to look for
        
        Returns:
            List of video file paths
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        
        video_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise ValueError(f"Directory {directory} does not exist")
        
        for ext in extensions:
            video_files.extend(directory_path.glob(f"**/*{ext}"))
            video_files.extend(directory_path.glob(f"**/*{ext.upper()}"))
        
        return [str(video_file) for video_file in sorted(video_files)]
    
    def upload_video(self, video_file_path: str, force_reupload: bool = False) -> Optional[str]:
        """
        Upload a video file to Gemini.
        
        Args:
            video_file_path: Path to the video file
            force_reupload: If True, upload even if already in mapping
        
        Returns:
            The name/ID of the uploaded video file, or existing name if already uploaded
        """
        # Check if already uploaded
        if not force_reupload and video_file_path in self.mapping:
            print(f"Video {video_file_path} already uploaded: {self.mapping[video_file_path]}")
            return self.mapping[video_file_path]
        
        print(f"Uploading {video_file_path}...")
        
        try:
            video_file = self.client.files.upload(file=video_file_path)
            
            while video_file.state == "PROCESSING":
                print(f'  Waiting for {os.path.basename(video_file_path)} to be processed...')
                time.sleep(10)
                video_file = self.client.files.get(name=video_file.name)
            
            if video_file.state == "FAILED":
                raise ValueError(f"Video processing failed: {video_file.state}")
            
            print(f'  Video processing complete: {video_file.uri}')
            
            # Update mapping
            self.mapping[video_file_path] = video_file.name
            self.save_mapping()
            
            return video_file.name
        
        except Exception as e:
            print(f"Error uploading {video_file_path}: {str(e)}")
            return None
    
    def upload_directory(self, directory: str, extensions: List[str] = None, 
                        skip_existing: bool = True) -> Dict[str, Optional[str]]:
        """
        Upload all videos in a directory.
        
        Args:
            directory: Directory containing videos
            extensions: Video file extensions to look for
            skip_existing: Skip files already in mapping
        
        Returns:
            Dictionary mapping file paths to Gemini names (None if failed)
        """
        video_files = self.get_video_files(directory, extensions)
        results = {}
        
        print(f"Found {len(video_files)} video files in {directory}")
        
        for video_file in video_files:
            if skip_existing and video_file in self.mapping:
                results[video_file] = self.mapping[video_file]
                print(f"Skipping {video_file} (already uploaded)")
                continue
            
            gemini_name = self.upload_video(video_file)
            results[video_file] = gemini_name
        
        return results
    
    def get_gemini_name(self, video_path: str) -> Optional[str]:
        """Get the Gemini name for a video file path."""
        return self.mapping.get(video_path)
    
    def get_video_path(self, gemini_name: str) -> Optional[str]:
        """Get the video file path for a Gemini name."""
        for path, name in self.mapping.items():
            if name == gemini_name:
                return path
        return None
    
    def list_uploaded_videos(self) -> Dict[str, str]:
        """Get a copy of the current mapping."""
        return self.mapping.copy()
    
    def remove_from_mapping(self, video_path: str) -> bool:
        """
        Remove a video from the mapping.
        
        Args:
            video_path: Path to the video file
        
        Returns:
            True if removed, False if not found
        """
        if video_path in self.mapping:
            del self.mapping[video_path]
            self.save_mapping()
            return True
        return False


# Convenience functions for direct use
def upload_videos_from_directory(directory: str, mapping_file: str = "gemini_video_mapping.json",
                                extensions: List[str] = None, skip_existing: bool = True) -> Dict[str, Optional[str]]:
    """
    Convenience function to upload all videos from a directory.
    
    Args:
        directory: Directory containing videos
        mapping_file: Path to save mapping file
        extensions: Video file extensions to look for
        skip_existing: Skip files already in mapping
    
    Returns:
        Dictionary mapping file paths to Gemini names
    """
    uploader = GeminiVideoUploader(mapping_file)
    return uploader.upload_directory(directory, extensions, skip_existing)


def get_gemini_mapping(mapping_file: str = "gemini_video_mapping.json") -> Dict[str, str]:
    """
    Load and return the current video mapping.
    
    Args:
        mapping_file: Path to the mapping file
    
    Returns:
        Dictionary mapping file paths to Gemini names
    """
    uploader = GeminiVideoUploader(mapping_file)
    return uploader.list_uploaded_videos()