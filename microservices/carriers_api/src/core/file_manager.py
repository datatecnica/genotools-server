import os
import glob


class FileManager:
    """Simple file management utilities"""
    
    @staticmethod
    def cleanup_directory(directory_path: str, cleanup_enabled: bool = True):
        """Remove existing files in the directory if cleanup is enabled."""
        print("*" * 100)
        if not cleanup_enabled:
            print(f"Cleanup disabled, skipping removal of existing files in {directory_path}")
            return
        
        existing_files = glob.glob(f'{directory_path}/*')
        for file_path in existing_files:
            try:
                os.remove(file_path)
                print(f"Removed existing file: {file_path}")
            except OSError as e:
                print(f"Error removing file {file_path}: {e}")
        print("*" * 100)
    
    @staticmethod
    def ensure_directory(directory_path: str):
        """Create directory if it doesn't exist"""
        os.makedirs(directory_path, exist_ok=True) 