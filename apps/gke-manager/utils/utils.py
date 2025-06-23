import time

import streamlit as st
from watchdog.events import FileSystemEventHandler

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, file_path):
        self.file_path = file_path
        self.last_position = 0

    def on_modified(self, event):
        if event.src_path == self.file_path:
            st.write("File modified. Reading new content...")
            with open(self.file_path, 'r') as file:
                file.seek(self.last_position)  # Move to the last read position
                new_content = file.read()
                if new_content:
                    st.write(new_content, end='')  # Print new content without adding extra newlines
                    self.last_position = file.tell()  # Update the last read position

# if __name__ == "__main__":
#     file_to_watch = "example.txt"  # Replace with your file path
#     event_handler = FileChangeHandler(file_to_watch)
    
#     observer = Observer()
#     observer.schedule(event_handler, path=".", recursive=False)
#     observer.start()

#     print(f"Monitoring file: {file_to_watch}. Press Ctrl+C to stop.")
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()

# Simulate real-time streaming
def stream_data(file_path):
    """
    Simulate streaming data from a file.
    Reads the file line-by-line and displays new lines incrementally.
    """
    with open(file_path, "r") as file:
        # Track the current position in the file
        while True:
            line = file.readline()
            if not line:
                # If no new data, wait and check again
                time.sleep(10)
                continue
            yield line

