import os
def get_bash_files(folder_path):
  """
  Returns a list of file names in the specified folder.

  Args:
    folder_path: The path to the folder.

  Returns:
    A list of strings, where each string is a file name.
  """
  try:
    file_list = os.listdir(folder_path)
    files_only = [f for f in file_list if os.path.isfile(os.path.join(folder_path, f))]
    return files_only
  except FileNotFoundError:
      return f"Error: Folder not found at path: {folder_path}"
  except NotADirectoryError:
      return f"Error: Not a directory: {folder_path}"

def get_deployment_files(folder_path):
  """
  Returns a list of file names in the specified folder.

  Args:
    folder_path: The path to the folder.

  Returns:
    A list of strings, where each string is a file name.
  """
  try:
    file_list = os.listdir(folder_path)
    files_only = [f for f in file_list if os.path.isfile(os.path.join(folder_path, f))]
    return files_only
  except FileNotFoundError:
      return f"Error: Folder not found at path: {folder_path}"
  except NotADirectoryError:
      return f"Error: Not a directory: {folder_path}"
