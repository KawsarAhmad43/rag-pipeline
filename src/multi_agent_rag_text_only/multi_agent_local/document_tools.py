from langchain_core.tools import tool
from pathlib import Path
from typing import Dict, List, Optional
from typing_extensions import Annotated
import uuid
import os

class DocumentTools:
    """Handles document-related operations."""

    def __init__(self):
        # Define the base data directory and subdirectory with a random UUID
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.working_directory = Path(os.path.join(data_dir, str(uuid.uuid4())[:8]))
        # Create the full directory path, including parents, if it doesn't exist
        os.makedirs(self.working_directory, exist_ok=True)

    def _sanitize_path(self, file_name: str) -> Path:
        """Sanitize file_name to prevent path traversal and ensure it resolves within working_directory."""
        file_path = (self.working_directory / file_name).resolve()
        if not str(file_path).startswith(str(self.working_directory)):
            raise ValueError(f"Invalid file path: {file_name}. Path must be within working directory.")
        return file_path

    @tool
    def create_outline(points: Annotated[List[str], "List of main points or sections."], file_name: Annotated[str, "File path to save the outline."]) -> Annotated[str, "Path of the saved outline file."]:
        """Create and save an outline to the specified file."""
        try:
            file_path = self._sanitize_path(file_name)
            with file_path.open("w") as file:
                for i, point in enumerate(points):
                    file.write(f"{i + 1}. {point}\n")
            return f"Outline saved to {file_name}"
        except (ValueError, OSError) as e:
            return f"Error creating outline: {str(e)}"

    @tool
    def read_document(file_name: Annotated[str, "File path to read the document."], start: Annotated[Optional[int], "The start line (1-indexed). Default is None"] = None, end: Annotated[Optional[int], "The end line (1-indexed). Default is None"] = None) -> str:
        """Read the specified document, optionally from start to end line."""
        try:
            file_path = self._sanitize_path(file_name)
            with file_path.open("r") as file:
                lines = file.readlines()
            start_idx = start - 1 if start is not None and start > 0 else 0
            end_idx = end if end is not None and end > 0 else len(lines)
            if start_idx > len(lines) or end_idx > len(lines) or start_idx > end_idx:
                return f"Error: Invalid line range (start: {start}, end: {end}) for document with {len(lines)} lines."
            return "\n".join(lines[start_idx:end_idx])
        except (ValueError, OSError) as e:
            return f"Error reading document: {str(e)}"

    @tool
    def write_document(content: Annotated[str, "Text content to be written into the document."], file_name: Annotated[str, "File path to save the document."]) -> Annotated[str, "Path of the saved document file."]:
        """Create and save a text document with the provided content."""
        try:
            file_path = self._sanitize_path(file_name)
            with file_path.open("w") as file:
                file.write(content)
            return f"Document saved to {file_name}"
        except (ValueError, OSError) as e:
            return f"Error writing document: {str(e)}"

    @tool
    def edit_document(file_name: Annotated[str, "Path of the document to be edited."], inserts: Annotated[Dict[int, str], "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line."] = {}) -> Annotated[str, "Path of the edited document file."]:
        """Edit a document by inserting text at specific line numbers."""
        try:
            file_path = self._sanitize_path(file_name)
            with file_path.open("r") as file:
                lines = file.readlines()
            sorted_inserts = sorted(inserts.items())
            for line_number, text in sorted_inserts:
                if not (1 <= line_number <= len(lines) + 1):
                    return f"Error: Line number {line_number} is out of range for document with {len(lines)} lines."
                lines.insert(line_number - 1, text + "\n")
            with file_path.open("w") as file:
                file.writelines(lines)
            return f"Document edited and saved to {file_name}"
        except (ValueError, OSError) as e:
            return f"Error editing document: {str(e)}"
