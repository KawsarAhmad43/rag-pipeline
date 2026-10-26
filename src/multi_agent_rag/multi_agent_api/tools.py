from langchain_core.tools import tool
from pathlib import Path

PROJECT_DIR = Path.cwd() / "llama_context_project"
DATA_DIR = PROJECT_DIR / "data"

@tool
def retrieve_information(query: str):
    """Use Retrieval Augmented Generation to retrieve information about the 'Extending Llama-3’s Context Ten-Fold Overnight' paper."""
    from pipeline import create_rag_chain
    return create_rag_chain().invoke({"question": query})

@tool
def create_outline(points: list[str], file_name: str) -> str:
    """Create and save an outline from a list of points."""
    file_path = PROJECT_DIR / file_name
    with file_path.open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return str(file_path)

@tool
def read_document(file_name: str, start: int = None, end: int = None) -> str:
    """Read and return specified lines from a document."""
    file_path = PROJECT_DIR / file_name
    with file_path.open("r") as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])

@tool
def write_document(content: str, file_name: str) -> str:
    """Write content into a new or existing document."""
    file_path = PROJECT_DIR / file_name
    with file_path.open("w") as file:
        file.write(content)
    return str(file_path)

@tool
def edit_document(file_name: str, inserts: dict[int, str] = {}) -> str:
    """Insert text at specific line numbers in a document."""
    file_path = PROJECT_DIR / file_name
    with file_path.open("r") as file:
        lines = file.readlines()
    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."
    with file_path.open("w") as file:
        file.writelines(lines)
    return str(file_path)
