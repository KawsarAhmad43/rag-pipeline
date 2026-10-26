from pathlib import Path
from pipeline import create_super_graph
from langchain_core.messages import HumanMessage

PROJECT_DIR = Path.cwd() / "llama_context_project"
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    super_graph = create_super_graph()
    for s in super_graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Write a LinkedIn post on the paper 'Extending Llama-3’s Context Ten-Fold Overnight'. First consult the research team. Then make sure you consult the LinkedIn team, and check for copy editing and dopeness, and write the file to disk."
                )
            ],
        },
        {"recursion_limit": 30},
    ):
        if "__end__" not in s:
            print(s)
            print("---")

if __name__ == "__main__":
    main()
