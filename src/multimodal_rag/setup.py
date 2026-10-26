from setuptools import setup, find_packages

setup(
    name="rag_pipelines",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "numpy",
        "langchain",
        "langchain-openai",
        "langchain-community",
        "streamlit",
    ],
    python_requires=">=3.12",
)
