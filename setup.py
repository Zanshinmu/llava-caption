from setuptools import setup, find_packages

setup(
    name="llava-caption",
    version="0.7.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "httpx",
        "transformers",
        "llama-cpp-python",
        "huggingface-hub",
        "Pillow",
        "ollama",
        "tqdm",
        "pandas",
        "torch",
        "json-repair",
        "mlx",
        "mlx-vlm",
    ],
    entry_points={
        "console_scripts": [
            "llava-caption=llava_caption.cli:main",
        ],
    },
    python_requires=">=3.8",
)