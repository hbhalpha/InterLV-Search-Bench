from setuptools import find_packages, setup

BASE_REQUIRES = [
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.3",
    "Pillow>=10.2.0",
    "tenacity>=8.2.3",
    "httpx[http2]>=0.27.0",
    "openai>=1.35.0",
]

EXTRAS = {
    "gemini": ["google-generativeai>=0.4.0"],
    "browser": ["playwright>=1.40.0"],
    "judge": ["tqdm>=4.66.0"],
    "local": [
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "qwen-vl-utils>=0.0.1",
    ],
}
EXTRAS["all"] = sorted({dep for name, deps in EXTRAS.items() if name != "local" for dep in deps})
EXTRAS["full"] = sorted({dep for deps in EXTRAS.values() for dep in deps})

setup(
    name="agentic-search",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=BASE_REQUIRES,
    extras_require=EXTRAS,
)
