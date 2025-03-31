from setuptools import setup, find_packages
from pathlib import Path

# 读取 README 作为长描述
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="research_work",  # 包名（确保在 PyPI 上唯一）
    version="0.1.1",  # 建议使用语义化版本号
    author="NY no dorami",
    author_email="shi.chen8866@gmail.com",
    description="A Python package for research tools including algorithms and preprocessing.",
    long_description=long_description,  # 从 README.md 读取
    long_description_content_type="text/markdown",  # README 的格式
    url="https://github.com/shichen137035/research_work",  # 替换为你的 GitHub 链接
    project_urls={  # 可选，但推荐
        "Bug Tracker": "https://github.com/shichen137035/research_work/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # 若你使用 MIT 许可
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,  # 包含 MANIFEST.in 指定的非代码文件
    install_requires=["numpy","torch","matplotlib"],  # 可填写依赖列表，如 ["numpy", "pandas"]
)