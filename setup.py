from setuptools import setup, find_packages

setup(
    name="research_work",             # 包名称
    version="0.1",                       # 包的版本号
    packages=find_packages(),            # 自动查找包
    author="NY no dorami",                  # 作者名称
    author_email="shi.chen8866@gmail.com",
    description="A sample Python package for local usage",  # 包的描述
    long_description="This package is intended for local use only.",  # 详细描述
    python_requires='>=3.8',             # 适用的 Python 版本要求
)