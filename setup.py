from setuptools import setup, find_packages

setup(
    name="flower",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pytest",
        "pytest-asyncio",
        "websockets",
        "flwr==1.14.0",
        "torch",
        "torchvision",
        "psutil",
        "poliastro",
        "astropy",
    ],
) 