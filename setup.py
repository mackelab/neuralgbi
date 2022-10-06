from setuptools import find_packages, setup

REQUIRED = [
    "sbi",
]

setup(
    name="gbi",
    python_requires=">=3.6.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)
