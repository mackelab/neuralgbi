from setuptools import find_packages, setup

REQUIRED = ["invoke", "svgutils==0.3.1", "jupyterlab", "sbibm", "cython"]

setup(
    name="gbi",
    python_requires=">=3.6.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)
