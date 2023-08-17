from setuptools import find_namespace_packages, setup

setup(
    name=f"shark-ace",
    version=f"0.1dev1",
    packages=find_namespace_packages(include=[
        "iree.ace",
        "iree.ace.*",
    ],),
    install_requires=[
        "numpy",
    ],
    extras_require={
    },
)