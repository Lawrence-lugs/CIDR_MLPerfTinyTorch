from setuptools import setup, find_packages

setup(
    name='CIDR_MLPerfTinyTorch',
    version='0.0.1',
    install_requires=[
        'requests',
        'importlib-metadata; python_version<"3.10"',
    ],
    include_package_data=True,
)

