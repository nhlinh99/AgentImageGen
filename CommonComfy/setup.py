import os
import setuptools


for folder, _, file_names in os.walk("./"):
    init_file_path = os.path.join(folder, "__init__.py")
    if not os.path.exists(init_file_path) \
            and "__pycache__" not in init_file_path \
            and ".idea" not in init_file_path \
            and ".egg-info" not in init_file_path \
            and "dist" not in init_file_path \
            and "build" not in init_file_path:
        open(init_file_path, "w")

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

PACKAGE_NAME = "comfyui-common-lib"

setuptools.setup(
    name=PACKAGE_NAME,
    version="0.2.0",
    author="CVTeam",
    author_email="cvteam@tmtco.asia",
    description="Package to create Common Lib for ComfyUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages('.', exclude=["commonlib"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data = True,
    python_requires='>=3.10',
    install_requires=required
)

# python setup.py bdist_wheel --universal
