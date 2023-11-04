import setuptools
from setuptools import find_packages
import re

with open("./autodistill_hls_geospatial/__init__.py", 'r') as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)
    
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autodistill-hls-geospatial",
    version=version,
    author="Roboflow",
    author_email="support@roboflow.com",
    description="Use the HLS Geospatial model made by NASA and IBM to generate masks for use in training a fine-tuned segmentation model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autodistill/autodistill-hls-geospatial",
    install_requires=[
        "autodistill",
        "supervision",
        "rasterio",
        "mmcv-full==1.5.0",
        "mmsegmentation@git+https://github.com/open-mmlab/mmsegmentation.git@186572a3ce64ac9b6b37e66d58c76515000c3280"
    ],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
