#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup, find_namespace_packages
import sys

with open("requirements.txt") as f:
    reqs = f.read()

if __name__ == "__main__":
    setup(
        name="densely_captioned_images-dataset",
        version="0.1",
        description="Research project for densely annotated mask-aligned images",
        #long_description=readme,
        #license=license,
        python_requires=">=3.8",
        packages=find_namespace_packages(include=['densely_captioned_images.*']),
        install_requires=reqs.strip().split("\n"),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: CC-BY-NC License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Natural Language :: English",
        ],
    )



