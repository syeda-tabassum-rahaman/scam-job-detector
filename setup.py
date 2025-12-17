"""
Package setup helper for the scam-job-detector project

This file builds the package metadata used by setuptools when installing
the project locally or when creating distributions
"""

import os
from setuptools import find_packages
from setuptools import setup

requirements = []

# read runtime requirements from requirements.txt when present
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        content = f.readlines()
    requirements.extend([x.strip() for x in content if 'git+' not in x])

# also include development requirements if provided
if os.path.isfile('requirements_dev.txt'):
    with open('requirements_dev.txt') as f:
        content = f.readlines()
    requirements.extend([x.strip() for x in content if 'git+' not in x])


setup(name='scam_job_detector',
      version="0.0.1",
      description="Project Description",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      # scripts=['scripts/packagename-run'],
      zip_safe=False)
