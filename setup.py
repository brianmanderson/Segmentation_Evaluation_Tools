__author__ = 'Brian M Anderson'
# Created on 9/15/2020


from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='SegmentationEvaluationTools',
    author='Brian Mark Anderson',
    author_email='bmanderson@mdanderson.org',
    version='0.0.4',
    description='Tools for evaluating predictions on Simple-ITK Images',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=['SegmentationEvaluationTools'],
    package_dir={'': 'src'},
    url='https://github.com/brianmanderson/Segmentation_Evaluation_Tools',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
    ],
    install_requires=required,
)
