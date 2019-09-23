import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='trivago2019',
    version='0.1',
    scripts=['trivago2019'],
    author="Federico Parroni",
    author_email="federico.parroni@live.it",
    description="All the files to win the ACM RecSys 2019",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keyblade95/recsys2019.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )