from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='postproc',
    version='0.0.1',
    description='My PhD post-processing tools',
    long_description=readme,
    author='Bernat Font Garcia',
    author_email='b.fontgarcia@soton.ac.uk',
    keywords=['post-processing, cfd, fortran'],
    url='https://github.com/b-fg/postproc',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
