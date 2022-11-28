from setuptools import setup
import re
# from yaml import parse 

with open('README.md') as fp:
    long_description = fp.read()

def parse_req_line(line):
    line = line.strip()
    if not line or line[0] == '#':
        return None
    return line

def load_requirements(file_name):
    with open(file_name) as fp:
        reqs = filter(None, (parse_req_line(line) for line in fp))
        return list(reqs)

def find_version():
    with open('nlp/__init__.py') as fp:
        for line in fp:
            #__version__ = '0.1.0'
            match = re.search(r"__version__\s*=\s*'([^']+)'", line)
            if match:
                return match.group(1)
    assert False, 'cannot find version'

install_requires = load_requirements('requirements.txt')
tests_require = load_requirements('requirements-dev.txt')

setup(
    name='dpr',
    version=find_version(),
    packages=['dprpy'],
    entry_points={
        'console_scripts':['dprd = nlp.httpd:main']
    },
    install_requires=install_requires,
    tests_require=tests_require,
    description='DPR for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    maintainer='rgbayrak',
    maintainer_email='roza.g.bayrak@vanderbilt.edu',
    url='rgbayrak.github.io'
)