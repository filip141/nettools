from distutils.core import setup

setup(
    name='nettools',
    version='0.1',
    packages=['nettools', 'nettools.utils', 'nettools.epidemic', 'nettools.monoplex', 'nettools.multiplex'],
    url='',
    license='',
    author='filip141',
    author_email='201134@student.pwr.wroc.pl',
    description='Tools for analyzing multiplex networks.', requires=['numpy', 'networkx', 'matplotlib', 'scipy']
)
