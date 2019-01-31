from distutils.core import setup

setup(
    name='MobRob-Localization',
    version='0.1',
    description='A module for localizing a cup on a table.',
    author='Daniel Klitzke, Yassine El Himer',
    author_email='info@dk-s.de',
    url='https://github.com/dani2112/mobrob-localization',
    packages=['mrlocalization'],
    install_requires=['tensorflow', 'keras', 'scikit-learn', 'opencv-contrib-python']
    license='MIT',
)
