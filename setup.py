from codecs import open as codecs_open
from setuptools import setup, find_packages


with codecs_open('README', encoding='utf-8') as f:
    long_description = f.read()


setup(name='shclassify',
      version='0.0.1',
      description=u"SLS HRIS Land Cover Classification",
      long_description=long_description,
      classifiers=[],
      keywords='',
      author=u"Jotham Apaloo",
      author_email='jothamapaloo@gmail.com',
      url='',
      license='',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      package_data={
          'shclassify': ['data/*'],
      },
      zip_safe=False,
      install_requires=[
          'click',
          'pandas==0.17.1',
      ],
      extras_require={
          'test': ['pytest'],
      },
      entry_points="""
      [console_scripts]
      shclassify=shclassify.scripts.cli:cli
      """
      )
