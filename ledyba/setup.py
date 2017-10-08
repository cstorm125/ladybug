from setuptools import setup

setup(name='ledyba',
      version='0.1',
      description='Girl-ladyboy detection with VGG-Face',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='ladyboy vgg-face',
      url='https://github.com/cstorm125/ladybug',
      author='cstorm125',
      author_email='cebril@gmail.com',
      license='MIT',
      packages=['ledyba'],
      install_requires=[
          'keras','numpy'
      ],
      include_package_data=True)