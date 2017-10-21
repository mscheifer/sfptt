import setuptools

REQUIRED_PACKAGES = ['unidecode>=0.4']

setuptools.setup(name='trainer', version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)
