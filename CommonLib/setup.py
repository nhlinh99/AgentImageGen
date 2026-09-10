import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="common-lib",
    version="0.1.0",
    description="Shared job models, config, and infra services for ImageGenBackend + ImageGenCelery",
    packages=setuptools.find_packages('.'),
    python_requires='>=3.10',
    install_requires=required,
)
