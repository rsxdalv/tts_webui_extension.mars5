
import setuptools

setuptools.setup(
	name="tts_webui_extension.mars5",
    packages=setuptools.find_namespace_packages(),
	version="0.0.2",
	author="rsxdalv",
	description="MARS5: A novel speech model for insane prosody",
	url="https://github.com/rsxdalv/tts_webui_extension.mars5",
    project_urls={},
    scripts=[],
    install_requires=[
        "mars5 @ git+https://github.com/rsxdalv/mars5-tts@master",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

