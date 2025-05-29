import platform
from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
	def has_ext_modules(self):
		return True

# Cross-platform binary files
if platform.system() == "Windows":
	binary_files = ['*.dll', '*.pyd']
elif platform.system() == "Darwin":
	binary_files = ['*.dylib', '*.so']
else:  # Linux and other Unix-like
	binary_files = ['*.so']

setup(
	name="vecflow",
	version="0.0.1",
	description="VecFlow: A High-Performance Vector Data Management System for Filtered-Search on GPUs",
	author="SSAIL lab, University of Illinois at Urbana-Champaign",
	url="https://supercomputing-system-ai-lab.github.io/projects/vecflow/",
	packages=find_packages(),
	include_package_data=True,
	package_data={
		'vecflow': binary_files,
	},
	distclass=BinaryDistribution,
	install_requires=[
		"numpy>=1.19.0",  # Core dependency only
	],
	extras_require={
		'gpu': ['cupy>=10.0.0'],  # Optional GPU support
		'cuda11': ['cupy-cuda11x>=10.0.0'],
		'cuda12': ['cupy-cuda12x>=10.0.0'],
	},
	python_requires=">=3.8",
	zip_safe=False,
	classifiers=[
		"Development Status :: 3 - Alpha",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9", 
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Python :: 3.12",
		"Programming Language :: C++",
		"Operating System :: OS Independent",
		"Topic :: Scientific/Engineering",
	],
)