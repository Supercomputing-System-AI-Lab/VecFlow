from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
	def has_ext_modules(self):
		return True

setup(
	name="vecflow",
	version="0.0.1",
	packages=find_packages(),
	include_package_data=True,
	package_data={
		'vecflow': ['*.so'],  # Include .so file
	},
	distclass=BinaryDistribution,
	install_requires=[
		"numpy>=1.23",
	],
	python_requires=">=3.10"
)