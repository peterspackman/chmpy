from setuptools import setup

setup(
    name="promolecule",
    version="0.1a1",
    description="Promolecule and Hirshfeld surfaces using python",
    url="https://github.com/peterspackman/promolecule-python",
    keywords=[
        "chemistry",
        "molecule",
        "crystal",
        "electron density",
        "isosurface"
    ],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=["promolecule",],
    package_data={
        "promolecule": ["*.npz"],
    },
    install_requires=[
        "numpy",
        "scipy",
    ],
    zip_safe=True,
)
