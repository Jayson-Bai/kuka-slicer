from setuptools import find_packages, setup


package_name = "external_npz_preprocessor"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "path_processing_core"],
    zip_safe=True,
    maintainer="jayson",
    maintainer_email="luobo796@gmail.com",
    description="External path NPZ preprocessor for my_project system NPZ export.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "external_npz_preprocessor_cli = external_npz_preprocessor.cli:main",
            "external_npz_preprocessor_ui = external_npz_preprocessor.app:main",
        ],
    },
)
