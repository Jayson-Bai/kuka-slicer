from setuptools import find_packages, setup

package_name = "path_processing_core"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="jayson",
    maintainer_email="luobo796@gmail.com",
    description="Shared path command types, calibration, sampling, and system NPZ export core.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
)
