from setuptools import find_packages, setup

package_name = 'localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='mgassmann@ethz.ch',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'localization = localization.icp_scan_to_line:main',
            'filter_scan = localization.filter_scan:main',
            'map_odom_freezer = localization.map_odom_freezer:main',
        ],
    },
)
