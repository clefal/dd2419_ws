from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Install config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.csv')),

        # Install launch files (optional but recommended)
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='diego',
    maintainer_email='difaosfo@gmail.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mapping = mapping.mapping:main',
            'mapping_prob = mapping.mapping_prob:main',
            'workspace_loader = mapping.workspace_loader:main',
        ],
    },
)
