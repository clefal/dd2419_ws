from setuptools import find_packages, setup

package_name = 'learning_tf2_py'

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
    maintainer='dduberg',
    maintainer_email='danielduberg@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'executors = learning_tf2_py.executors:main',
            'interpolation = learning_tf2_py.interpolation:main',
            'threading = learning_tf2_py.threading:main',
            'timestamp = learning_tf2_py.timestamp:main',
        ],
    },
)
