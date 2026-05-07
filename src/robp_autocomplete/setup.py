from setuptools import find_packages, setup

package_name = 'robp_autocomplete'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/bash-completion/completions',
         ['completion/bash/colcon', 'completion/bash/ros2', 'completion/bash/rosidl']),
        ('share/zsh/site-functions', ['completion/zsh/_colcon',
         'completion/zsh/_ros2', 'completion/zsh/_rosidl']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dduberg',
    maintainer_email='danielduberg@gmail.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
