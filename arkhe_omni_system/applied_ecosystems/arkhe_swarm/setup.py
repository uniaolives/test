from setuptools import setup
import os
from glob import glob

package_name = 'arkhe_swarm'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Architect',
    maintainer_email='architect@arkhe.ai',
    description='Arkhe(n) Drone Swarm with Constitutional Protection',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arkhe_core = arkhe_swarm.arkhe_core:main',
            'ghz_consensus = arkhe_swarm.ghz_consensus:main',
            'cognitive_guard = arkhe_swarm.cognitive_guard:main',
            'ai_guard = arkhe_swarm.ai_guard:main',
            'authority_guard = arkhe_swarm.authority_guard:main',
            'transparency_guard = arkhe_swarm.transparency_guard:main',
        ],
    },
)
