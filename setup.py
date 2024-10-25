from setuptools import setup, find_packages

setup(
    name="colight",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'pyyaml>=5.4.1',
        'gymnasium>=0.28.1',
        'matplotlib>=3.4.3',
        'pandas>=1.3.0',
        'seaborn>=0.11.2'
    ],
    author="George Jiang, Xiang Fu",
    description="Multi-Agent Adaptive Signal Traffic Reinforcement Optimization",
    python_requires='>=3.8',
)