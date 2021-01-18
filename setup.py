import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="replay-monitor", # Replace with your own username
    version="0.0.2",
    author="Leor Cohen",
    author_email="liorcohen5@gmail.com",
    description="A tool for easy data exploration in reinforcement learning environments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liorcohen5/replay-monitor",
    download_url="https://github.com/liorcohen5/replay-monitor/archive/0.0.2.tar.gz",
    keywords=['reinforcement learning', 'tool', 'data exploration', 'replay', 'monitor', 'analytical tool'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    Install_requires=['gym', 'bokeh', 'tensorflow', 'numpy'],
    entry_points={
        'console_scripts': ['replay-monitor=replay_monitor.visualizer:start_server'],
    },
    python_requires='>=3.6',
)