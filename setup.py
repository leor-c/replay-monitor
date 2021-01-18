import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="replay-monitor", # Replace with your own username
    version="0.0.1",
    author="Leor Cohen",
    author_email="liorcohen5@gmail.com",
    description="A tool for easy data exploration in reinforcement learning environments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liorcohen5/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': ['replay-monitor=replay_monitor.visualizer:start_server'],
    },
    python_requires='>=3.6',
)