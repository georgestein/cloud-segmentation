from setuptools import find_packages, setup

setup(
    name='cloud_segmentation',
    version="0.0",
    url='http://github.com/georgestein/cloud-segmentation/',
    author="George Stein",
    author_email='george.f.stein@gmail.com',
    packages=find_packages(), #['cloud_seg, cloud_seg.pc_apis'], #find_packages(exclude=["tests", "docs", "scripts"]),
    package_data={
        'cloud_seg': [],
        },
    install_requires=[],
    dependency_links=[]
)
