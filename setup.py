from setuptools import find_packages, setup

setup(
    name='beepose',
    packages=find_packages(),
    version='0.1.0',
    description='Temporary repo for refactor code. After rewriting the code of the pose estimation I will create a new one. ',
    author='Ivan Felipe rodriguez',
    license='BSD-3',
    entry_points={
        'console_scripts': [
            'train_stages_aug = beepose.train.train_stages_aug:main',
            'process_folder_full_video = beepose.inference.process_folder_full_video:main',
            'get_inference_model = beepose.train.get_inference_model:main'
        ],
    }
)
