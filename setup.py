from typing import List
from setuptools import find_packages, setup

Hypen_E= "-e ."

def get_requirements(File_path: str) -> List[str]:
    """
        this function will return list of requirements
    """
    requirements= []
    with open(File_path) as file_obj:
        requirements= file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]

        if Hypen_E in requirements:
            requirements.remove(Hypen_E)
    
    return requirements

setup(
    name= 'NLP',
    version= '0.0.1',
    author= "Muhammad Hamza Anjum",
    author_email= "hamza.anjum380@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
