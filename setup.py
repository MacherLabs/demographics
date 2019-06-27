from setuptools import setup
import platform
machine = platform.machine()

setup( 
    name='demographics',
    
    version='0.11',
    description='Face demographics based on age and gender',
    url='http://demo.vedalabs.in/',

    # Author details    
    author='Atinderpal Singh',
    author_email='atinderpalap@gmail.com',

    packages=['demographics'],
    install_requires=['numpy'] if machine=='x86_64' else ['numpy'],
    package_data={
        'demographics':['model_age/checkpoint*', 'model_gender/checkpoint*'],
    },

    zip_safe=False
    )
