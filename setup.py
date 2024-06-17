from setuptools import setup, find_packages
setup(
    name = 'ip_topic',
    packages = find_packages(),
)

def initial():
    os.system('pip install -r requirements.txt')
if __name__ == '__main__':
    initial()
