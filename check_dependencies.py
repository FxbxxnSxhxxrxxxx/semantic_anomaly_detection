#!/home/schlarma/anaconda3/bin/python

import subprocess
import sys

dependencies = [
    "_libgcc_mutex=0.1=main",
    "_openmp_mutex=5.1=1_gnu",
    "anyio=3.5.0=py311h06a4308_0",
    "argon2-cffi=21.3.0=pyhd3eb1b0_0",
    "argon2-cffi-bindings=21.2.0=py311h5eee18b_0",
    "asttokens=2.0.5=pyhd3eb1b0_0",
    "attrs=23.1.0=py311h06a4308_0",
    "backcall=0.2.0=pyhd3eb1b0_0",
    "beautifulsoup4=4.12.2=py311h06a4308_0",
    "blas=1.0=mkl",
    "bleach=4.1.0=pyhd3eb1b0_0",
    "bottleneck=1.3.5=py311hbed6279_0",
    "bzip2=1.0.8=h7b6447c_0",
    "ca-certificates=2023.08.22=h06a4308_0",
    "cffi=1.15.1=py311h5eee18b_3",
    "comm=0.1.2=py311h06a4308_0",
    "debugpy=1.6.7=py311h6a678d5_0",
    "decorator=5.1.1=pyhd3eb1b0_0",
    "defusedxml=0.7.1=pyhd3eb1b0_0",
    "entrypoints=0.4=py311h06a4308_0",
    "executing=0.8.3=pyhd3eb1b0_0",
    "gensim=4.3.0=py311hba01205_1",
    "icu=73.1=h6a678d5_0",
    "idna=3.4=py311h06a4308_0",
    "intel-openmp=2023.1.0=hdb19cb5_46305",
    "ipykernel=6.25.0=py311h92b7b1e_0",
    "ipython=8.15.0=py311h06a4308_0",
    "ipython_genutils=0.2.0=pyhd3eb1b0_1",
    "jedi=0.18.1=py311h06a4308_1",
    "jinja2=3.1.2=py311h06a4308_0",
    "joblib=1.2.0=py311h06a4308_0",
    "jsonschema=4.17.3=py311h06a4308_0",
    "jupyter_client=7.4.9=py311h06a4308_0",
    "jupyter_core=5.3.0=py311h06a4308_0",
    "jupyter_server=1.23.4=py311h06a4308_0",
    "jupyterlab_pygments=0.1.2=py_0",
    "ld_impl_linux-64=2.38=h1181459_1",
    "libffi=3.4.4=h6a678d5_0",
    "libgcc-ng=11.2.0=h1234567_1",
    "libgfortran-ng=11.2.0=h00389a5_1",
    "libgfortran5=11.2.0=h1234567_1",
    "libgomp=11.2.0=h1234567_1",
    "libsodium=1.0.18=h7b6447c_0",
    "libstdcxx-ng=11.2.0=h1234567_1",
    "libuuid=1.41.5=h5eee18b_0",
    "libxml2=2.10.4=hf1b16e4_1",
    "libxslt=1.1.37=h5eee18b_1",
    "lxml=4.9.3=py311hdbbb534_0",
    "markupsafe=2.1.1=py311h5eee18b_0",
    "matplotlib-inline=0.1.6=py311h06a4308_0",
    "mistune=0.8.4=py311h5eee18b_1000",
    "mkl=2023.1.0=h213fc3f_46343",
    "mkl-service=2.4.0=py311h5eee18b_1",
    "mkl_fft=1.3.8=py311h5eee18b_0",
    "mkl_random=1.2.4=py311hdb19cb5_0",
    "nbclassic=0.5.5=py311h06a4308_0",
    "nbclient=0.5.13=py311h06a4308_0",
    "nbconvert=6.5.4=py311h06a4308_0",
    "nbformat=5.9.2=py311h06a4308_0",
    "ncurses=6.4=h6a678d5_0",
    "nest-asyncio=1.5.6=py311h06a4308_0",
    "notebook=6.5.4=py311h06a4308_1",
    "notebook-shim=0.2.2=py311h06a4308_0",
    "numexpr=2.8.7=py311h65dcdc2_0",
    "numpy=1.26.0=py311h08b1b3b_0",
    "numpy-base=1.26.0=py311hf175353_0",
    "openssl=3.0.11=h7f8727e_2",
    "packaging=23.1=py311h06a4308_0",
    "pandas=1.5.3=py311hba01205_0",
    "pandocfilters=1.5.0=pyhd3eb1b0_0",
    "parso=0.8.3=pyhd3eb1b0_0",
    "pexpect=4.8.0=pyhd3eb1b0_3",
    "pickleshare=0.7.5=pyhd3eb1b0_1003",
    "pip=23.2.1=py311h06a4308_0",
    "platformdirs=3.10.0=py311h06a4308_0",
    "prometheus_client=0.14.1=py311h06a4308_0",
    "prompt-toolkit=3.0.36=py311h06a4308_0",
    "psutil=5.9.0=py311h5eee18b_0",
    "ptyprocess=0.7.0=pyhd3eb1b0_2",
    "pure_eval=0.2.2=pyhd3eb1b0_0",
    "pycparser=2.21=pyhd3eb1b0_0",
    "pygments=2.15.1=py311h06a4308_1",
    "pyrsistent=0.18.0=py311h5eee18b_0",
    "python=3.11.3=h955ad1f_1",
    "python-dateutil=2.8.2=pyhd3eb1b0_0",
    "python-fastjsonschema=2.16.2=py311h06a4308_0",
    "pytz=2023.3.post1=py311h06a4308_0",
    "pyzmq=23.2.0=py311h6a678d5_0",
    "readline=8.2=h5eee18b_0",
    "scikit-learn=1.2.2=py311h6a678d5_1",
    "scipy=1.11.3=py311h08b1b3b_0",
    "send2trash=1.8.0=pyhd3eb1b0_1",
    "setuptools=68.0.0=py311h06a4308_0",
    "six=1.16.0=pyhd3eb1b0_1",
    "smart_open=5.2.1=py311h06a4308_0",
    "sniffio=1.2.0=py311h06a4308_1",
    "soupsieve=2.5=py311h06a4308_0",
    "sqlite=3.41.2=h5eee18b_0",
    "stack_data=0.2.0=pyhd3eb1b0_0",
    "tbb=2021.8.0=hdb19cb5_0",
    "terminado=0.17.1=py311h06a4308_0",
    "threadpoolctl=2.2.0=pyh0d69192_0",
    "tinycss2=1.2.1=py311h06a4308_0",
    "tk=8.6.12=h1ccaba5_0",
    "tornado=6.3.3=py311h5eee18b_0",
    "traitlets=5.7.1=py311h06a4308_0",
    "typing-extensions=4.7.1=py311h06a4308_0",
    "typing_extensions=4.7.1=py311h06a4308_0",
    "tzdata=2023c=h04d1e81_0",
    "wcwidth=0.2.5=pyhd3eb1b0_0",
    "webencodings=0.5.1=py311h06a4308_1",
    "websocket-client=0.58.0=py311h06a4308_4",
    "wheel=0.41.2=py311h06a4308_0",
    "xz=5.4.2=h5eee18b_0",
]

# Function to check if a specific dependency is installed with the correct version
def check_dependency(dependency, installed_dependencies, missing_dependencies):
    # Split the dependency into package name and version
    package, version = dependency.split("=")[:2]
    try:
        # Run the 'conda list' command to get information about the package
        output = subprocess.check_output(["conda", "list", package])
        output = output.decode("utf-8")
        for line in output.split("\n"):
            # Check if the line contains the package name
            if line.startswith(package):
                # Get the installed version
                installed_version = line.split()[1]
                # Compare the installed version with the required version
                if installed_version == version:
                    installed_dependencies.append(f"{package}: Installed correctly (version {version})")
                else:
                    missing_dependencies.append(f"{package}: Incorrect version (expected {version}, found {installed_version})")
                break
        else:
            missing_dependencies.append(f"{package}: Not found")
    except subprocess.CalledProcessError:
        missing_dependencies.append(f"{package}: Not found")

# Function to check if a specific version of Conda is installed
def check_conda_version(required_version):
    try:
        # Run the 'conda --version' command and capture its output
        result = subprocess.run(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        installed_version = result.stdout.strip().split()[-1]
        
        # Compare the installed version with the required version
        if installed_version == required_version:
            print(f"Conda Version: {installed_version} - OK")
        else:
            print(f"Conda Version: {installed_version} - Required: {required_version}")
    except FileNotFoundError:
        print("Conda is not installed or not in the system PATH - Required: {required_version}")

def main():
    check_conda_version("23.5.2")
    installed_dependencies = []
    missing_dependencies = []
    for dependency in dependencies:
        check_dependency(dependency, installed_dependencies, missing_dependencies)

    print("\nInstalled Dependencies:")
    print("\n".join(installed_dependencies))

    print("\nMissing or Incorrect Dependencies:")
    print("\n".join(missing_dependencies))

if __name__ == "__main__":
    main()
