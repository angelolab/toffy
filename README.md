# toffy
Scripts for interacting with and generating data from the commercial MIBIScope.

### System
Depending on what operating system you are planning to install toffy on, the requirements will differ slightly.
- [Install on Windows](#windows)
- [Install on macOS](#macos)

## Requirements
### Windows

- You must have **C++ Build Tools** (VS19) installed. 
Go to  https://visualstudio.microsoft.com/visual-cpp-build-tools/ and click 'Download Build Tools'.
Open the installer and make sure you are installing the package labeled *C++ build tools*, then follow the prompts.
    - **(If installing on CAC, you will need the admin password and must contact support@ionpath.com)**

- You will need the latest version of Anaconda (**Miniconda** preferred). 
Download here: https://docs.conda.io/en/latest/miniconda.html and select the appropriate download for your system.
Choose "Just Me" option for installation, and do not need to select the "Tutorial" or "Getting Started" options.
Continue with the installation.

### macOS
- You will need the latest version of Anaconda (**Miniconda** preferred). 
Download here: https://docs.conda.io/en/latest/miniconda.html and select the appropriate download for your system.
Choose "Just Me" option for installation, and do not need to select the "Tutorial" or "Getting Started" options.
Continue with the installation.

## Setup
* For Windows, you will need open the Anaconda powershell prompt instead of the regular powershell prompt for the following.
<img height="400" src="templates/img/conda_powershell.png" width="500"/>
* If macOS user, open terminal. 

If you do not already have git installed, run
```
conda install git
```
Navigate to the desired location (ex: Documents) and clone the repo.
```
cd .\Documents\
git clone https://github.com/angelolab/toffy.git
```

Move into directory and build the environment

```
cd toffy
conda env create -f environment.yml
```

## Usage

Activate the environment:

```
conda activate toffy_env
```

Once activated, notebooks can be used via this command:

```
jupyter lab --allow-root
```

## Updating

Run the command

```
git pull
```

> The following step will probably be changed in the future

You may have to rebuild the environment which can be done via:

```
conda remove --name toffy_env --all
conda env create -f environment.yml
```


## Questions?

Feel free to open an [issue](https://github.com/angelolab/toffy/issues) on our GitHub page.

Before opening, please double check and see that someone else hasn't opened an issue for your question already.
