# toffy
Scripts for interacting with and generating data from the commercial MIBIScope

# System
Depending on what operating system you are planning to install toffy on, the process will differ slightly.
- [Install on Windows](#windows-users)
- [Install on MacOS](#mac-users)

# Windows
## Requirements

- You must have **C++ Build Tools** (VS19) installed. 
Go to  https://visualstudio.microsoft.com/visual-cpp-build-tools/ and click 'Download Build Tools'.
Open the installer and then follow the prompts.

- Also, you will need the latest version of Anaconda (**Miniconda** preferred). 
Download here: https://docs.conda.io/en/latest/miniconda.html and select the appropriate download for your system.
Choose "Just Me" option for installation, and do not need to select the "Tutorial" or "Getting Started" options.
Continue with the installation.

## Setup
After confirming you have the necessary software, you can clone the repository.

Now, open the Anaconda powershell prompt instead of the regular powershell prompt.
If you do not already have git installed, run
```
conda install git
```
At this point, you will need to contact an Ionpath employee for the admin password.
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

# MacOS
## Requirements
You will need the latest version of Anaconda (**Miniconda** preferred). 
Download here: https://docs.conda.io/en/latest/miniconda.html and select the appropriate download for your system.
Choose "Just Me" option for installation, and do not need to select the "Tutorial" or "Getting Started" options.
Continue with the installation.

## Setup
After confirming you have the necessary software, you can clone the repository.

Now, open the Anaconda powershell prompt instead of the regular powershell prompt.
If you do not already have git installed, run
```
conda install git
```
At this point, you will need to contact an Ionpath employee for the admin password.
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

Feel free to open an issue on our github page
