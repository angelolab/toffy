# creed-helper
Utility and QC scripts for the commercial MibiScope access computer

## Requirements

Latest version of conda (miniconda prefered)

## Setup

Clone the repo

```
git clone https://github.com/angelolab/creed-helper.git
```

Move into directory and build environment

```
cd creed-helper
conda env create -f environment.yml
```

## Usage

Activate the environment

```
conda activate bin_tools_env
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
conda remove --name bin_tools_env --all
conda env create -f environment.yml
```

## Questions?

Feel free to open an issue on our github page
