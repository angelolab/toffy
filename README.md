# toffy
The toffy repo is designed to simplify the process of generating and processing data on the MIBIScope platform.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- TBD once cami finishes installation update

## Overview
The repo has three main parts, with associated code and jupyter notebooks for each

### Setting up a MIBI run 
For large MIBI runs, it is often convenient to automatically generate the JSON file containing the individual FOVs. There are two notebooks for this task, one for large tiled regions, the second for TMAs. If you will be tiling multiple adjacent FOVs together into a single image, the [generate tiled regions](./templates/generate_tiled_regions.ipynb) notebook can automate this process. You provide the location of the top corner of the tiled region, along with the number of fovs along the rows and columns, and it will automatically create the appropriate JSON file. 

The second notebook, [autolabel tma cores](./templates/autolabel_tma_cores.ipynb), is for TMAs. This notebook is run after you have selected the appropriate cores from the TMA. It will generate an overlay with the image of the TMA and the locations you picked to ensure you selected the correct cores. It will then check that they are named correctly and that there are no duplicates.

### Monitoring a MIBI run
While the MIBI is running, it is often useful to generate certain QC plots to ensure everything is going smoothly. We have created the [MIBI watcher notebook](TBD) to automate this process. Once you've started a run, simply tell the watcher notebook the name of your run and it will automatically monitor its process. The notebook will generate plots to track signal intensity, monitor the status of the detector, and even begin processing your data as soon as it comes off the instrument

## Requirements

Latest version of conda (miniconda prefered)

## Setup

Clone the repo

```
git clone https://github.com/angelolab/toffy.git
```

Move into directory and build environment

```
cd toffy
conda env create -f environment.yml
```

## Usage

Activate the environment

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
