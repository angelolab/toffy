# toffy
The toffy repo is designed to simplify the process of generating and processing data on the MIBIScope platform.

This repo is currently in beta testing. None of the code has been published yet, and we will be making breaking changes frequently. If you find bugs, please [open an issue](https://github.com/angelolab/toffy/issues/new/choose). If you have questions or want to collaborate, please reach out to Noah (nfgreen@stanford.edu)

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
While the MIBI is running, it is often useful to generate certain QC plots to ensure everything is going smoothly. We are in the process of creating the [MIBI watcher notebook](tbd) to automate this process. Once you've started a run, simply tell the watcher notebook the name of your run and it will automatically monitor its progress. The notebook will generate plots to track signal intensity, monitor the status of the detector, and even begin processing your data as soon as it comes off the instrument.

### Processing MIBI data
Once your run has finished, you can begin to process the data to make it ready for analysis. To remove background signal contamination, as well as compensate for channel crosstalk, you can use the [compensate image data](./templates/Compensate_Image_Data.ipynb) notebook. This will guide you through the process of determining the correct coefficients for the Rosetta algorith, which uses a flow-cytometry style compensation approach to remove spurious signal. 

Following compensation, you will want to normalize your images to ensure consistent intensity across the run. This functionality is currently in the works, and we'll have a beta version available to test soon. 

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
