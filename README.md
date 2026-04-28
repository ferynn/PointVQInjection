# ReadMe
A collection of code resources for reproducing figures used in AGU submission. 

## Structure

The files in the top level directory are the core of the presented project. faultPlotInjection does create a figure, but is also a showcase of the percolation component of the model.

MiscFigures holds scripts used to generate other plots in the work. They can be ran by adding them to the top level directory.

SampleData contains pregenerated data for "faultPlotInjection.py" and "comparisonsPercentDiff.py". They can be generated using the appropriate file, by setting a boolean flag to false, which controls if data is loaded from file.
Generating the data from scratch takes a moderate amount of time.
If pre-generated data is desired to be used, add it to the top level directory

## Environment Details

This code was developed and executed on an Anaconda distribution. All packages used are included in the default distribution.
