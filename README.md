![alt text](https://github.com/Anchewei/ALPara/blob/main/Images/AL_Logo.png)

# Alignment Parameters: Quantifying Dense Core Alignment in Star-forming Regions
Alignment parameters utilize 2D core positions and core weights to quantitatively assess core alignment. These parameters offer an objective, automated and reproducible method for measuring core alignment, with higher values indicating more **aligned** configurations and lower values suggesting more **clustered** arrangements. To employ these parameters as a two-label classification tool, a threshold of 3.3 can be used to differentiate between "clustered" and "aligned" categories.

![alt text](https://github.com/Anchewei/ALPara/blob/main/Images/ALuwIllus.png)

If these alignment parameters are used in published work, please cite the [relevant paper](https://iopscience.iop.org/article/10.3847/1538-4357/ad9a5b). This paper details the derivation of the parameters, their performance testing, and the determination of the classification threshold. Additionally, the parameters are applied to the 1.3 mm dust continuum images from the ASHES survey (PI: P. Sanhueza) to investigate the relationship between core alignment and clump-scale properties.

## Dependencies
* NumPy,
* Matplotlib

## Using the code
An example notebook, `TestClump.ipynb`, demonstrates how to use the functions to calculate alignment parameters and reproduce the test clumps described in Section 2.1 of the accompanying paper.