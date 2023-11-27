# PHYS515

This repo contains the short module given at the end of PHYS515 (Fall 2023) concerning applications to 3D stellar hydro data analysis. 

The notes are provided as Markdown files, with dependent images in a sub-directory. Clone the entire repository and view the notes in a Markdown editor or viewer. If you need a quick-start how to use `git` have a look in the [PHYS248 notes](https://github.com/UVic-CompPhys/PHYS248). There are many around. One option is to view in Jupyter environment in the PPMstar virtual research platform. Other examples are [Visual Studio Code](https://code.visualstudio.com) and [Typora](https://typora.io). 

The notebooks should be executed in the PPMstar virtual research platform as described in the notes. 

**First, look at the `P515_2023_part1.md` for instructions on how to log in to the PPMstar virtual research platform.** Once you have cloned this repo to the ppmstar hub as described, view `P515_2023_part1.md` in the hub with the build-in Markdown viewer. 

## Content of this repository

File | Content
-----|---------
Moments_statistics.md | Notes on moments and hpothesis testing
Parallel_computing.md | Parallel computing notes and instructions how to access the ppmstar.org virtual research platform
Spectra.md | Notes on spectra 
Mollweide_HW1.ipynb | Examples and homework 1 
Spectra_HW2.ipynb | Examples and homework 2 
Assignment | Deadline Monday, Dec 11 

## Outline of the module

**Thursday, Nov 16**

* Revisit estimators of distributions, specifically
  * Skew
  * Kurtosis 
* Revisit _goodness of fit_ methods

  * specifically Kolmogorov-Smirnov and it's pros and cons
  * Anderson-Darling Test: primarily a test for assessing whether a sample comes from a specific distribution (often normal), it is more sensitive to the tails of the distribution compared to other tests like the Kolmogorov-Smirnov test. This sensitivity can make it more effective in detecting departures from normality that manifest in skewness and kurtosis.
  * Jarque-Bera Test: a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution.
* Data-analysis of 3D simulations
  * Introduction to the data
  * Algorithmic compression, data structures, indexing

**Monday, Nov 20**

* Parallel and threaded programing 
  * Hardware foundations and the need for parallel processing
  * General principles
  * Examples: Bash, Python, Fortran

**Thursday, Nov 23**

* Finish parallel computing examples
* Temporal and spatial spectra
* Examples
