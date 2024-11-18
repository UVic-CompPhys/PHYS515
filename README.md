# PHYS515

This repo contains the short module given at the end of PHYS515 (Fall 2024) concerning methods and analysos related to 3D stellar hydro data simulations. 

The notes are provided as Markdown files, with dependent images in a sub-directory. Clone the entire repository and view the notes in a Markdown editor or viewer. If you need a quick-start how to use `git` have a look in the [PHYS248 notes](https://github.com/UVic-CompPhys/PHYS248). There are many around. One option is to view in Jupyter environment in the PPMstar virtual research platform. Other examples are [Visual Studio Code](https://code.visualstudio.com) and [Typora](https://typora.io). 

The notebooks should be executed in the PPMstar virtual research platform as described in the notes. The document `Parallel_computing.md` explains how ro use the [PPMstar virtual research platform](https://www.ppmstar.org). Once you have cloned this repo to the ppmstar hub as described, you may view Markdown files in the hub with the build-in Markdown viewer. 

* [x] Get students GitHub user names and add them to the _PPMstar Collaborate_ hub.

## Content of this repository

File/Directory | Content
-----|---------
Parallel_computing.md | Parallel computing notes and instructions how to access the ppmstar.org virtual research platform
Spectra.md | Notes on spectra 
Parallel_code_examples | Directory with various examples and templates for parallel computing with Python and Fortran, covering MPI, OpenMP and Python `multiprocessing` 

## Outline of the module

**Monday, Nov 18**

* Large-scale 3D stellar hydro simulations: a case study

  * Which equations and how are they solved
  * Data-analysis of 3D simulations

  * Algorithmic compression, data structures, indexing

* Parallel and threaded programing 

  * Hardware foundations and the need for parallel processing

  * General principles

  * Examples: Bash, Python, Fortran

* Revision of temporal and spatial spectra

