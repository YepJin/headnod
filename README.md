# headnod

# The Code is based on ActionFormer (https://github.com/happyharrycn/actionformer_release), with adjustment in replacing the features input and some configuration files.




## Code Overview
The structure of this code repo is heavily inspired by ActionFormer. Some of the main components are
* ./libs/core: Parameter configuration module.
* ./libs/datasets: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and postprocessing.

## Installation
* Follow INSTALL.md for installing necessary dependencies and compiling the ActionFormer code.

## Before Running

*Run aug3.py to randomly clip the videos and add gaussian noise to existing video features.


