# A Generalized and Open-Source State-Space Framework to Derive Small-Signal Models for EMT Dynamics of Inverter-Dominated Grids

Welcome! This repository contains the case studies included in the manuscript:

<a id="SSH26"></a>[SSH26] P. Serna-Torre, A. Sedlak, and P. Hidalgo-Gonzalez. "A generalized and open-source state-space framework to derive small-signal models for EMT dynamics of inverter-dominated grids", *TechRxiv*, 2026.

To run the files in this repository, follow:

1. To download this repository, run in your terminal:
    ```
    git clone https://github.com/REAM-lab/Generalized-Framework-for-Small-Signal-Modeling-Case-studies.git
    ```

2. In the directory where you downloaded the repository, create and activate a virtual environment:
    ```
    python -m venv .venv 
    source .venv/bin/activate
    ```
> Please make sure that you have Python 3.14. 

3. Install the following packages. To this end, run:
    ```
    pip install -r requirements.txt
    ```

4. You can go to the folders and run the file "run.py".

> [!WARNING] 
> In this repository, we only show the case studies included in the manuscript. The full implementation of the framework is a module of STING:
> https://github.com/REAM-lab/STING.git
> STING is part of the packages that are automatically installed when you run step 3.