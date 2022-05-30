# Predicting E-Commerce Site Cisitors’ Behavior

# SIADS 695 Milestone 2: Team Project 

## Overview

This is the repo that includes the code, data, and model components for the analysis.

## Prerequisites

You will need the following things properly installed on your computer.

* [Git](https://git-scm.com/downloads)
* [Python3.6+](https://www.python.org/downloads/)

### NOTE: If you have already setup the environment below, you can run the following: `make`

## Setup Env

Please create a `venv` with `python3`:
```
$ python3 -m venv milestone_2_venv
```

Please use the virtual env when developing the backend: 
```
$ source milestone_2_venv/bin/activate
```

Install the `python` dependencies:
```
(milestone_2_venv) $ pip install -r requirements.txt
```

Moreover, whenever you want to add a new package to our backend, run the following to document the dependencies:
```
(milestone_2_venv) $ pip install <package_name> && pip freeze > requirements.txt
```
