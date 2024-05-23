# Drone-Data-Labelling-Toolkit
Toolkit assembled for VENLab researchers to identify tracked objects from drone video footage


# Labeling Script

This repository contains a script for labeling data using OpenCV. Follow the steps below to set up the environment and run the script.

## Prerequisites

- Python 3.x
- [VSCode](https://code.visualstudio.com/)
- Videos from your specific assigned session 
- Crowd identities file from your specific session 

## Setup Instructions

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/Hector-Taylor/Drone-Data-Labelling-Toolkit
cd Drone-Data-Labelling-Toolkit
```

### 2. Create a virtual environment 

in your terminal:

```bash
python -m venv venv 
```


### 3. Activate the Virtual Environment

Activate the virtual environment: 

Windows:
```bash
venv\Scripts\activate
```
Mac/Linux:
```bash
source venv/bin/activate
```

### 4. Install the dependencies to your virtual environment 

```bash
pip install -r requirements.txt 
```
### 5. Run the script! 

```bash
python AssociateIdentities_Analogue.py
```