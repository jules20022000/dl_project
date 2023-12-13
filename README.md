# Voice-Driven Disease Classification: A Deep Learning Approach

## Date: December 13, 2023

## Contributors

| Name            | Email                      |
|-----------------|----------------------------|
| Paul Nadal      | paul.nadal@epfl.ch         |
| Jules Maglione  | jules.maglione@epfl.ch     |


## Abstract

This project employs deep learning techniques on the "Medical Speech, Transcription, and Intent" dataset to enhance emergency response systems. Through rapid analysis of diverse audio utterances and transcriptions, our model categorizes reported symptoms, directing users to appropriate health services. The project delves into both audio and text-based classification methods, establishing an integrated pipeline that harnesses features from both modalities.

## Loading Data

You can obtain the "Medical Speech, Transcription, and Intent" dataset [here](#). The dataset, totaling approximately 6GB, encompasses thousands of audio utterances related to common medical symptoms like "knee pain" or "headache." In total, it offers over 8 hours of aggregated audio content, with each utterance crafted by individual human contributors based on specific medical symptoms. This extensive collection of audio snippets serves as valuable training data for conversational agents in the medical field. After downloading, unzip the file and copy the content of one of the two folders inside the root directory into a "data" folder at the project's root. Note that the two subfolders contain identical content, so copying from either suffices.

Here is how your `data` folder structure should appear:

```
data
│
├── recordings
│   ├── train
│   │   ├── file1.wav
│   │   ├── file2.wav
│   │   └── ...
│   ├── test
│   │   ├── file1.wav
│   │   ├── file2.wav
│   │   └── ...
│   ├── validate
│   │   ├── file1.wav
│   │   ├── file2.wav
│   │   └── ...
└── overview-of-recordings.csv
```


## Running the Code

### Install Python Libraries

We recommend creating a specific conda environment for the project with Python 3.9 before installing libraries.

```bash
pip install -r requirements.txt
```
> Note: The installation includes the basic PyTorch library. If you plan to use CUDA, please refer to the [PyTorch](https://pytorch.org/get-started/locally/) website for installation instructions.

### Execute the Code

The project comprises four Jupyter notebook files:

- [main.ipynb](./main.ipynb): This file handles data loading, cleaning, and runs each subsequent Jupyter notebook, displaying the results of each section.

- [part1.ipynb](./part1.ipynb): Explore different techniques and train models for audio classification.

- [part2.ipynb](./part2.ipynb): Explore different techniques and train models for text classification.

- [part3.ipynb](./part3.ipynb) : Explore how to convert audio to text before classifying the data.

The final results are also available in the `results` folder created at the root of the project.

