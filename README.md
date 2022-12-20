# DrawTeX Model

## Overview
This repository contains the deep neural network model used for DrawTeX.

## Raw Data

Download the dataset from [https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)  
Decompress the `.rar` file and place the decompressed `extracted_images` folder under the `./data` directory of the repository's root directory.  
In the end, the repository's directory structure should resemble the below.
```
Root
├── data
│   ├── extracted_images
│   │   ├── !
|   |   ├── ...
├── prototypes
├── ├── ...
├── README.md
├── ...
```
Note that unzipping `archive.zip` from the direct download will get you an **incomplete** `extracted_images` folder, unzip the `.rar` file for the complete dataset.

`ldots`, `prime`, `ascii 124 (|)`, and `,` were removed from the dataset because they were redundant/unnecessary.