# DrawTeX Model

## Overview
This repository contains the deep neural model used for DrawTeX.

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
├── README.md
├── ...
```
Note that unzipping `archive.zip` from the direct download will get you an **incomplete** `extracted_images`, unzip the `.rar` file to get the complete dataset.  

Next, delete the `prime` folder within the `extracted_images` directory. I don't know how that shit made it past the dataset creator's quality control because half of the data in there looks exactly like `ascii_124` and `1`.
