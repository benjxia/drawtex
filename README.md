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

Combined with HASYv2 dataset. 
1. Combined test/train data for HASYv2
2. Consolidated classes
   1. `\varpropto` -> `\propto`
   2. `\mid` -> `|`
   3. `\triangle` -> `\delta`
   4. `\check` -> `\checkmark`
   5. `\setminus` -> `\backslash`
   6. `\with` -> `\&`
   7. `\triangledown` -> `\nabla`
   8. `\longmapsto` -> `\mapsto`
   9. `\dotsc` -> `\dots`
   10. `\mathsection` -> `\S`
   11. `\vartriangle` -> `\delta`
   12. `\mathbb{Z}` -> `\mathds{Z}`
   13. `\mathbb{R}` -> `\mathds{R}`
   14. `\mathbb{Q}` -> `\mathds{Q}`
   15. `\mathbb{N}` -> `\mathds{N}`
   16. Combine lowercase letters w/ capital case
