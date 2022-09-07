# CSV Flow Data Module

Welcome in this Data Module ! The objective of this code is to have a dynamic data module based on CSV file that indicates the path of the data we want to fetch and the train / val / test separation at hand. It is made to be easy to use and normalised so it can be used accross a vast range of projects and datasets !

## Write the data Split file

The core element of this datamodule is the Data Split File : it indicates the path of the file to select and their different functions.

Usually a DataSplit is a combination of 3 CSV Files : [train, val, test]

In order to represent that they belong to the same split they carry the same name with the variation given : for instance if my datasplit is "MySuperSplit" then I will need to have 3 files ("MySuperSplit_train.csv", "MySuperSplit_val.csv", "MySuperSplit_test.csv").

Fill the csv files with columns names following the standard format. Here we need to be careful because there have to be a perfect coordination between the fields in the csv and the future `request` from the user. In order to keep a good consitency all fields are written in CamelCase and in Singular. For now the fields supported are ['Flow', 'Image', 'GtMask'] but you can implement new ones by modifying "FilesLoader"

## Use the Data Module

Only one function should be used to interact with this code :

```python

CsvDataModule(data_path: str, base_dir: str, batch_size: int, request: list, img_size : tuple, img_transforms_args)


- data_path : path of the csv to use for the split. Should be of the shape "MySuperSplit_*.csv" so the code can then replace the * with 'train', 'val' or 'test'

- base_dir : path in the csv set are relative so we can adapt to different hdd settings and distribute training, the base_dir is the absolute path to append to each path of the csv to get the real file.

- batch_size : int batch size to use

- request : list of columns / data modality to select columns from the csvFile.

- img_size : list of dimension to resize the image / mask / flow to ex : [480, 980]


```
