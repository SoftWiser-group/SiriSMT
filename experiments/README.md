# Dataset

The dataset is available at https://zenodo.org/records/13764675, which has been split into training, validation, 
and test sets.

## get_dataset.sh

This script will download and extract the given dataset to the data directory 
([dataset] should in "AProVE", "sage2", "leipzig", "hycomp", "core", "coreutils")

```shell
./get_dataset.sh [dataset]
```

## build_graph.sh

This script will create a .npz graph file for each .smt file in the specified folder.

```shell
./build_graphs.sh -j [pararrel_num] -d [dataset_folder]
```