# Datamining pipeline project

## Datamining framework documentation

* **`__init__.py`** acts as the API layer and routes the calls from main the the specific class and function.
* **`core.py`** is a abstract class that defines the fundamental architecture. It also provides the interfaces for the algorithms that is used like `DistanceMeasure`, `ClusteringTechnique`, and `QualityMeasure`
* **`distance_measures.py`** implements 3 different distance metrics for comparing the data points with each other. It implements Euclidean, Manhattan and Cosine distance.
* **`clustering_techniques.py`** implements 3 different clustering algorithms with configurable hyperparameters. here we implement `Kmeans`, `DBScan` and `Hierarchichal` which were presented in the lecture slides
* **`quality_measures.py`** implement 3 different quality measure metrics to evaluate the performance. From the slides i implemented `silhouetteScore` but there is also some other which were not presented in the slides but that are supposed to be common.

## How to run (For Linux and Mac)

1. go to root directory
```
cd path/datamining-project
```
2. create venv and activate it
```
Python3 -m venv venv
source venv/bin/activate
```
3. install dependencies
```
pip install -r requirements.txt
```
4. run main
```
python3 main.py
```
5. display results
```
cat project1_results.csv
```

## Usage

In order to get a small tutorial on how to run it, look at **`small_example.py`** and play around with it.