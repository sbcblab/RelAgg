# Relevance aggregation for neural networks interpretability and knowledge discovery on tabular data

## How to use

To configure all the hyperparameters of relevance aggregation, you only need to create a ```config.py``` file. An example can be downloaded [here](config.py).

You will need Python 3 to run this code. Check if the needed libraries are installed with:

```
python3 check_dep.py
```
To train neural networks, use:
```
python3 train.py config.py
```
The relevance scores can then be computed by:
```
python3 get_relevances.py config.py
```
To generate the table heatmaps, run:
```
python3 heatsheets.py config.py
```
And for the weighted t-SNE visualization, run:
```
python3 visualize.py config.py
```

## Datasets

You can download the datasets used in the experiments [here](DATA/DATA.md).

## Contact information

- [Bruno I. Grisci](https://orcid.org/0000-0003-4083-5881) - PhD student ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

    - bigrisci@inf.ufrgs.br

- [Dr. Mathias J. Krause](https://www.lbrg.kit.edu/~mjkrause/) - ([Institute for Applied and Numerical Mathematics 2](http://www.math.kit.edu/ianm2/en) - [KIT](http://www.kit.edu/english/index.php))

    - mathias.krause@kit.edu

- [Dr. Marcio Dorn](https://orcid.org/0000-0001-8534-3480) - Adjunct Professor ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

    - mdorn@inf.ufrgs.br

- http://sbcb.inf.ufrgs.br/