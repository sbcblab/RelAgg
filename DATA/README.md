# Relevance aggregation for neural networks interpretability and knowledge discovery on tabular data

In this page you can find the cleaned and preprocessed datasets used in the experiments of the main paper.

## Synthetic datasets

- [XOR (classification)](XOR/xor_2in50_500.csv)
- [XOR (regression)](XOR/regxor_2in50_500.csv)
- [3-classes](syn/3_5in1000_1000.zip)
- [Regression](syn/regression_4in100_1000.zip)

The XOR datasets were inspired by the publication:

- M. Tan, M. Hartley, M. Bister, R. Deklerck, _Automated feature selectionin neuroevolution_, Evolutionary Intelligence 1 (**2009**) 271–292, DOI: [10.1007/s12065-009-0018-z](https://doi.org/10.1007/s12065-009-0018-z)

The 3-classes and regression datasets were created using [scikit-learn](https://scikit-learn.org/stable/index.html):

- [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
- [make_sparse_uncorrelated](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_uncorrelated.html)

## Real-world datasets

- [Breast cancer gene expression](Breast_GSE45827.zip)
- [E-commerce](shoppers/shoppers.zip)
- [ENEM](enem/enem2016.zip)

The original datasets are from:

- [CuMiDa](http://sbcb.inf.ufrgs.br/cumida)
    - B. C. Feltes, E. B. Chandelier, B. I. Grisci, M. Dorn, _Cumida: An extensively curated microarray database for benchmarking and testing of machine learning approaches in cancer research_, Journal of Computational Biology 26 (**2019**) 376–386, DOI: [10.1089/cmb.2018.0238](https://doi.org/10.1089/cmb.2018.0238)

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
    - C. O. Sakar, S. O. Polat, M. Katircioglu, Y. Kastro, _Real-time predictionof online shoppers’ purchasing intention using multilayer perceptron andlstm recurrent neural networks_, Neural Computing and Applications 31 (**2019**) 6893–6908. 271–292, DOI: [10.1007/s00521-018-3523-0](https://doi.org/10.1007/s00521-018-3523-0)

- [Kaggle](https://www.kaggle.com/davispeixoto/codenation-enem2)
    - [Codenation](https://codenation.dev/)