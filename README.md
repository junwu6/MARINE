# MARINE (MAnifold-RegularIzed Network Embedding)
An implementation for "Scalable Manifold-Regularized Attributed Network Embedding via Maximum Mean Discrepancy" (CIKM'19). [[Paper]](https://dl.acm.org/citation.cfm?id=3358091)

## Environment Requirements
The code has been tested under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.12.0
* numpy == 1.15.4
* scipy == 1.1.0
* sklearn == 0.20.0
* networkx == 2.3

## Data sets
We used four data sets in our experiments: [Cora, Citeseer, Pubmed](https://github.com/tkipf/gcn/tree/master/gcn/data) and [Wiki](https://github.com/thunlp/TADW/tree/master/wiki).

## Run the Codes
```
python main.py
```

## Acknowledgement
This is the latest source code of **MARINE** for CIKM2019. If you find that it is helpful for your research, please consider to cite our paper:

```
@inproceedings{MARINE_cikm19,
  author    = {Jun Wu and Jingrui He},
  title     = {Scalable Manifold-Regularized Attributed Network Embedding via Maximum Mean Discrepancy},
  booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  year      = {2019},
  organization = {ACM}
}
```
