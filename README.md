# Event2vec
This repository provides a reference implementation of *Event2vec* as described in the [paper](https://arxiv.org/abs/1901.10234). 

### Environment
```
$ pip install -r requirements.txt
```

### Basic Usage
```
$ python main.py --input 'data/dblp/dblp.txt' --output 'output/dblp/dblp_e2v_embeddings.txt'
```
>noted: your can just checkout main.py to get what you want.

### Input
Your input graph data should be a **txt** file.

#### file format
The txt file should be **edgelist**.

#### txt file sample
	0 163
	0 359
	0 414
	...
	5297 4973

> noted: The graph should be an undirected graph, so if (I  J) exist in the Input file, (J  I) should not.

### Citing
If you find *Event2vec* useful in your research, please cite our paper:

	@article{fu2019representation,
     title={Representation Learning for Heterogeneous Information Networks via Embedding Events},
     author={Fu, Guoji and Yuan, Bo and Duan, Qiqi and Yao, Xin},
     journal={arXiv preprint arXiv:1901.10234},
     year={2019}
    }
