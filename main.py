import tensorflow as tf
import numpy as np
import argparse
from eventnet import *
import pickle
import time
import os
import networkx as nx
from sklearn import preprocessing

import event2vec


def main():
    ## Parameters setting
    #           beta    rep_dim     epochs      batch_size      learning_rate
    # Movielens: 30        64        2000           128             0.01
    # DBLP     : 30        64        200            128             0.01
    # Douban   : 30        64        200            128             0.01
    # IMDB     : 2         64        200            128             0.01
    # Yelp     : 30        64        2000           128             0.01
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', default=30)
    parser.add_argument('--representation_dim', default=64)
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--learning_rate', default=0.01)

    parser.add_argument('--graph_file', default='dblp/dblp.txt')            
    # parser.add_argument('--graph_file', default='douban/douban.txt')
    # parser.add_argument('--graph_file', default='imdb/imdb.txt')
    # parser.add_argument('--graph_file', default='yelp/yelp.txt')    
    
    parser.add_argument('--event_id_idx', default=[0]) # for dblp | douban | yelp
    # parser.add_argument('--event_id_idx', default=[1]) # for imdb   
     
    parser.add_argument('--node_types', default=['a', 'p', 'c', 't']) # for dblp 
    # parser.add_argument('--node_types', default=['m', 'a', 'd', 'u']) # douban
    # parser.add_argument('--node_types', default=['a', 'm', 'u', 'd']) # imdb
    # parser.add_argument('--node_types', default=['b', 'u', 'l', 'c']) # yelp
     
    parser.add_argument('--output_file', default='output/dblp/dblp_e2v_embeddings.txt') 
    # parser.add_argument('--output_file', default='output/douban/douban_e2v_embeddings.txt')      
    # parser.add_argument('--output_file', default='output/imdb/imdb_e2v_embeddings.txt')      
    # parser.add_argument('--output_file', default='output/yelp/yelp_e2v_embeddings.txt')               
    
    args = parser.parse_args()
    train(args)

def train(args):
    data = LoadData()
    data.load_data(filename=args.graph_file, event_id_idx=args.event_id_idx)
    
    nodes_num = data.nodes_num
    event_num = data.event_num
    node_ind = data.node_ind
    event_dict_int = data.event_dict_int
    
    node_types = args.node_types     
    
    del data
    
    event = EventNet(event_dict=event_dict_int, 
                    nodes_num=nodes_num, 
                    event_num=event_num,
                    node_types=node_types)
    
    node_deg = event.node_deg
    inc_mat = event.inc_mat

    print('number of events: {}'.format(event_num))
    print(nodes_num)
    model = event2vec.EVENT2VEC(nodes_num=nodes_num, 
                                node_types=node_types,
                                nodes_ind=node_ind,
                                beta=args.beta,
                                rep_size=args.representation_dim,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate)
    
    
    for node_type in node_types:
        inc_mat[node_type] = inc_mat[node_type].T
    
    t1 = time.time()
    model.train(inc_mat)
    t2 = time.time()
    print('training time: %s'%(t2-t1))

    model.save_embeddings(args.output_file, inc_mat, node_deg)
    
    

if __name__ == '__main__':
    main()
    