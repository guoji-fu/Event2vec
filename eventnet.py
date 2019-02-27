import numpy as np
from numpy.linalg import inv
from sklearn import preprocessing
import random
import pickle as pkl
import networkx as nx


class LoadData(object):
    def __init__(self):
        self.node_size = 0
        self.nodes_num = {}
        self.node_ind = {}
        self.event_num = 0
        self.event_dict = {}
        self.event_dict_int = {}
        self.event_identifier = {}
        self.node_event_weight = {}

    def load_data(self, filename, event_id_idx):
        print('loading data...')
        with open(filename, 'r') as f:
            for line in f:
                items = line.strip().split()
                event_id = ''
                for idx in event_id_idx:
                    event_id += items[idx]
                event_items = {}
                event_items_int = {}
                count = 0
                for item in items:
                    indx = self.encode_node(item)
                    if item[0] not in event_items.keys():
                        event_items[item[0]] = []
                        event_items_int[item[0]] = []
                    event_items[item[0]].append(self.node_ind[item[0]][indx])
                    event_items_int[item[0]].append(indx)
                if event_id not in self.event_identifier.keys():
                    self.event_identifier[event_id] = self.event_num
                    self.event_dict[self.event_num] = {}
                    self.event_dict_int[self.event_num] = {}
                    for node_type in event_items.keys():
                        if node_type not in self.event_dict[self.event_num].keys():
                            self.event_dict[self.event_num][node_type] = []
                            self.event_dict_int[self.event_num][node_type] = []
                        self.event_dict[self.event_num][node_type] = event_items[node_type]
                        self.event_dict_int[self.event_num][node_type] = event_items_int[node_type]

                    self.event_num += 1
                else:
                    exist_event_idx = self.event_identifier[event_id]
                    exist_event_items = self.event_dict[exist_event_idx]
                    exist_event_items_int = self.event_dict_int[exist_event_idx]
                    for node_type in event_items.keys():
                        if node_type not in exist_event_items.keys():
                            exist_event_items[node_type] = []
                            exist_event_items_int[node_type] = []
                        self.event_dict[exist_event_idx][node_type] = list(set(exist_event_items[node_type] + event_items[node_type]))
                        self.event_dict_int[exist_event_idx][node_type] = list(set(exist_event_items_int[node_type] + event_items_int[node_type]))
        
        f.close()
        print('finished loading data.')

    
    def load_data_edge(self, filename, event_id_idx):
        print('loading data...')
        with open(filename, 'r') as f:
            for line in f:
                item1, item2, weight = line.strip().split()
                items = [item1, item2]
                event_id = ''
                for idx in event_id_idx:
                    event_id += items[idx]
                event_items = {}
                event_items_int = {}
                count = 0
                for item in items:
                    indx = self.encode_node(item)
                    if item[0] not in event_items.keys():
                        event_items[item[0]] = []
                        event_items_int[item[0]] = []
                    event_items[item[0]].append(self.node_ind[item[0]][indx])
                    event_items_int[item[0]].append(indx)
                if event_id not in self.event_identifier.keys():
                    self.event_identifier[event_id] = self.event_num
                    self.event_dict[self.event_num] = {}
                    self.event_dict_int[self.event_num] = {}
                    for node_type in event_items.keys():
                        if node_type not in self.event_dict[self.event_num].keys():
                            self.event_dict[self.event_num][node_type] = []
                            self.event_dict_int[self.event_num][node_type] = []
                        self.event_dict[self.event_num][node_type] = event_items[node_type]
                        self.event_dict_int[self.event_num][node_type] = event_items_int[node_type]

                    self.event_num += 1
                else:
                    exist_event_idx = self.event_identifier[event_id]
                    exist_event_items = self.event_dict[exist_event_idx]
                    exist_event_items_int = self.event_dict_int[exist_event_idx]
                    for node_type in event_items.keys():
                        if node_type not in exist_event_items.keys():
                            exist_event_items[node_type] = []
                            exist_event_items_int[node_type] = []
                        self.event_dict[exist_event_idx][node_type] = list(set(exist_event_items[node_type] + event_items[node_type]))
                        self.event_dict_int[exist_event_idx][node_type] = list(set(exist_event_items_int[node_type] + event_items_int[node_type]))

                if item2[0] not in self.node_event_weight.keys():
                    self.node_event_weight[item2[0]] = {}
                if self.event_identifier[event_id] not in self.node_event_weight[item[0]].keys():
                    self.node_event_weight[item2[0]][self.event_identifier[event_id]] = {}
                self.node_event_weight[item2[0]][self.event_identifier[event_id]][indx] = weight

        f.close()
        print('finished loading data.')
                

    def encode_node(self, node_name):
        if node_name[0] not in self.node_ind.keys():
            self.node_ind[node_name[0]] = {}
        if node_name[0] not in self.nodes_num.keys():
            self.nodes_num[node_name[0]] = 0

        if node_name not in self.node_ind[node_name[0]].values():
            self.node_ind[node_name[0]][self.nodes_num[node_name[0]]] = node_name
            indx = self.nodes_num[node_name[0]]
            self.nodes_num[node_name[0]] += 1

            return indx
        else:
            return list(self.node_ind[node_name[0]].keys())[list(self.node_ind[node_name[0]].values()).index(node_name)]


class EventNet(object):

    def __init__(self, event_dict, nodes_num, event_num, node_types, node_event_weight=None):
        self.event_dict = event_dict
        self.nodes_num = nodes_num
        self.event_num = event_num
        self.node_types = node_types
        self.node_event_weight = node_event_weight

        self.inc_mat = {}
        self.event_mat = None
        self.event_deg = {}
        self.node_deg = {}
        self.event_mat_deg = None
        self.enet = nx.DiGraph()

        self.incidence_matrix()
        self.event_degree()
        self.node_degree()
    
    def incidence_matrix(self):
        print('generating incidence matrix...')
        inc_mat = {}
        if self.node_event_weight is None:
            for node_type in self.node_types:
                inc_mat[node_type] = np.zeros((self.event_num, self.nodes_num[node_type]))
                for event in self.event_dict.keys():
                    if node_type in self.event_dict[event].keys():
                        inc_mat[node_type][event][self.event_dict[event][node_type]] = 1

                self.inc_mat[node_type] = inc_mat[node_type].T
                
        else:
            for node_type in self.node_types:
                inc_mat[node_type] = np.zeros((self.event_num, self.nodes_num[node_type]))
                for event in self.event_dict.keys():
                    if node_type in self.event_dict[event].keys():
                        for node in self.event_dict[event][node_type]:
                            inc_mat[node_type][event][node] = self.node_event_weight[node_type][event][node]

                self.inc_mat[node_type] = inc_mat[node_type].T
        print('finished generating incidence matrix')

    def event_degree(self):
        for node_type in self.node_types:
            self.event_deg[node_type] = self.inc_mat[node_type].sum(axis=0)

    def node_degree(self):
        for node_type in self.node_types:
            self.node_deg[node_type] = np.diag(self.inc_mat[node_type].sum(axis=1))

        
if __name__ == '__main__':
    data = LoadData()
    data.load_data_edge(filename='douban.txt', event_id_idx=[0])

    event = EventNet(event_dict=data.event_dict_int, 
                     nodes_num=data.nodes_num, 
                     event_num=data.event_num, 
                     node_types=['m', 'a', 'd', 'u'])

    print('------inc_mat------')
    print(event.inc_mat['u'])

