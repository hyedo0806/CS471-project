import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn

def read_graph_nodes_relations(data):
  graph_ids = list(data)
  
  nodes, graphs = {}, {}
  for node_id, graph_id in zip(data.index, graph_ids):
      if graph_id not in graphs:
          graphs[graph_id] = []
      graphs[graph_id].append(node_id)
      nodes[node_id] = graph_id
  
  graph_ids = np.unique(list(graphs.keys()))
  for graph_id in graphs:
      graphs[graph_id] = np.array(graphs[graph_id])
  return nodes, graphs

def make_edge(graphs):
  global nodes
  global deleteGraph
  global deleteNode
  edges = []
   
  for graph in graphs:
    try :
      for i in range(5):
        edge = [graphs[graph][i], graphs[graph][i+5]]
        edges.append(edge)

        edge = [graphs[graph][i+5], graphs[graph][i]]
        edges.append(edge)
      
      for i in range(5):
        for j in range(5):
          if i != j :
            edge = [graphs[graph][i], graphs[graph][j]]
            edges.append(edge)

            edge = [graphs[graph][i+5], graphs[graph][j+5]]
            edges.append(edge)

    except:
     deleteGraph.append(graph)
      

  for g in deleteGraph:
    for _n in graphs[g]:
      deleteNode.append(_n)
      del nodes[_n]
    del graphs[g]

  return edges

def read_graph_adj(nodes, graphs):
  edges = make_edge(graphs)

  adj_dict = {}
  for edge in edges:
      node1 = edge[0] # -1 because of zero-indexing in our code
      node2 = edge[1]
      graph_id = nodes[node1]
      assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
      if graph_id not in adj_dict:
          n = len(graphs[graph_id])
          adj_dict[graph_id] = np.zeros((n, n))
      ind1 = np.where(graphs[graph_id] == node1)[0]
      ind2 = np.where(graphs[graph_id] == node2)[0]
      assert len(ind1) == len(ind2) == 1, (ind1, ind2)
      adj_dict[graph_id][ind1, ind2] = 1
      
  adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
  
  return adj_list


def itemEncode(_item, column):
 
    for i in tqdm(range(1,7)):
        locals()['_item{}'.format(i)] = pd.get_dummies(_item['item'+str(i)])
        locals()['_item{}'.format(i)] = locals()['_item{}'.format(i)].reindex(columns=uniqueItemList, fill_value=0)
        
        if i==1: data = locals()['_item{}'.format(i)]
        else : data += locals()['_item{}'.format(i)]

    return data

def champEncode(data):
    pass

def calculate_degree(num_node, edges):
    deleteEdges = []
    degree = torch.zeros(num_node, dtype=torch.float64)
    for v1, v2 in tqdm(edges):
      try:
        degree[pair[v1]] += 1
      except : 
         deleteEdges.append([v1, v2])

    print(deleteEdges[:5])
    print(len(deleteEdges))
    exit()
    for item in tqdm(deleteEdges):
       edges.remove(item)
         
    return degree, edges


if __name__=="__main__":
    # champInfo = pd.read_csv("champs.csv")
    # match = pd.read_csv("matches.csv")
    participants = pd.read_csv("participants.csv")
    stats1 = pd.read_csv("stats1.csv")
    stats2 = pd.read_csv("stats2.csv")

    stats1 = stats1[['id', 'kills', 'deaths', 'assists',"item1", "item2","item3","item4","item5","item6" ]]
    stats2 = stats2[['id', 'kills', 'deaths', 'assists',"item1", "item2","item3","item4","item5","item6" ]]
    stats1 = stats1.set_index('id')
    stats2 = stats2.set_index('id')
    participants = participants.set_index('id')

    stats = pd.concat([stats1, stats2], axis=0).sort_index()
    participants = participants[participants.index.isin(stats.index)]

    #       ** --- KDA --- **
    kda = stats[["kills", "deaths", "assists"]]
    # kda.to_csv('kda.txt', sep='\t', index=True)

    #       ** --- ITEM onehotEncode --- **
    uniqueItemList = np.unique(stats[["item1", "item2","item3","item4","item5","item6"]])
    itemE = itemEncode(stats, uniqueItemList)
    # itemE.to_csv('itemEncode.txt', sep='\t', index=True)
    
    #       ** --- CHAMP onehotEncode --- **
    champE = pd.get_dummies(participants['championid'])
    # champE.to_csv('champEncode.txt', sep='\t', index=True)

    #       ** --- POS onehotEncode --- **
    posE = pd.get_dummies(participants['position'])
    # posE.to_csv('champEncode.txt', sep='\t', index=True)

    #       ** --- node feature ___ **
    deleteGraph, deleteNode = [], []
    # num_nodes : num_players
    # num_graphs : num_matches(list of playerid)
    featureE = pd.concat([itemE, champE, posE, kda], axis=1, join="inner")
    
    nodes, graphs = read_graph_nodes_relations(participants["matchid"])
    edges = make_edge(graphs)

    feat= featureE[~featureE.index.isin(deleteNode)]
    
    pair = {}
    for i,idx in enumerate(feat.index):
        pair[i] = idx

    num_node = feat.shape[0]
    dim_feat = feat.shape[1]

    degree, edges = calculate_degree(num_node, edges)

    idx_shuffle = list(range(num_node))
    random.shuffle(idx_shuffle)

    idx_train = idx_shuffle[:int(0.8 * num_node)]
    idx_valid = idx_shuffle[int(0.8 * num_node):int(0.9 * num_node)]
    idx_test = idx_shuffle[int(0.9 * num_node):]

    

    