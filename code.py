import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold
torch.autograd.set_detect_anomaly(True)


def read_graph_nodes_relations(data):

  dataFiltered = data.groupby("matchid").filter(lambda x: len(x) == 10)
  graph = dataFiltered['matchid'].unique()
 
  return [i for i in range(graph.shape[0]*10)], graph

def edgeAndDegree(num, nodes):
  
  edges = torch.zeros((50*num,2))
  cnt = 0
  for k in tqdm(range(num)):
    for i in range(5):
      for j in range(5):
        if i!=j : 
          edges[cnt][0] = i+k*10
          edges[cnt][1] = j+k*10
          edges[cnt+1][0] = i+5+k*10
          edges[cnt+1][1] = j+5+k*10   
        else: 
          edges[cnt][0] = i+k*10
          edges[cnt][1] = i+5+k*10
          edges[cnt+1][0] = i+5+k*10
          edges[cnt+1][1] = i+k*10
        cnt +=2

          
  degrees = torch.empty((10 * num,))
  degrees.fill_(5)

  return edges.to(device), degrees.to(device)


class GraphSageLayer(nn.Module):
  def __init__(self, dim_in, dim_out, agg):
    super(GraphSageLayer, self).__init__()

    self.dim_in = dim_in
    self.dim_out = dim_out
    self.agg = agg
    self.act = nn.ReLU()

    if self.agg == 'gcn':
      self.weight = nn.Linear(self.dim_in, self.dim_out, bias=False, dtype=torch.float32) # W_l
      self.bias = nn.Linear(self.dim_in, self.dim_out, bias=False, dtype=torch.float32) # B_l

    elif self.agg == 'mean':
      self.weight = nn.Linear(2 * self.dim_in, self.dim_out, bias=False, dtype=torch.float32) # W_l

    elif self.agg == 'maxpool':
      self.linear_pool = nn.Linear(self.dim_in, self.dim_in, bias=True, dtype=torch.float32) # W_{pool}, b
      self.weight = nn.Linear(2 * self.dim_in, self.dim_out, bias=False, dtype=torch.float32) # W_l
                      
  def forward(self, feat, edge, degree):
    if self.agg == 'gcn':   
      
      indices = edge[:,1].long()
      feat_t = feat[indices]
      idx_h = edge[:, 0]
      agg_neighbor = torch.zeros(feat.shape[0], feat.shape[1], dtype=torch.float32).to(device).index_add_(0, idx_h.long(), feat_t.float())

      inv_degree = torch.where(degree == 0.0, 1.0, 1.0 / degree).unsqueeze(-1)
      agg = agg_neighbor * inv_degree
      return F.normalize(self.act(self.weight(agg) + self.bias(feat)), 2, -1)
    
    elif self.agg == 'mean':
    # TODO: Implement GraphSAGE(Mean) layer (Hint: Use index_add_())

      feat_t = feat[edge[:, 1].long()]
      idx_h = edge[:, 0]
      agg_neighbor = torch.zeros(feat.shape[0], feat.shape[1], dtype=torch.float32).index_add_(0, idx_h.long(), feat_t)
      
      inv_degree = torch.where(degree == 0.0, 1.0, 1.0 / degree).unsqueeze(-1)
      agg = agg_neighbor * inv_degree
      return F.normalize(self.act(self.weight(torch.cat((agg, feat), 1))), 2, -1)

    elif self.agg == 'maxpool':
      # TODO: Implement GraphSAGE(Maxpool) layer (Hint: Use scatter_reduce)
      feat = self.act(self.linear_pool(feat))
      feat_t = feat[edge[:, 1]]
      idx_h = edge[:, 0]
      scatter_idx = idx_h.unsqueeze(-1).repeat(1, feat.shape[1])
      
      agg = torch.zeros(feat.shape[0], feat.shape[1], dtype=torch.float32).scatter_reduce(0, scatter_idx, feat_t, reduce='amax', include_self=False)
    
      return F.normalize(self.act(self.weight(torch.cat((agg, feat), 1))), 2, -1)


class GraphSage(nn.Module):
  def __init__(self, num_layers, dim_in, dim_hidden, dim_out, agg):
                    # 2,       dim_feat,     128,       2,    'gcn'
    super(GraphSage, self).__init__()

    self.num_layers = num_layers
    self.dim_in = dim_in
    self.dim_hidden = dim_hidden
    self.dim_out = dim_out
    self.agg = agg

    layers = [GraphSageLayer(self.dim_in, self.dim_hidden, agg)]
    for _ in range(num_layers - 1):
      layers.append(GraphSageLayer(self.dim_hidden, self.dim_hidden, agg))

    self.layers = nn.ModuleList(layers)

    self.classifier = Classifier()    

  def forward(self, feat, edge, degree):
    list_feat = [feat]

    for layer in self.layers:
      list_feat.append( layer(list_feat[-1], edge, degree))

    out = self.classifier(list_feat[-1])

    return out

                              
class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()
  
    self.sigmoid = nn.Sigmoid()
    self.classifier = nn.Linear(32, 1, dtype=torch.float32)

  def forward(self, x):

    out = self.classifier(x)
    out = self.sigmoid(out)

    return out

### trainset index를 셔플하지 않고 그래도 순차적으로 사용하면 이 코드를 사용할 수 있으나, 아래와 같이 index를 random하게 사용하면 다른 코드를 작성해야함. 하지만 계산 시간이 걸린다.
def win_loss(out):
  new_x = out.clone()  # 새로운 텐서를 생성하여 결과를 저장할 준비

  for i in tqdm(range(0, out.shape[0], 10)):  # 10개씩 묶음을 만들기 위해 0부터 18000까지 10씩 증가하는 인덱스 사용
      batch = out[i:i+10]  # 10개씩 묶음을 선택

      first_sum = torch.sum(batch[:5])  # 묶음의 앞쪽 5개 요소의 합
      second_sum = torch.sum(batch[5:])  # 묶음의 다음 5개 요소의 합

      if first_sum > second_sum:
          new_x[i:i+5] = torch.full((5, 1), 0.9)  # 앞쪽 5개 요소의 합이 큰 경우 0.9로 대체
          new_x[i+5:i+10] = torch.full((5, 1), 0.1)  # 앞쪽 5개 요소의 합이 큰 경우 0.9로 대체
      else:
          new_x[i:i+5] = torch.full((5, 1), 0.1)  # 앞쪽 5개 요소의 합이 큰 경우 0.9로 대체
          new_x[i+5:i+10] = torch.full((5, 1), 0.9)  # 앞쪽 5개 요소의 합이 큰 경우 0.9로 대체
 

  return new_x


def train(model, agg, feat, edge, degree, label, dim_hidden=128, dim_out=7,
          lr=0.001, num_epoch=200):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  loss_fn = nn.BCELoss()

  best_valid = -1
  list_valid_f1 = []
  list_loss = []

  for epoch in range(num_epoch):
    ## ----- random index for training ( lab3 참고 )
    idx_shuffle = list(range(num_node))
    random.shuffle(idx_shuffle)
    idx_train = idx_shuffle[:int(0.8 * num_node)]
    idx_valid = idx_shuffle[int(0.8 * num_node):int(0.9 * num_node)]


    optimizer.zero_grad()
    target = label[idx_train]

    # TODO: Compute output features
    feature = model(feat, edge, degree)
    
    # TODO: Calculate loss funciton using loss_fn
    loss = loss_fn(feature[idx_train], target)

    loss.backward()
    optimizer.step()

    list_loss.append(loss.item())

    #### Validation ####
    model.eval()
    with torch.no_grad():
      target = label[idx_valid]

      # TODO: Compute output features
      feature = model(feat, edge, degree)
      
      # TODO: Extract predicted labels
      _, pred = torch.max(feature[idx_valid], 1)
      pred = pred.detach().cpu()

      # TODO: Calculate F1 score (micro) using f1_score()
      f1_val = f1_score(target, pred, average='micro')

      list_valid_f1.append(f1_val)
      print(f"F1 Score: {f1_val}")

      if f1_val > best_valid:
        print("Checkpoint updated!")
        torch.save(model, f'model-{agg}-{epoch}th.pt')
        best_valid = f1_val

    model.train()
  
  return list_loss, list_valid_f1

def visualize(num_epoch, list_loss, list_valid_f1, title):
  t = np.arange(num_epoch)

  fig, ax1 = plt.subplots()
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  line1 = ax1.plot(t, list_loss, color='blue', label='Loss')

  ax2 = ax1.twinx()
  ax2.set_ylabel('F1 Score (Micro)')
  line2 = ax2.plot(t, list_valid_f1, color='orange', label='F1 Score (Micro)')

  lines = line1 + line2
  labels = [line.get_label() for line in lines]

  ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.1, 1))

  plt.title(title)
  plt.savefig("f{}_loss.png".format(num_epoch))


if __name__=="__main__":

    device = torch.device('cpu')

    trainsetEncoded = pd.read_csv("trainset.txt", delimiter="\t")
    trainsetEncoded = trainsetEncoded.drop(["Unnamed: 0", "team"], axis=1)
    ### ----- trainsetEncoded columns
    #  Index(['win', 'kills', 'deaths', 'assists', 'largestkillingspree',
    #        'largestmultikill', 'killingsprees', 'longesttimespentliving',
    #        'doublekills', 'triplekills', 'quadrakills', 'pentakills',
    #        'legendarykills', 'totdmgdealt', 'magicdmgdealt', 'physicaldmgdealt',
    #        'truedmgdealt', 'largestcrit', 'totdmgtochamp', 'magicdmgtochamp',
    #        'physdmgtochamp', 'truedmgtochamp', 'totheal', 'totunitshealed',
    #        'dmgselfmit', 'dmgtoobj', 'dmgtoturrets', 'visionscore', 'totdmgtaken',
    #        'magicdmgtaken', 'physdmgtaken', 'truedmgtaken', 'goldearned',
    #        'goldspent', 'turretkills', 'inhibkills', 'totminionskilled',
    #        'neutralminionskilled', 'ownjunglekills', 'enemyjunglekills',
    #        'totcctimedealt', 'champlvl', 'pinksbought', 'wardsbought',
    #        'wardsplaced', 'wardskilled', 'firstblood', 'matchid', 'BOT', 'JUNGLE',
    #        'MID', 'SUPPORT', 'TOP'],
    #       dtype='object')
    
    nodes, graphs = read_graph_nodes_relations(trainsetEncoded[["matchid"]])

    ## ----- X : feature, Y : label 
    featureE = trainsetEncoded.drop(['win', 'matchid'], axis=1)
    label = trainsetEncoded[["win"]]

    ## ----- data 를 tensor 형식으로 변환
    featN = featureE.to_numpy().astype(np.float32)
    featT = torch.from_numpy(featN).to(device) 

    labelN = label.to_numpy().astype(np.float32)
    labelT = torch.from_numpy(labelN).to(device)

    num_node = featureE.shape[0]
    dim_feat = featureE.shape[1]
    
    ## ----- edge & degree
    edgeT, degreeT = edgeAndDegree(num_node//10, nodes)

    ## -----  모델 구조 lab3 참고
    mode = 'gcn'
    model = GraphSage(2, dim_feat, 32, 2, mode).to(device)
    list_loss_gcn, list_valid_f1_gcn = train(model, mode, featT, edgeT, degreeT, labelT)
    
    torch.save(model.state_dict(), 'model.pth')
    
# #### 5/27 19:33 기준 F1 score log  ####
# 0.5로 계속 유지 -> 학습이 되질 않는다.

# F1 Score: 0.49916627759621157
# Checkpoint updated!
# F1 Score: 0.5007753618355233
# Checkpoint updated!
# F1 Score: 0.5009337690922431
# Checkpoint updated!
# F1 Score: 0.49964983659040885
# F1 Score: 0.5010338157806976
# Checkpoint updated!
# F1 Score: 0.4967818315213766
# F1 Score: 0.5008420596278264
# F1 Score: 0.4988244514106583
# F1 Score: 0.5011088507970386
# Checkpoint updated!
# F1 Score: 0.5024177949709865
# Checkpoint updated!
# F1 Score: 0.4984409391049156
# F1 Score: 0.5008754085239778
# F1 Score: 0.4991245914760221
# F1 Score: 0.49949142933368906
# F1 Score: 0.5029847262055626
# Checkpoint updated!
# F1 Score: 0.49985826719135595
# F1 Score: 0.5004168612018942
# F1 Score: 0.5010421530047355
# F1 Score: 0.4990412192356433
# F1 Score: 0.4978823450943774
# F1 Score: 0.5007003268191823
# F1 Score: 0.5007336757153338
# F1 Score: 0.5003585006336291
# F1 Score: 0.49926632428466616
# F1 Score: 0.49810745014340024
# F1 Score: 0.5008003735076368
# F1 Score: 0.5017174681518042
# F1 Score: 0.5010421530047355
# F1 Score: 0.5002751283932502
# F1 Score: 0.5013923164143267
# F1 Score: 0.4998499299673181
# F1 Score: 0.5006336290268792
# F1 Score: 0.5013172813979857
# F1 Score: 0.5005085706663109
# F1 Score: 0.5001667444807577
# F1 Score: 0.5002751283932502
# F1 Score: 0.49944140598946174
# F1 Score: 0.4999499766557727
# F1 Score: 0.5027262722603881
# F1 Score: 0.5004752217701595
# F1 Score: 0.49947475488561327
# F1 Score: 0.5003334889615154
# F1 Score: 0.4998332555192423
# F1 Score: 0.4980157406789835
# F1 Score: 0.5012589208297206
# F1 Score: 0.5008503968518642
# F1 Score: 0.5002334422730608
# F1 Score: 0.4976572400453545
# F1 Score: 0.4998999533115454