from typing import Dict, List, Optional
import faiss, time
from tqdm import tqdm 
import numpy as np
from utils import pickle_load

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter:
    """Computes and stores the average and current value on device"""

    def __init__(self, device, length):
        self.device = device
        self.length = length
        self.reset()

    def reset(self):
        self.values = torch.zeros(self.length, device=self.device, dtype=torch.float)
        self.counter = 0
        self.last_counter = 0

    def append(self, val):
        self.values[self.counter] = val.detach()
        self.counter += 1
        self.last_counter += 1

    @property
    def val(self):
        return self.values[self.counter - 1]

    @property
    def avg(self):
        return self.values[:self.counter].mean()

    @property
    def values_list(self):
        return self.values[:self.counter].cpu().tolist()

    @property
    def last_avg(self):
        if self.last_counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = self.values[self.counter - self.last_counter:self.counter].mean()
            self.last_counter = 0
            return self.latest_avg


@torch.no_grad()
def recall_at_ks(query_features: torch.Tensor,
                 query_labels: torch.LongTensor,
                 ks: List[int],
                 gallery_features: Optional[torch.Tensor] = None,
                 gallery_labels: Optional[torch.Tensor] = None,
                 cosine: bool = False) -> Dict[int, float]:
    """
    Compute the recall between samples at each k. This function uses about 8GB of memory.

    Parameters
    ----------
    query_features : torch.Tensor
        Features for each query sample. shape: (num_queries, num_features)
    query_labels : torch.LongTensor
        Labels corresponding to the query features. shape: (num_queries,)
    ks : List[int]
        Values at which to compute the recall.
    gallery_features : torch.Tensor
        Features for each gallery sample. shape: (num_queries, num_features)
    gallery_labels : torch.LongTensor
        Labels corresponding to the gallery features. shape: (num_queries,)
    cosine : bool
        Use cosine distance between samples instead of euclidean distance.

    Returns
    -------
    recalls : Dict[int, float]
        Values of the recall at each k.

    """
    device = query_features.device 

    offset = 0
    if gallery_features is None and gallery_labels is None:
        offset = 1
        gallery_features = query_features
        gallery_labels = query_labels
    elif gallery_features is None or gallery_labels is None:
        raise ValueError('gallery_features and gallery_labels needs to be both None or both Tensors.')

    if cosine:
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_f, q_l, g_f, g_l = map(to_cpu_numpy, [query_features, query_labels, gallery_features, gallery_labels])
    max_k = max(ks)

    index = faiss.IndexFlatL2(g_f.shape[1])
    # if device == torch.device(type='cpu'):
    #     index = faiss.IndexFlatL2(g_f.shape[1])
    #     # if faiss.get_num_gpus() > 0:
    #     #     index = faiss.index_cpu_to_all_gpus(index)
    # else:
    #     res = faiss.StandardGpuResources()
    #     flat_config = faiss.GpuIndexFlatConfig()
    #     flat_config.device = 0
    #     index = faiss.GpuIndexFlatL2(res, g_f.shape[1], flat_config)
    print('--------------------------------------------')
    print('FAISS initialized')
    index.add(g_f)
    print('--------------------------------------------')
    print('Index features added, start KNN searching')
    closest_dists, closest_indices = index.search(q_f, max_k + offset)
    print('--------------------------------------------')
    closest_dists = closest_dists[:,:(int(max_k) + offset)]
    closest_indices = closest_indices[:,:(int(max_k) + offset)]

    recalls = {}
    for k in ks:
        indices = closest_indices[:, offset:k + offset]
        recalls[k] = (q_l[:, None] == g_l[indices]).any(1).mean()
    return {k: round(v * 100, 2) for k, v in recalls.items()}, closest_dists[:, offset:], closest_indices[:, offset:]


@torch.no_grad()
def recall_at_ks_rerank(
    query_features: torch.Tensor,
    query_labels: torch.LongTensor,
    ks: List[int],
    model: nn.Module,
    cache_nn_inds: torch.Tensor,
    gallery_features: Optional[torch.Tensor] = None,
    gallery_labels: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """
    Compute the recall between samples at each k. This function uses about 8GB of memory.

    Parameters
    ----------
    query_features : torch.Tensor
        Features for each query sample. shape: (num_queries, num_features)
    query_labels : torch.LongTensor
        Labels corresponding to the query features. shape: (num_queries,)
    ks : List[int]
        Values at which to compute the recall.
    gallery_features : torch.Tensor
        Features for each gallery sample. shape: (num_queries, num_features)
    gallery_labels : torch.LongTensor
        Labels corresponding to the gallery features. shape: (num_queries,)
    cosine : bool
        Use cosine distance between samples instead of euclidean distance.

    Returns
    -------
    recalls : Dict[int, float]
        Values of the recall at each k.

    """

    ground_truth = pickle_load("/content/AML_Rerank_MobileNet/rrt_sop_caches/rrt_r50_sop_nn_inds_positives.pkl")
    ground_truth = np.array(ground_truth)

    print(f"Query features and labels shape: {query_features.shape} {query_labels.shape}")
    print(f"Gallery features and labels shape: {gallery_features.shape} {gallery_labels.shape}")
    if gallery_features is None and gallery_labels is None:
        gallery_features = query_features
        gallery_labels = query_labels
    elif gallery_features is None or gallery_labels is None:
        raise ValueError('gallery_features and gallery_labels needs to be both None or both Tensors.')

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_l, g_l = map(to_cpu_numpy, [query_labels, gallery_labels])

    device = next(model.parameters()).device

    print(f"cache_nn_inds size: {cache_nn_inds.size()}")

    num_samples, top_k = cache_nn_inds.size()
    print(f"num_samples: {num_samples}")
    top_k = min(top_k, 100)
    top_k = gallery_features.shape[0]
    _, fsize, h, w = query_features.size()
    scores = []

    #gallery_features = torch.gather(gallery_features, dim=0, index=cache_nn_inds[0:100])

    gallery_batches = gallery_features.split(1000)

    reranking_size = 100

    i = 0

    for query in tqdm(query_features):
        gallery_batch = torch.index_select(gallery_features, dim=0, index=cache_nn_inds[i, 0:reranking_size]).split(1000)
        i += 1
        k_scores = []
        for gallery in gallery_batch:
            gallery_size = gallery.size()[0]
            query_batch = query.repeat(gallery_size, 1, 1, 1)
            current_score = model(None, True, src_global=None, src_local=query_batch.to(device),
                                  tgt_global=None, tgt_local=gallery.to(device))
            k_scores.append(current_score.cpu())
            #print(f"k_scores len: {len(k_scores)}")
        k_scores = torch.cat(k_scores, 0)
        #print(f"k_scores size: {k_scores.size()}")
        scores.append(k_scores)

    # bsize = batch_size for reranking
    # Changed.
    bsize = min(num_samples, 500)
    total_time = 0.0
    ######################################################################
    """
    print('--------------------------------------------')
    print('Start reranking')
    for i in tqdm(range(top_k)): #top_k = 20
        k_scores = []
        for j in range(0, num_samples, bsize):
            current_query = query_features[j:(j+bsize)]
            current_index = gallery_features[cache_nn_inds[j:(j+bsize), i]]
            start = time.time()
            current_scores = model(None, True, src_global=None, src_local=current_query.to(device), 
                tgt_global=None, tgt_local=current_index.to(device))
            end = time.time()
            total_time += end-start
            k_scores.append(current_scores.cpu())
        k_scores = torch.cat(k_scores, 0)
        scores.append(k_scores)
    """
    scores = torch.stack(scores, 0)

    with open("/content/drive/MyDrive/scores.pkl", "wb+") as f:
        pickle.dump(scores, f)

    closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
    closest_dists = closest_dists.numpy()
    # closest_indices = torch.zeros((num_samples, top_k)).long()
    # for i in range(num_samples):
    #     for j in range(top_k):
    #         closest_indices[i, j] = cache_nn_inds[i, indices[i, j]]
    # closest_indices = closest_indices.numpy()
    closest_indices = torch.gather(cache_nn_inds[:, 0:reranking_size], -1, indices).numpy()   #predictions

#### For each query, check if the predictions are correct
    #ground_truth = np.array(cache_nn_inds) # ground truth
    predictions = np.array(closest_indices)
    recalls = np.zeros(len(ks))

    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(ks): #1, 5, 10, 20
            if np.any(np.in1d(preds[:n], cache_nn_inds[query_index][:n])):
                recalls[i:] += 1
                break
    
    recalls = recalls / query_features.size(0) * 100

    print(f"Recalls: {recalls}")

    return recalls, closest_dists, indices

