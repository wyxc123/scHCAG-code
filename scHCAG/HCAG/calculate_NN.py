import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_dict_mnn_para(data_matrix, batch_index, k=10, metric='cosine', approx=True, return_distance=False, njob=1):
    
    batches = np.unique(batch_index)
    if len(batches) < 2:
        raise ValueError("至少需要两个不同的批次")
    
    mnn_pairs = []
    distances = [] if return_distance else None
    
    
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            batch_a = batches[i]
            batch_b = batches[j]
            
            
            mask_a = (batch_index == batch_a)
            mask_b = (batch_index == batch_b)
            data_a = data_matrix[mask_a]
            data_b = data_matrix[mask_b]
            indices_a = np.where(mask_a)[0]  
            indices_b = np.where(mask_b)[0]  
            
            
            nn_a = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=njob)
            nn_a.fit(data_b)
            neighbors_a = nn_a.kneighbors(data_a, return_distance=return_distance)
            
            
            
            nn_b = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=njob)
            nn_b.fit(data_a)
            neighbors_b = nn_b.kneighbors(data_b, return_distance=return_distance)
            
            
            
            for a_idx in range(len(data_a)):
                
                b_neighbors = neighbors_a[1][a_idx]
                
                for b_local in b_neighbors:
                    
                    a_neighbors = neighbors_b[1][b_local]
                    if a_idx in a_neighbors:
                        
                        global_a = indices_a[a_idx]
                        global_b = indices_b[b_local]
                        mnn_pairs.append((global_a, global_b))
                        
                        
                        if return_distance:
                            dist_a = neighbors_a[0][a_idx][np.where(b_neighbors == b_local)[0][0]]
                            dist_b = neighbors_b[0][b_local][np.where(a_neighbors == a_idx)[0][0]]
                            distances.append((dist_a + dist_b) / 2)
    
    if return_distance:
        return mnn_pairs, distances
    else:
        return mnn_pairs
