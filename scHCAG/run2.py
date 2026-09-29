import anndata as ad
import numpy as np
import os, argparse
from HCAG import HCAGModel
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from HCAG.utils import accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='chen', help='dataset name')
    parser.add_argument('--t', type=int, default=1, help='number of laplacian filtering')
    parser.add_argument('--highly_genes', type=int, default=2000, help='number of highly variable genes')
    parser.add_argument('--k', type=int, default=18, help='number of nearest neighbors')
    parser.add_argument('--target_clusters', type=int, default=0, help='number of clusters to assign')
    parser.add_argument('--wzinb', type=float, default=0, help='fixed weight of ZINB loss')
    parser.add_argument('--init_method', type=str, default='kaiming_uniform_', help='initialize method of linear layer')
    parser.add_argument('--use_gisi', action='store_true', help='Use GISI for clustering instead of Leiden/Louvain')
    
    parser.add_argument('--no_weighting', action='store_true', default=False, help='Disable hard sample weighting')
    parser.add_argument('--no_gisi_pseudo', action='store_true', default=False, help='Use Leiden for pseudo-label generation')
    parser.add_argument('--no_gisi_final', action='store_true', default=False, help='Use Leiden for final clustering')
    parser.add_argument('--no_zinb', action='store_true', default=False, help='Disable ZINB loss')
    
    
    args = parser.parse_args()
    dataset = args.dataset
    t = args.t
    highly_genes = args.highly_genes
    k = args.k
    target_clusters = args.target_clusters
    wzinb = args.wzinb
    init_method = args.init_method


    use_weighting = not args.no_weighting
    use_gisi_pseudo = not args.no_gisi_pseudo
    use_gisi_final = not args.no_gisi_final
    use_zinb = not args.no_zinb



    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    file = f'./temp/HCAG.csv'
    save_dir = './temp'

    try:
        adata_raw = ad.read(f'./data/{dataset}.h5ad')
        adata_raw.raw = adata_raw

        assert target_clusters != 0 or 'cluster' in adata_raw.obs, \
        "Either target_clusters must be specified, or 'cluster' must exist in adata_raw.obs."
        nclusters = len(np.unique(adata_raw.obs['cluster'])) if target_clusters == 0 else target_clusters

        log_path = f"{save_dir}/{dataset}_log.txt"

        hcag = HCAGModel(log_path=log_path, info=True)
        adata = hcag.preprocess(adata_raw, t=t, k=k, highly_genes=highly_genes)
        hcag.train(adata, target_clusters=nclusters, wzinb=wzinb, init_method=init_method, use_weighting=use_weighting,
            use_gisi_pseudo=use_gisi_pseudo,
            use_gisi_final=use_gisi_final,
            use_zinb=use_zinb)



        
        true_labels = [str(x) for x in adata.obs["cluster"].values]
        pred_labels = [str(x) for x in adata.obs["reassign_cluster"].values]

        
        true_k = len(np.unique(true_labels))

        
        pred_k = len(np.unique(pred_labels))

        
        k_diff = pred_k - true_k

        
        k_match = int(true_k == pred_k)

        ari = adjusted_rand_score(pred_labels, true_labels)
        nmi = normalized_mutual_info_score(pred_labels, true_labels)

        
        le = LabelEncoder()
        combined = np.concatenate([true_labels, pred_labels])
        le.fit(combined)
        true_int = le.transform(true_labels)
        pred_int = le.transform(pred_labels)
        acc = accuracy(pred_int, true_int)

        adata.write(f"{save_dir}/{dataset}_hcag.h5ad", compression='gzip')

        
        write_header = (not os.path.exists(file)) or os.path.getsize(file) == 0

        with open(file, 'a+', encoding='utf-8') as f:
            if write_header:
                f.write('dataset,true_k,pred_k,k_diff,k_match,nmi,ari,acc\n')

            f.write(
                f'{dataset},'
                f'{true_k},'
                f'{pred_k},'
                f'{k_diff},'
                f'{k_match},'
                f'{nmi},'
                f'{ari},'
                f'{acc}\n'
            )



        
    except Exception as e:
        write_header = (not os.path.exists(file)) or os.path.getsize(file) == 0

        with open(file, 'a+', encoding='utf-8') as f:
            if write_header:
                f.write('dataset,true_k,pred_k,k_diff,k_match,nmi,ari,acc,error\n')

            f.write(
                f'{dataset},'
                f'ERROR,'
                f'ERROR,'
                f'ERROR,'
                f'ERROR,'
                f'ERROR,'
                f'ERROR,'
                f'ERROR,'
                f'{str(e).replace(",", ";")}\n'
            )

if __name__ == '__main__':
    main()
