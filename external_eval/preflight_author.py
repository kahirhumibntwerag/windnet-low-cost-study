"""Fixed first lexical historical block per fold: reference equivalence, no tuning."""
import json
from pathlib import Path
import numpy as np
from matched.runner import numeric_npy, sha
from external_eval.author_inference import AuthorWindNet, EPOCHS

# Set before external data access. A failed check blocks external evaluation.
MAX_ABSOLUTE_ERROR_KMS = 0.05
MAX_RMS_ERROR_KMS = 0.01

def main():
    root=Path('data/external_models/pre2019')
    members=json.loads((root/'members.json').read_text(encoding='utf8'))
    rows=[]
    for fold in range(1,6):
        group=sorted([m for m in members if m['fold']==fold],key=lambda m:m['file'])
        first=group[0]
        with np.load(root/first['file']) as data:
            assert np.all(data['output'][:,0]<2019)
            model=AuthorWindNet(fold)
            try:
                emb=model.encode(data['images'])
                p=model.predict_embeddings(np.stack([emb[i:i+4] for i in range(14)]))
            finally:model.close()
            raw=Path(f'data/reference/fold{fold}_WindNet_Predict_Test_Iter_no{EPOCHS[fold-1]}.npy').read_bytes()
            ref=np.asarray(numeric_npy(raw)['data'],dtype=np.float64)[:14]
            aligned=bool(np.allclose(ref[:,1],data['output'][6:20,11],atol=.001,rtol=0))
            error=p-ref[:,0]
            row={'fold':fold,'member':first['member'],'n':len(p),'reference_sha256':sha(raw),'labels_aligned':aligned,
                 'max_abs_error_kms':float(np.max(np.abs(error))),'rms_error_kms':float(np.sqrt(np.mean(error**2)))}
            row['passed']=aligned and row['max_abs_error_kms']<=MAX_ABSOLUTE_ERROR_KMS and row['rms_error_kms']<=MAX_RMS_ERROR_KMS
            rows.append(row);print(json.dumps(row),flush=True)
    result={'external_data_accessed':False,'type':'technical implementation equivalence, not scientific model selection',
            'sampling':'first lexical block from every historical fold, fixed before inference',
            'max_absolute_error_tolerance_kms':MAX_ABSOLUTE_ERROR_KMS,'max_rms_error_tolerance_kms':MAX_RMS_ERROR_KMS,
            'folds':rows,'passed':all(r['passed'] for r in rows)}
    Path('external_eval/author_restoration_preflight.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    assert result['passed'],'Original restoration failed; external access remains prohibited'

if __name__=='__main__':main()
