"""Inference in the authors' saved TensorFlow graphs; no training operations run."""
import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from matched.runner import sha

tf.compat.v1.disable_eager_execution()

EPOCHS = [60, 63, 108, 290, 99]

class AuthorWindNet:
    def __init__(self, fold, model_root=Path('data/external_models')):
        self.fold = fold
        self.normalization = json.loads((model_root/'author_normalization.json').read_text(encoding='utf8'))['folds'][str(fold)]
        prefix = model_root/f'author_fold{fold}'/f'model_no-{EPOCHS[fold-1]}'
        self.graph = tf.Graph()
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(str(prefix)+'.meta', clear_devices=True)
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=2)
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            saver.restore(self.session, str(prefix))
        self.image = self.graph.get_tensor_by_name('Placeholder:0')
        self.history = self.graph.get_tensor_by_name('Placeholder_6:0')
        self.keep = self.graph.get_tensor_by_name('Placeholder_4:0')
        self.embedding = self.graph.get_tensor_by_name('concat:0')
        self.output = self.graph.get_tensor_by_name('Elu_3:0')
        assert self.embedding.shape.as_list() == [None,832]

    def encode(self, images, batch_size=16):
        values=[]
        for start in range(0,len(images),batch_size):
            resized=np.stack([cv2.resize(np.asarray(im,dtype=np.float64),(224,224)) for im in images[start:start+batch_size]])
            x=np.repeat(resized[:,:,:,None],3,axis=-1)-self.normalization['image_mean']
            values.append(self.session.run(self.embedding,{self.image:x,self.keep:1.0}))
        return np.concatenate(values)

    def predict_embeddings(self, histories, batch_size=256):
        assert histories.ndim==3 and histories.shape[1:]==(4,832)
        out=[]
        for start in range(0,len(histories),batch_size):
            raw=self.session.run(self.output,{self.history:histories[start:start+batch_size],self.keep:1.0})
            out.append(raw[:,0]*self.normalization['target_range']+self.normalization['target_min'])
        return np.concatenate(out).astype(np.float64)

    def close(self):
        self.session.close()

def recover_normalization():
    """Author BifurcatedDataloader/Imresize and ModelRestoration_cv.py semantics."""
    root=Path('data/external_models');folder=root/'pre2019'
    members=json.loads((folder/'members.json').read_text(encoding='utf8'))
    summaries=[]
    for m in members:
        with np.load(folder/m['file']) as d:
            assert np.all(d['output'][:,0]<2019)
            images=np.stack([cv2.resize(x.astype(np.float64),(224,224)) for x in d['images']])
            summaries.append({**m,'image_sum':float(images.sum(dtype=np.float64)), 'pixels':int(images.size), 'ymin':float(d['output'][:,11].min()), 'ymax':float(d['output'][:,11].max())})
    rows={}
    for held in range(1,6):
        fit=[r for r in summaries if r['fold']!=held]
        rows[str(held)]={'image_mean':sum(r['image_sum'] for r in fit)/sum(r['pixels'] for r in fit),
                         'target_min':min(r['ymin'] for r in fit),
                         'target_range':max(r['ymax'] for r in fit)-min(r['ymin'] for r in fit),
                         'fitting_blocks':len(fit),'fitting_days_including_all20':len(fit)*20,
                         'training_source_hashes':{r['member']:r['source_sha256'] for r in fit}}
    result={'source':'author pre2019 fitting folds only; all20 days per original normalization code', 'external_data_used':False,'folds':rows}
    path=root/'author_normalization.json'
    assert not path.exists(),'Refuse to replace normalization'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    Path('external_eval/author_normalization.json').write_bytes(path.read_bytes())
    print(json.dumps({'normalization_recovered':True,'sha256':sha(path.read_bytes())}),flush=True)

if __name__=='__main__':recover_normalization()
