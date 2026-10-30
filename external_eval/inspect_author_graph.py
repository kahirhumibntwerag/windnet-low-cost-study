"""Inspect public author checkpoint graph metadata; performs no forecasts."""
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2


def main():
    rows=[]
    for fold, epoch in enumerate([60, 63, 108, 290, 99], 1):
        path=Path('data/external_models')/f'author_fold{fold}'/f'model_no-{epoch}.meta'
        meta=meta_graph_pb2.MetaGraphDef();meta.ParseFromString(path.read_bytes())
        placeholders=[{'name':n.name, 'shape':[int(x.size) for x in n.attr['shape'].shape.dim]} for n in meta.graph_def.node if n.op=='Placeholder']
        elu=[n.name for n in meta.graph_def.node if n.op=='Elu']
        concat=[n.name for n in meta.graph_def.node if n.op in ['Concat','ConcatV2']]
        constant_shapes=[{'name':n.name, 'shape':[int(x.size) for x in n.attr['value'].tensor.tensor_shape.dim]} for n in meta.graph_def.node if n.op=='Const' and n.attr['value'].tensor.tensor_shape.dim and n.attr['value'].tensor.tensor_shape.dim[-1].size in [64,256,512,832]]
        record={'fold':fold,'epoch':epoch,'nodes':len(meta.graph_def.node),'placeholders':placeholders,'elu':elu,'last_concat':concat[-8:],'normalization_candidate_shapes':constant_shapes[-20:]}
        rows.append(record);print(json.dumps(record))
    Path('external_eval/author_graph_inventory.json').write_text(json.dumps({'tensorflow_inspection_runtime':tf.__version__,'forecasts_executed':False,'folds':rows},indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
