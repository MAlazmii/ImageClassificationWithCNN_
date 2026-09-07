import ast
import json
from pathlib import Path
import numpy as np
import pytest

def test_notebook_model_compiles_and_trains_one_batch():
    tf=pytest.importorskip('tensorflow')
    tf.keras.utils.set_random_seed(42)
    notebook=json.loads((Path(__file__).resolve().parents[1]/'ImageClassificationWithCNN.ipynb').read_text())
    source=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'model = models.Sequential' in ''.join(c['source']))
    tree=ast.parse(source)
    tree.body=[node for node in tree.body if not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='history' for t in node.targets))]
    scope={'tf':tf,'layers':tf.keras.layers,'models':tf.keras.models}
    exec(compile(tree,'notebook-model','exec'),scope)
    model=scope['model']
    images=np.zeros((4,28,28,1),dtype=np.float32)
    labels=tf.keras.utils.to_categorical([0,1,2,3],10)
    result=model.train_on_batch(images,labels)
    assert np.isfinite(result).all()
    assert model.output_shape==(None,10)
