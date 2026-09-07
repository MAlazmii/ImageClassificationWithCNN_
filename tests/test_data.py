import gzip
import importlib.util
import json
import ast
import struct
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]

def loader():
    path=ROOT/'data_utils.py'
    if path.exists():
        spec=importlib.util.spec_from_file_location('data_utils',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    cell=json.loads((ROOT/'ImageClassificationWithCNN.ipynb').read_text())['cells'][1]
    tree=ast.parse(''.join(cell['source']))
    namespace={'gzip':gzip,'np':np}
    exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)],type_ignores=[]),'original','exec'),namespace)
    return type('Original',(),{'load_mnist':staticmethod(namespace['load_mnist'])})

def write_idx(tmp_path,magic=2051,count=2):
    with gzip.open(tmp_path/'train-images-idx3-ubyte.gz','wb') as f:f.write(struct.pack('>IIII',magic,count,28,28)+bytes([255])*1568)
    with gzip.open(tmp_path/'train-labels-idx1-ubyte.gz','wb') as f:f.write(struct.pack('>II',2049,2)+bytes([0,1]))

def test_invalid_idx_magic_is_rejected(tmp_path):
    write_idx(tmp_path,magic=1)
    with pytest.raises(ValueError):loader().load_mnist(tmp_path)

def test_count_mismatch_is_rejected(tmp_path):
    write_idx(tmp_path,count=9)
    with pytest.raises(ValueError):loader().load_mnist(tmp_path)

def test_small_valid_idx_does_not_assume_full_dataset(tmp_path):
    write_idx(tmp_path)
    images,labels=loader().load_mnist(tmp_path)
    assert images.shape==(2,784)
    np.testing.assert_array_equal(labels,[0,1])
    assert hasattr(loader(),'prepare_images'),'Independent validated image preprocessing is missing'
    normalized=loader().prepare_images(images)
    assert normalized.shape==(2,28,28,1)
    assert normalized.dtype==np.float32
    assert normalized.min()==normalized.max()==1
