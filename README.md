# Image Classification with a CNN

A TensorFlow/Keras study of ten-class, 28 × 28 grayscale image classification, from IDX data loading to convolutional training and evaluation.

[Explore the notebook](ImageClassificationWithCNN.ipynb) · [Read the report](Report-ImageClassificationWithCNN%20-%20Google%20Docs.pdf)

## Inside the experiment

The notebook builds a compact CNN with two convolutional stages (32 and 64 filters), batch normalization, max pooling, a 128-unit dense layer, and a ten-class softmax output. Inputs are normalized to `[0, 1]`, and 20% of the training data is reserved for validation with a fixed split seed.

Training is configured for 100 epochs with batches of 128, categorical cross-entropy, and momentum SGD. The notebook also plots training and validation accuracy and loss, evaluates a held-out test set, and saves the model as `fashion_mnist_model.keras`.

## Recorded result

![Saved training and validation accuracy and loss curves](assets/training-curves.png)

*Original saved training plot from the committed notebook; not a new run.*

| Measure | Saved notebook output |
| --- | ---: |
| Test accuracy | 90.84% |
| Test loss | 0.6774 |

These are historical outputs stored in the notebook, not a freshly reproduced benchmark. The saved training log stops partway through epoch 5 even though the code requests 100 epochs; it does not establish the exact training history of the evaluated model.

## Open and run

Use Python 3.11 and an isolated environment:

```sh
python -m pip install -r requirements.txt
jupyter lab ImageClassificationWithCNN.ipynb
```

Run all cells from a fresh kernel. The notebook explicitly loads **Fashion-MNIST** using Keras; the first run downloads the dataset. Training is configured by `EPOCHS = 100`, which you can lower for an exploratory run. All random generators are seeded through Keras and deterministic TensorFlow operations are enabled. Reproducibility still depends on the software/hardware environment.

For existing local IDX data, `data_utils.load_mnist(path, kind='train')` and `kind='t10k'` validate headers, dimensions, counts and class ranges. IDX format alone cannot establish dataset identity; use Fashion-MNIST files for this experiment. Preprocessing supports variable sample counts while requiring 28 × 28 grayscale images.

## Maintenance validation

The model now uses the supported `SGD` optimizer and an explicit input layer. No augmentation is applied, matching the original executed training path; the unused augmentation generator was removed. Corrected notebook cells have cleared outputs so old metrics cannot be mistaken for a new run.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest --rootdir=. tests
```

The validation checks corrupted IDX headers/counts, small valid datasets, preprocessing, and a real model build plus one training batch on synthetic images. The model smoke test uses TensorFlow 2.16.2/Keras 3.15.1 on Python 3.11. It does not reproduce the historical 90.84% accuracy, download the full dataset, or establish generalization performance. The saved figure and report retain the earlier results with that distinction explicit.
