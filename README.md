Compositional Structural Embeddings of Terms with Graph Convolutional Networks
====

## At a glance

  - [xgcn/models.py](xgcn/models.py) describes the the `XGCNModel` that defines a bilingual GCN [model](https://fairseq.readthedocs.io/en/latest/models.html) designed for cross-lingual mapping of terminologies.
  - [xgcn/criterion.py](xgcn/criterion.py) defines the `DictionaryCriterion` associated with training the model according to the terminology word-translation task.

For other elements of the training pipeline, the model interfaces heavily with the PyTorch version [`fairseq`](https://fairseq.readthedocs.io/en/latest/) [(docs)](https://fairseq.readthedocs.io/en/latest/), relying on the package to provide training task definition, dataset handling and a optionally multi-GPU trainer. Specifically

  - A [`TranslationTask`](https://fairseq.readthedocs.io/en/latest/tasks.html#translation) is created to handle the word translation data, in supervised or unsupervised variants.
  - The [`Trainer`](`https://github.com/pytorch/fairseq/blob/v0.6.2/fairseq/trainer.py`) handles the training process by applying the optimizer according to the criterion to the model.

## Training

### Requirements

  - `Python 3.7`
  - `PyTorch 1.3`
  - `fairseq 0.9.0`

### Download & Process data

Download the MeSH XML file.

```bash
# Download data
curl -# -o data/desc2020.xml ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/xmlmesh/desc2020.xml
```

Then preprocess the downloaded XML into language pair text formats.

```bash
# API key obtained under license from the UMLS Terminology Services at https://uts.nlm.nih.gov/license.html
python tools/get_mesh.py --mesh data/desc2019.xml --apikey xxxxxx --target-langs FRE
```

Finally split the processed files into train/test sets.

```bash
python tools/generate_mesh.py -s ENG -t FRE --file-pre data/desc2019.xml --dest-dir data/
```

### Preprocessing

Construct the dictionaries with respect to the dataset, and tokenize the raw text as word pieces if training with BERT features.

```bash
# For the ENG-FRE language pair
python preprocess.py --source-langs ENG --target-langs FRE --train-pre data/train --test-pre data/test --align-suffix edge --dest-dir data/
```

### Training

The training script builds a [translation task](https://fairseq.readthedocs.io/en/latest/tasks.html#translation) given the training language pairs, trains the GCN model on the training data according to the criterion.

```bash
# Training for the ENG-FRE language pair
python train.py --task translation --source-lang ENG --target-lang FRE --train-subset train --valid-subset test --dataset-impl cached --load-alignments --arch xgcn --optimizer adam --lr 1 --sentence-avg --num-workers=0 --distributed-world-size 1 --criterion dictionary --save-dir checkpoints/ --skip-invalid-size-inputs-valid-test --log-format tqdm data/
```

### Evaluation

```bash
# List multiple checkpoint files separated with colon for ensemble models
python generate.py --task translation --source-lang ENG --target-lang FRE --gen-subset test --dataset-impl cached --path checkpoints/ENG-FRE/checkpoint_best.pt --load-alignments --k 1 10 --num-workers 0 --log-format tqdm -o output/xgcn.ENG-FRE.out data
```

### Licence

This repo contains software with multiple licenses. Refer to the heading of each file for the copyright information and find the [LICENSE](LICENSE).

To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed without any warranty. You should have received a copy of the [CC0 Public Domain Dedication](LICENSE) along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
