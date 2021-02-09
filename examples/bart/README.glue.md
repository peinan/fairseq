# Fine-tuning BART on GLUE tasks

### 0) Download pre-trained BART model from [the official BART README](https://github.com/pytorch/fairseq/blob/6225dccb98/examples/bart/README.md) and extract it

```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar zxf bart.base.tar.gz
```

### 1) Download the data from GLUE website (https://gluebenchmark.com/tasks) using following commands:
```bash
wget https://gist.githubusercontent.com/peinan/2c01caaf8aa2ab7a4c3d6053cf516270/raw/1bd72f88b9c54f52ba705d136c8bd22d7d52eb61/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
```

### 2) Preprocess GLUE task data (same as RoBERTa):

```bash
./examples/roberta/preprocess_GLUE_tasks.sh data/glue STS-B
```
`glue_task_name` is one of the following:
`{ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA}`
Use `ALL` for preprocessing all the glue tasks.

#### 2.2) Replace the generated `dict.txt` with the one in the model

```bash
cp models/bart/en/base/dict.txt fairseq/STS-B-bin/input0/
cp models/bart/en/base/dict.txt fairseq/STS-B-bin/input1/
```

### 3) Fine-tuning on GLUE task:
Example fine-tuning cmd for `STS-B` task

```bash
DATA_PATH=/user/fairseq/STS-B-bin/
BART_PATH=/user/models/bart/en/base/model.pt
SAVE_DIR=/user/fairseq/STS-B-ft/
WANDB_PJ=glue                      # hangs after finish learning
TENSORBOARD_DIR=/user/tensorboard  # Not working in multi-GPU training (#2357)
ARCH=bart_base
CUDA_VISIBLE_DEVICES=0,1,2,3
MAX_SENTENCES=32                  # Batch size
NUM_CLASSES=1                     # Regression task
LR=2e-05                          # Peak LR for polynomial LR scheduler
TOTAL_NUM_UPDATES=1799
WARMUP_UPDATES=107

fairseq-train $DATA_PATH \
    --wandb-project $WANDB_PJ \
    --restore-file $BART_PATH \
    --num-classes $NUM_CLASSES \
    --regression-target --best-checkpoint-metric loss \
    --save-dir $SAVE_DIR \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --add-prev-output-tokens \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --arch $ARCH \
    --criterion sentence_prediction \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters
```

For each of the GLUE task, you will need to use following cmd-line arguments:

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-classes` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 5e-6 | 1e-5 | 1e-5 | 1e-5 | 5e-6 | 2e-5 | 2e-5 | 2e-5
`bsz` | 128 | 32 | 32 | 32 | 128 | 64 | 64 | 32
`--total-num-update` | 30968 | 33112 | 113272 | 1018 | 5233 | 1148 | 1334 | 1799
`--warmup-updates` | 1858 | 1986 | 6796 | 61 | 314 | 68 | 80 | 107

For `STS-B` additionally add `--regression-target --best-checkpoint-metric loss` and remove `--maximize-best-checkpoint-metric`.

**Note:**

a) `--total-num-updates` is used by `--polynomial_decay` scheduler and is calculated for `--max-epoch=10` and `--batch-size=32/64/128` depending on the task.

b) Above cmd-args and hyperparams are tested on Nvidia `V100` GPU with `32gb` of memory for each task. Depending on the GPU memory resources available to you, you can use increase `--update-freq` and reduce `--batch-size`.

### Inference on GLUE task
After training the model as mentioned in previous step, you can perform inference with checkpoints in `checkpoints/` directory using following python code snippet:

```python
from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
    $SAVE_DIR,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='STS-B-bin'
)
bart.cuda()
bart.eval()

import csv
import pandas as pd
from scipy.stats import pearsonr

df_valid = pd.read_csv('/user/data/glue/STS-B/dev.tsv', sep='\t',
    usecols=['sentence1', 'sentence2', 'score'], quoting=csv.QUOTE_NONE)

y_valid = df_valid['score'].values / 5
y_pred_bart = []
for idx, row in tqdm(df_valid.iterrows()):
    sent1 = row['sentence1']
    sent2 = row['sentence2']
    tokens = bart.encode(sent1, sent2)
    y_pred_bart.append(bart.predict('sentence_classification_head', tokens, return_logits=True).item())
y_pred_bart = np.array(y_pred_bart)
print('Peason Corr.:', pearsonr(y_valid, y_pred_bart)[0])

# Pearson Corr.: 0.8832975926229741
```
