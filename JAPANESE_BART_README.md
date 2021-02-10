# Japanese BART Pretrained Model
We pretrained [BART](https://arxiv.org/pdf/1910.13461.pdf) models with Japanese Wikipedia's text.

## Download 
```shell
# BART base v1.1 (1.2G)
wget http://lotus.kuee.kyoto-u.ac.jp/nl-resource/JapaneseBARTPretrainedModel/japanese_bart_base_1.1.tar.gz
# BART large v1.0 (3.6G)
wget http://lotus.kuee.kyoto-u.ac.jp/nl-resource/JapaneseBARTPretrainedModel/japanese_bart_large_1.0.tar.gz
```
They include a pretrained bart model (`bart_model.pt`), a sentencepiece model (`sp.model`) and a dictionary (`dict.txt`).  

## Requirements
Python >= 3.6  
[Juman++](https://github.com/ku-nlp/jumanpp) == 2.0.0-rc3  
fairseq (this branch of this forked repo)  
[zenhan](https://pypi.org/project/zenhan/0.5/)  
[pyknp](https://github.com/ku-nlp/pyknp)  
[sentencepiece](https://github.com/google/sentencepiece/tree/master/python)  
[tensorboardX](https://github.com/lanpa/tensorboardX) (optional)  
  

## Installation
You can use pipenv command for installing requirements.  
```shell
pipenv install
```

## Preprocess
We applied Juman++ to datasets for word segmentation first, and then applied SentencePiece to them for subword segmentation.  
Besides, before applying Juman++, we converted half-width characters to full-width characters.  
  
Prepare datasets (`$TRAIN_SRC`, ...), which contain one sentence per one line.  
```shell
cat $TRAIN_SRC | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/train.src-tgt.src
cat $TRAIN_TGT | python3 ./jaBART_preprocess.py  --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/train.src-tgt.tgt
cat $VALID_SRC | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/valid.src-tgt.src
cat $VALID_TGT | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/valid.src-tgt.tgt
cat $TEST_SRC | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/test.src-tgt.src
cat $TEST_TGT | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/test.src-tgt.tgt
cp $DICT $DATASET_DIR/dict.src.txt
cp $DICT $DATASET_DIR/dict.tgt.txt
```

## Finetune
Set `$ARCH` for `bart_base` or `bart_large`.  
```bash
DATA_PATH=/user/fairseq/LP-T-sm-bin/
BART_PATH=/user/models/bart/ja/base/model.pt
SAVE_DIR=/user/fairseq/LP-T-sm-ft/
WANDB_PJ=bart-ja-nlg
ARCH=bart_base
CUDA_VISIBLE_DEVICES=0,1,2,3
MAX_EPOCH=2                        # number of epochs
MAX_SENTENCES=32                   # Batch size
NUM_CLASSES=1                      # Regression task
LR=3e-05                           # Peak LR for polynomial LR scheduler
TOTAL_NUM_UPDATES=40000
WARMUP_UPDATES=2500

fairseq-train $DATA_PATH \
    --wandb-project $WANDB_PJ \
    --restore-file $BART_PATH \
    --save-dir $SAVE_DIR \
    --max-tokens 1024 --update-freq 2 \
    --task translation_from_pretrained_bart \
    --source-lang src --target-lang tgt \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.2 \
    --dataset-impl raw \
    --optimizer adam \
    --adam-eps 1e-06 --adam-betas '{0.9, 0.98}' --lr-scheduler polynomial_decay --lr $LR \
    --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters --reset-lr-scheduler \
    --arch $ARCH \
    --max-epoch $MAX_EPOCH \
    --ddp-backend no_c10d --max-update 80000 \
    --encoder-normalize-before --decoder-normalize-before --prepend-bos \
```

## Text Generation
```bash
DATA_PATH=/user/fairseq/LP-T-sm-bin/
BART_PATH=/user/models/bart/ja/base/model.pt
SAVE_DIR=/user/fairseq/LP-T-sm-ft/
GENERATED_PATH=/user/fairseq/LP-T-sm-bin/generated.txt

fairseq-generate $DATA_PATH \
    --path $SAVE_DIR/checkpoint_best.pt \
    --task translation_from_pretrained_bart \
    --dataset-impl raw \
    --gen-subset test \
    -s src -t tgt \
    --max-sentences 64 \
    --prepend-bos > $GENERATED_PATH

cat $GENERATED_PATH | grep -P "^H" | cut -f 3- | sed 's/<<unk>>/<unk>/g' | sed 's/▁//g' > ${GENERATED_PATH}.pred
cat $GENERATED_PATH | grep -P "^S" | cut -f 2- | sed 's/<<unk>>/<unk>/g' | sed 's/▁//g' > ${GENERATED_PATH}.src
cat $GENERATED_PATH | grep -P "^T" | cut -f 2- | sed 's/<<unk>>/<unk>/g' | sed 's/▁//g' > ${GENERATED_PATH}.tgt
```

## Pretrain
We set most of the parameters according to the papers of [BART](https://arxiv.org/pdf/1910.13461.pdf), [mBART](https://arxiv.org/abs/2001.08210) and [RoBERTa](https://arxiv.org/abs/1907.11692).  
  
The following command is an example of pretraining bart by yourself.  
`$DATASET_DIR` must contain `train`, `valid` and `dict.txt`.  

```shell
fairseq-train $DATASET_DIR --arch bart_base --task denoising --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --mask-length span-poisson --mask 0.35 --poisson-lambda 3.5 --permute-sentences 1.0 --replace-length 1 \
    --tokens-per-sample 512 --max-sentences $MAX_SENTENCES  --rotate 0.0 \
    --max-update 500000 --tensorboard-logdir $TENSORBOARD_DIR --update-freq $UPDATE_FREQ \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 1 --save-dir $SAVEPT \
    --fp16 --num-workers 0 --fp16-init-scale 4 
```

The following command is an example of pretraining multilingual (ja and en) bart by yourself.  
`$DATASET_DIR` must contain `dict.txt`.  
`$DATASET_DIR/{ja,en}` must contain `train` and `valid`.  
```shell
fairseq-train $DATASET_DIR --arch mbart_base --task multilingual_denoising --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --mask-length span-poisson --mask 0.35 --poisson-lambda 3.5 --permute-sentences 1.0 --replace-length 1 \
    --tokens-per-sample 512 --max-sentences $MAX_SENTENCES  --rotate 0.0 \
    --max-update 500000 --tensorboard-logdir $TENSORBOARD_DIR --update-freq $UPDATE_FREQ \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 1 --save-dir $SAVEPT \
    --fp16 --num-workers 0 --fp16-init-scale 4 \
    --langs ja,en --add-lang-token
```

