#! /usr/bin/env bash
set -ue
stage=1
fairseq=$HOME/fairseq # path of the fairseq toolkit to locate the decode python script.

lang=eu
model_path=${PATH_TO_THE_MODEL.pt}
lexicon="data/$lang/lexicon.txt"
lm=${PATH_TO_THE_LANGUAGE_MODEL.bin}
decode_dir=${PATH_TO_THE_DECODE_RESULT}

if [ $stage -le 1 ]; then    
    python $fairseq/examples/speech_recognition/infer.py $PWD/data/${lang} \
        --task audio_finetuning \
        --nbest 1 \
        --path $model_path \
        --gen-subset valid \
        --results-path $decode_dir \
        --w2l-decoder kenlm \
        --lm-model $lm \
        --lm-weight 4 \
        --lexicon $lexicon \
        --word-score -1 \
        --sil-weight 0 \
        --criterion ctc \
        --labels wrd \
        --max-tokens 1280000 \
        --post-process none \
        --beam 50
fi

if [ $stage -le 2 ]; then
    model_name=$(basename $model_path)

    hyp=$decode_dir/hypo.word-$model_name-valid.txt
    ref=$decode_dir/ref.word-modified.txt

    python tools/modify_ref.py $hyp $PWD/data/${lang}/valid.wrd $ref
    
    sclite -h $hyp -r $ref -i wsj \
        -o sum pralign -O $decode_dir
fi