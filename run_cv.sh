set -ue

model_label=xlsr53 # xlsr53 or xlsr128
lang=el # languages
w2v_path=${PATH_TO_THE_PRE_TRAINED_MODEL.pt}
save_dir=${PATH_TO_SAVE_MODELS.pt}

fairseq-hydra-train \
    hydra.run.dir=run_babel/$model_label/student/$lang \
    task.data=$PWD/data/$lang \
    common.tensorboard_logdir=tblogs \
    model.w2v_path=$w2v_path \
    checkpoint.save_dir=$save_dir \
    --config-dir $PWD/configs \
    --config-name common_voice
