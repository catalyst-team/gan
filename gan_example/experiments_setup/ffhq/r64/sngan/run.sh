#!/usr/bin/env bash

# logdir with ALL of the experiments
MAINLOGDIR=logs/experiments
# experiment id (relative path in main logdir)
EXPERIMENT_ID=ffhq/r64/sngan_grid_search

LOGDIR="$MAINLOGDIR/$EXPERIMENT_ID"
OUTDIR="gan_example/rendered/$EXPERIMENT_ID"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python catalyst_ext/rendering.py -t \
    gan_example/tconfigs/config_base.yml \
    gan_example/tconfigs/eval/inception/fid.yml \
    gan_example/tconfigs/data/FFHQ.yml \
    gan_example/tconfigs/tloss.yml \
    gan_example/tconfigs/model/r64/dcgan.yml \
    gan_example/tconfigs/optim/tstatic_choise.yml \
    -n base eval data loss model optim \
    -p "$DIR/params.yml" \
    --out_dir $OUTDIR \
    --exp_dir $LOGDIR

# check similar to dcgan
./$DIR/gan/run.sh
./$DIR/wgan/run.sh
./$DIR/wgan_gp/run.sh

#./$OUTDIR/run_command_check.txt
./$OUTDIR/run_command.txt