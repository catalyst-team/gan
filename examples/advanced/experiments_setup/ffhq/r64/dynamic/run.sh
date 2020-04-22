#!/usr/bin/env bash
# grid search dynamic params

# logdir with ALL of the experiments
MAINLOGDIR=logs/experiments
# experiment id (relative path in main logdir)
EXPERIMENT_ID=ffhq/r64/dcgan/dynamic_gan/v007_enforcing_static

LOGDIR="$MAINLOGDIR/$EXPERIMENT_ID"
OUTDIR="examples/advanced/rendered/$EXPERIMENT_ID"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python catalyst_gan/rendering.py -t \
    examples/advanced/tconfigs/config_base.yml \
    examples/advanced/tconfigs/eval/inception/fid.yml \
    examples/advanced/tconfigs/data/FFHQ.yml \
    "$DIR/tloss_nowrappers.yml" \
    examples/advanced/tconfigs/model/r64/dcgan.yml \
    examples/advanced/tconfigs/optim/tstatic_choise.yml \
    "$DIR/dynamic_cfg.yml" \
    -n base eval data loss model optim optim_dyn \
    -p "$DIR/params.yml" \
    --out_dir $OUTDIR \
    --exp_dir $LOGDIR

#./$OUTDIR/run_command_check.txt
./$OUTDIR/run_command.txt
