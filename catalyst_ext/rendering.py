import json
import os
import shutil
import itertools
import sys
from argparse import ArgumentParser
from pathlib import Path

from catalyst.utils import load_config
from jinja2 import Environment, FileSystemLoader


parser = ArgumentParser()
parser.add_argument(
    '--in_template', '-t', type=Path, required=True,
    help="path to class config"
)
parser.add_argument(
    '--in_params', '--in_grid', '-p', '-g', type=Path, required=True,
    help="path to save dataset copy (if omitted no copy will be saved)"
)
parser.add_argument(
    '--out_dir', '-o', type=Path, required=True,
    help="path to folder where to save all configs"
)


def main(in_template, in_params, out_dir):
    assert os.path.exists(in_template)
    assert os.path.exists(in_params)
    # assert not os.path.exists(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(in_template, os.path.join(out_dir, "_template.yml"))
    shutil.copy(in_params, os.path.join(out_dir, "_params.yml"))

    # processing template
    env = Environment(
        loader=FileSystemLoader(str(in_template.absolute().parent)),
        trim_blocks=True,
        lstrip_blocks=True
    )
    # env.globals.update(zip=zip)  # enable zip command inside jinja2 template
    template = env.get_template(in_template.name)

    # read param grid
    params_grid = load_config(in_params, ordered=True)
    params_kv = dict()
    grid_kv = dict()
    num_rendered_configs = 1
    for k, v in params_grid.items():
        if isinstance(v, list):
            if len(v) > 1:
                grid_kv[k] = v
                num_rendered_configs *= len(v)
            else:
                params_kv[k] = v
        elif isinstance(v, dict):
            assert v.pop("_value", False)
            assert "value" in v
            v = v["value"]
            params_kv[k] = v
        else:
            params_kv[k] = v

    if num_rendered_configs == 1:
        assert len(grid_kv) == 0

        out_config = out_dir / "config.yml"
        out_config.write_text(
            template.render(
                **params_kv
            )
        )
    else:
        # assertion can be safely removed
        assert num_rendered_configs < 1000, "be careful, too many configs"

        grid_keys = list(grid_kv.keys())
        for i, values_list in enumerate(itertools.product(*grid_kv.values())):
            assert len(grid_keys) == len(values_list)
            curr_kv = {
                grid_keys[j]: values_list[j] for j in range(len(values_list))
            }
            info = '-'.join([f"{k}_{v}" for k, v in curr_kv.items()])

            prefix = f"{i:03d}"
            if len(info) < 30:
                prefix = info
            out_config = out_dir / f"config_{prefix}.yml"
            out_config.write_text(
                template.render(
                    **params_kv,
                    **curr_kv
                )
            )


def _main():
    args = parser.parse_args()
    main(
        in_template=args.in_template,
        in_params=args.in_params,
        out_dir=args.out_dir
    )


if __name__ == '__main__':
    _main()
