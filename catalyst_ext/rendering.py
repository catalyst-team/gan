import json
import os
import shutil
import itertools
import yaml
from collections import defaultdict
import sys
from argparse import ArgumentParser
from pathlib import Path
from pathlib import PurePosixPath as _unix

from catalyst.utils import load_config
from jinja2 import Environment, FileSystemLoader


parser = ArgumentParser()
parser.add_argument(
    '--in_template', '-t', type=Path, nargs='+', required=True,
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
parser.add_argument(
    '--out_names', '-n', type=str, nargs='*', required=False,
    help="names of rendered templates to save. if specified must have "
         "same number of arguments as --in_template"
)
parser.add_argument(
    '--exp_dir', '-l', type=Path, default='./logs/tmp_experiment',
    help="logdir for running all of the rendered experiments"
         "(to specify in runs.txt)"
)


def save_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.dump(obj, f)


def copy_multiple(src_list, dst_dir, dst_fname_list):
    for src, dst_fname in zip(src_list, dst_fname_list):
        shutil.copy(src, os.path.join(dst_dir, dst_fname))


def save_templates(in_templates, out_dir, out_names):
    assert len(in_templates) == len(out_names)
    assert len(in_templates) > 0
    if len(in_templates) == 1:
        shutil.copy(in_templates[0],
                    os.path.join(out_dir, "_template.yml"))
    else:
        out_dir = os.path.join(out_dir, "_templates")
        os.makedirs(out_dir, exist_ok=True)
        copy_multiple(in_templates, out_dir, out_names)


def get_out_names(in_templates, out_names=None):
    if out_names is None:
        out_names = [f"t{i:02d}.yml" for i, _ in enumerate(in_templates)]
    else:
        for i in range(len(out_names)):
            if not out_names[i].endswith('.yml'):
                out_names[i] += '.yml'
    return out_names


def read_params(in_params):
    # read param grid
    params_grid = load_config(in_params, ordered=True)
    params_kv = dict()  # constants
    grid_kv = dict()  # grid search
    num_rendered_configs = 1
    for k, v in params_grid.items():
        if isinstance(v, list):
            if len(v) > 1:
                grid_kv[k] = v
                num_rendered_configs *= len(v)
            elif len(v) == 1:
                params_kv[k] = v[0]
            else:
                raise NotImplementedError("What to do with [] - empty list?")
        elif isinstance(v, dict):
            assert v.pop("_value", False)
            assert "value" in v
            v = v["value"]
            params_kv[k] = v
        else:
            params_kv[k] = v
    return params_kv, grid_kv, num_rendered_configs


def create_jinja_template(in_template):
    # processing template
    env = Environment(
        loader=FileSystemLoader(str(in_template.absolute().parent)),
        trim_blocks=True,
        lstrip_blocks=True
    )
    # env.globals.update(zip=zip)  # enable zip command inside jinja2 template
    template = env.get_template(in_template.name)
    return template


def iterate_grid(grid_kv):
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
        dirname = f"config_{prefix}"
        yield dirname, curr_kv


def main(in_templates, in_params, out_dir, out_names, exp_dir):
    assert all(os.path.exists(t) for t in in_templates)
    assert os.path.exists(in_params)
    # assert not os.path.exists(out_dir)
    out_names = get_out_names(in_templates, out_names)
    assert len(out_names) == len(in_templates)

    os.makedirs(out_dir, exist_ok=True)
    # shutil.copy(in_template, os.path.join(out_dir, "_template.yml"))
    save_templates(in_templates, out_dir, out_names)
    shutil.copy(in_params, os.path.join(out_dir, "_params_raw.yml"))

    # read param grid
    params_kv, grid_kv, num_rendered_configs = read_params(in_params)
    if params_kv:
        save_yaml(params_kv, os.path.join(out_dir, "_params_const.yml"))
    if grid_kv:
        save_yaml(grid_kv, os.path.join(out_dir, "_params_grid.yml"))

    run_commands = defaultdict(lambda: f"catalyst-dl run -C")
    for in_template, out_name in zip(in_templates, out_names):
        template = create_jinja_template(in_template)
        if num_rendered_configs == 1:
            out_config = out_dir / out_name
            out_config.write_text(template.render(**params_kv))
            run_commands[exp_dir] += f" {_unix(out_config)}"
        else:
            for config_dir, exp_grid_params in iterate_grid(grid_kv):
                curr_out_dir = out_dir / config_dir
                os.makedirs(curr_out_dir, exist_ok=True)
                save_yaml(exp_grid_params, curr_out_dir / "_params_grid.yml")
                out_config = curr_out_dir / out_name
                out_config.write_text(
                    template.render(**params_kv, **exp_grid_params)
                )
                run_commands[exp_dir / config_dir] += f" {_unix(out_config)}"

    with open(out_dir / "runs.txt", "w") as runs_file:
        for logdir, bash_cmd in run_commands.items():
            bash_cmd = f"{bash_cmd} --logdir {_unix(logdir)}{os.linesep}"
            runs_file.write(bash_cmd)


def _main():
    args = parser.parse_args()
    main(
        in_templates=args.in_template,
        in_params=args.in_params,
        out_dir=args.out_dir,
        out_names=args.out_names,
        exp_dir=args.exp_dir
    )


if __name__ == '__main__':
    _main()
