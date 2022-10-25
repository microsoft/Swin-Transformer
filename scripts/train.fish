# Really the only thing we would ever want to change in this script
# are the cfg and the tag.

function usage
    echo "Runs a training script from scratch."
    echo
    echo "Arguments:"
    echo "  -h/--help  print this message"
    echo "  --config   which config file to use (should be YAML)"
    echo "  --debug    run with only one process rather than 8 so pdb is useable."
    echo "  --tag      tag for the run (I use v0, v1, etc)"
    echo "  --venv     path to the virtual environment (default ./venv/)"
end

set -l options (fish_opt --short h --long help)
set -a options (fish_opt --short c --long config --required-val --long-only)
set -a options (fish_opt --short t --long tag --required-val --long-only)
set -a options (fish_opt --short v --long venv --long-only)
set -a options (fish_opt --short d --long debug --long-only)

argparse $options -- $argv

if not test -z $_flag_help
    usage
    exit 0
end

if test -z $_flag_config
    echo "You must provide a --config argument!"
    exit 1
end

if test -z $_flag_tag
    echo "You must provide a --tag argument!"
    exit 1
end

set venv ./venv
if not test -z $_flag_venv
    set venv $_flag_venv
end

set launcher $venv/bin/torchrun
set launcher_args --nproc_per_node 8 --master_port 12345

if not test -z $_flag_debug
    set launcher_args --nproc_per_node 1 --master_port 12345
end

$launcher $launcher_args \
    main.py \
    --cfg $_flag_config \
    --output runs \
    --tag $_flag_tag \
    --fused_window_process \
    --fused_layernorm
