# Really the only thing we would ever want to change in this script
# are the cfg and the tag.

function usage
    echo "Runs a training script from scratch."
    echo
    echo "Arguments:"
    echo "  -h/--help  print this message"
    echo
    echo "  --batch-size  GPU batch size (default 256)"
    echo "  --config      which config file to use (required)"
    echo "  --data        path to dataset (required)"
    echo "  --debug       run with only one process rather than 8 so pdb is useable."
    echo "  --port        the master processes port (default 12345)"
    echo "  --nprocs      number of processes (default 8)"
end

# Parse arguments
set -l options (fish_opt --short h --long help)
set -a options (fish_opt --short b --long batch-size --required-val --long-only)
set -a options (fish_opt --short c --long config --required-val --long-only)
# --short g for debuG because --data uses --short d
set -a options (fish_opt --short g --long debug --long-only)
set -a options (fish_opt --short d --long data --required-val --long-only)
set -a options (fish_opt --short p --long port --required-val --long-only)
set -a options (fish_opt --short n --long nprocs --required-val --long-only)

argparse --ignore-unknown $options -- $argv


# Print help
if not test -z $_flag_help
    usage
    exit 0
end


# Check environment variables
if not string length -q -- $VENV
    echo "You need to provide a VENV environment variable."
    echo "Try:"
    echo "  source scripts/env.fish"
    exit 2
end

if not string length -q -- $RUN_OUTPUT
    echo "You need to provide a RUN_OUTPUT environment variable."
    echo "Try:"
    echo "  source scripts/env.fish"
    exit 2
end

set launcher $VENV/bin/torchrun

# --batch size
set batch_size 256
if not test -z $_flag_batch_size
    set batch_size $_flag_batch_size
end

# --config
if test -z $_flag_config
    echo "You must provide a --config argument!"
    exit 1
end

# --config
if test -z $_flag_data
    echo "You must provide a --data argument!"
    exit 1
end

# --port
set port 12345
if not test -z $_flag_port
    set port $_flag_port
end

# --nprocs
set nprocs 8
if not test -z $_flag_nprocs
    set nprocs $_flag_nprocs
end

# --debug 
# Has to come after nprocs because it modifies nprocs
if not test -z $_flag_debug
    set nprocs 1
end

if set -q $argv[1]
    # There is at least one remaining variable.
    # Confirm that this is really what you want to do
    echo "Do you want to pass the additional arguments"
    echo
    echo "  $argv"
    echo
    echo "to main.py?"
    read --local --prompt-str "[y/N] " confirm
    switch $confirm
        case Y y
            echo "Okay!"
        case '*'
            echo "Exiting..."
            exit 0
    end
end

OMP_NUM_THREADS=32 $launcher --nproc_per_node $nprocs \
    --master_port $port \
    main.py \
    --cfg $_flag_config \
    --data-path $_flag_data \
    --output $RUN_OUTPUT \
    --batch-size $batch_size \
    --fused_window_process \
    --fused_layernorm \
    $argv  # include additional arguments
