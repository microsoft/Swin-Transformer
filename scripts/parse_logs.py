"""
This scripts parses the training logs to graph both training loss and validation accuracy over time.
"""

import argparse
import dataclasses
import re

import matplotlib.pyplot as plt
import preface


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Log file to parse. Typically named log_rank0.txt")
    parser.add_argument(
        "--last", help="How many of the latest epochs to look at.", default=10, type=int
    )
    parser.add_argument("--name", help="Run name")

    return parser.parse_args()


@dataclasses.dataclass
class ValidationLine:
    epoch: int
    batch: int
    batch_max: int
    loss: float
    mean_loss: float
    acc1: float
    mean_acc1: float
    acc5: float
    mean_acc5: float

    pattern0 = r"""
    ^\[.*?\]\                                                           
    INFO:\ Test:\ \                                                         
    \[\ ?(?P<batch>\d+)/(?P<batch_max>\d+)\]                               
    \ \                                                                   
    eta:\ \d+:\d\d:\d\d
    \ \                                                                    
    loss:\ (?P<loss>[\w.]+)\ \((?P<mean_loss>[\w.]+)\)                  
    \ \                                                                
    acc1:\ (?P<acc1>[\w.]+)\ \((?P<mean_acc1>[\w.]+)\)   
    \ \                                                                
    acc5:\ (?P<acc5>[\w.]+)\ \((?P<mean_acc5>[\w.]+)\)
    $
    """

    pattern1 = r"""
    ^\[.*?\]\                                                           
    \(main.py\ \d+\):\                                                  
    INFO\ Test:\                                                        
    \[(?P<batch>\d+)/(?P<batch_max>\d+)\]                               
    \t                                                                  
    Time\ \d+.\d+\ \(\d+.\d+\)
    \t                                                                   
    Loss\ (?P<loss>[\w.]+)\ \((?P<mean_loss>[\w.]+)\)                  
    \t                                                                 
    Acc@1\ (?P<acc1>[\w.]+)\ \((?P<mean_acc1>[\w.]+)\)   
    \t                                                                 
    Acc@5\ (?P<acc5>[\w.]+)\ \((?P<mean_acc5>[\w.]+)\)
    \t                                                                 
    Mem\ (?P<mem>.*)                                                   
    $
    """

    @classmethod
    def from_raw_line(cls, line, last_train):
        if "Test" not in line:
            return None

        # Example line:
        # [2022-06-08 07:34:50 swinv2_large_patch4_window7_224_inat21](main.py 258): INFO Test: [0/196]   Time 1.895 (1.895)      Loss 0.8066 (0.8066)    Acc@1 84.766 (84.766)   Acc@5 95.312 (95.312)      Mem 36916MB

        match = re.match(cls.pattern0, line, re.VERBOSE) or re.match(
            cls.pattern1, line, re.VERBOSE
        )

        if not match:
            print("Couldn't match validation line:", repr(line))
            return None

        epoch = 0
        if last_train:
            epoch = last_train.epoch

        return cls(
            epoch,
            int(match.group("batch")),
            int(match.group("batch_max")),
            float(match.group("loss")),
            float(match.group("mean_loss")),
            float(match.group("acc1")),
            float(match.group("mean_acc1")),
            float(match.group("acc5")),
            float(match.group("mean_acc5")),
        )


@dataclasses.dataclass
class TrainLine:
    epoch: int
    epoch_max: int
    batch: int
    batch_max: int
    lr: float
    wd: float
    loss: float
    mean_loss: float
    grad_norm: float
    mean_grad_norm: float
    loss_scale: float
    mean_loss_scale: float

    pattern0 = r"""
    ^\[.*?\]\                                                           # [2022-06-08 08:35:04 ...]
    INFO:\ Epoch:\ \[(?P<epoch>\d+)\]
    \ \ 
    \[\ *(?P<batch>\d+)/(?P<batch_max>\d+)\]                               #  [700/5247]
    \ \                                                                  #
    eta:\ (\d\ day,\ )?\d+:\d\d:\d\d                                     # eta 3:11:57
    \ \                                                                  #
    lr:\ (?P<lr>\d\.\d+)                                                 # lr 0.000040
    \ \                                                                 #
    loss:\ (?P<loss>[\w.]+)\ \((?P<mean_loss>[\w.]+)\)                   # loss 4.3632 (3.7333)
    \ \                                                                 #
    grad_norm:\ (?P<grad_norm>[\w.]+)\ \((?P<mean_grad_norm>[\w.]+)\)    # grad_norm 9.5996 (inf)
    \ \ 
    time:\ (?P<time>\d\.\d+)
    $
    """

    pattern1 = r"""
    ^\[.*?\]\                                                           # [2022-06-08 08:35:04 ...]
    \(main.py\ \d+\):\                                                  #
    INFO\ Train:\                                                       #
    \[(?P<epoch>\d+)/(?P<epoch_max>\d+)\]                               #  [56/90]
    \[(?P<batch>\d+)/(?P<batch_max>\d+)\]                               #  [700/5247]
    \t                                                                  #
    eta\ (\d\ day,\ )?\d+:\d\d:\d\d                                     # eta 3:11:57
    \                                                                   #
    lr\ (?P<lr>\d\.\d+)                                                 # lr 0.000040
    \t                                                                  #
    time\ \d+\.\d+\ \(\d+\.\d+\)                                        # time 2.5379
    \t                                                                  #
    loss\ (?P<loss>[\w.]+)\ \((?P<mean_loss>[\w.]+)\)                   # loss 4.3632 (3.7333)
    \t                                                                  #
    grad_norm\ (?P<grad_norm>[\w.]+)\ \((?P<mean_grad_norm>[\w.]+)\)    # grad_norm 9.5996 (inf)
    \t                                                                  #
    mem\ (?P<mem>.*)                                                    # mem 36916MB
    $
    """

    @classmethod
    def from_raw_line(cls, line):
        if "Train" not in line and "Epoch" not in line:
            return None

        match = re.match(cls.pattern1, line, re.VERBOSE) or re.match(
            cls.pattern0, line, re.VERBOSE
        )

        if not match:
            print("Can't match training line:", repr(line))
            return None

        return cls(
            int(match.group("epoch")),
            int(match.group("epoch_max")) if "epoch_max" in match.groupdict() else 0,
            int(match.group("batch")),
            int(match.group("batch_max")),
            float(match.group("lr")),
            0.0,  # float(match.group("wd")),
            float(match.group("loss")),
            float(match.group("mean_loss")),
            float(match.group("grad_norm")),
            float(match.group("mean_grad_norm")),
            0.0,  # float(match.group("loss_scale")),
            0.0,  # float(match.group("mean_loss_scale")),
        )


def parse_file(filepath):
    """
    Turns each line into a TrainLine instance.
    """
    with open(filepath) as fd:
        raw = [line.strip() for line in fd]

    lines = []
    last_train_line = None
    for line in raw:
        train_line = TrainLine.from_raw_line(line)
        if train_line:
            last_train_line = train_line

        val_line = ValidationLine.from_raw_line(line, last_train_line)

        lines.append(train_line or val_line)

    # Filter Nones
    return [line for line in lines if line]


def filter_same_epochs(lines):
    """
    If any of the lines have the same epoch (from training restarting), use only the most recent one.
    """

    if len(set(line.epoch for line in lines)) == len(lines):
        return lines

    assert sorted(lines, key=lambda line: line.epoch) == lines

    cleaned = []
    for second, first in preface.grouped(list(reversed(lines)), size=2):
        if first.epoch == second.epoch:
            continue

        cleaned.append(second)

    assert len(set(line.epoch for line in cleaned)) == len(cleaned)

    return list(reversed(cleaned))


def plot_all(train_epochs, train_loss, val_epochs, val_loss, val_acc1):
    fig, ax = plt.subplots()

    ax.plot(train_epochs, train_loss, color="tab:blue", label="Train Loss")
    ax.plot(val_epochs, val_loss, color="tab:orange", label="Val. Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Loss")

    ax2 = ax.twinx()
    ax2.plot(val_epochs, val_acc1, color="tab:green", label="Val Acc@1")
    ax2.set_ylabel("Accuracy")

    fig.legend()

    return fig


def select_field(lines, field, *, epochs=None):
    true_batch_max = max(line.batch for line in lines)
    last_batches = [
        line
        for line in lines
        if line.batch == true_batch_max
        and (line.epoch in epochs if epochs is not None else True)
    ]
    last_batches = filter_same_epochs(last_batches)

    return [getattr(line, field) for line in last_batches]


def visualize(data, run_name, last):
    # Visualize validation error per epoch
    val_lines = [line for line in data if isinstance(line, ValidationLine)]
    val_loss = select_field(val_lines, "mean_loss")
    val_acc1 = select_field(val_lines, "mean_acc1")
    val_epochs = select_field(val_lines, "epoch")
    assert len(set(val_epochs)) == len(val_epochs)

    # Visualize training loss per epoch.
    # Use mean_loss for the last batch in each epoch.
    train_lines = [line for line in data if isinstance(line, TrainLine)]

    train_loss = select_field(train_lines, "mean_loss", epochs=val_epochs)
    train_epochs = select_field(train_lines, "epoch", epochs=val_epochs)
    assert len(set(train_epochs)) == len(train_epochs)

    assert val_epochs == train_epochs

    fig = plot_all(train_epochs, train_loss, val_epochs, val_loss, val_acc1)
    fig.suptitle(f"{run_name} Progress")
    fig.savefig("training-loss.pdf")

    # Also look at the latest values

    def clip(lst):
        return lst[-last:]

    fig = plot_all(
        clip(train_epochs),
        clip(train_loss),
        clip(val_epochs),
        clip(val_loss),
        clip(val_acc1),
    )

    fig.suptitle(f"{run_name} Last {last} Epochs")
    fig.savefig(f"training-loss-{last}-epochs.pdf")


def get_run_name(file):
    pattern = r".*?/.*?/(.*?)/.*\.txt"

    return re.match(pattern, file).group(1)


def plot_lr(data):
    import matplotlib.pyplot as plt

    train_lines = [line for line in data if isinstance(line, TrainLine)]
    ys = [line.lr for line in train_lines]
    xs = list(range(len(ys)))

    fig, ax = plt.subplots()

    ax.plot(xs, ys, linewidth=0.1)

    fig.savefig("learning-rates.pdf")


def best_validation_epoch(data):
    # Visualize validation error per epoch
    val_lines = [line for line in data if isinstance(line, ValidationLine)]

    true_batch_max = max(line.batch for line in val_lines)
    last_batches = [line for line in val_lines if line.batch == true_batch_max]
    last_batches = filter_same_epochs(last_batches)

    return min(last_batches, key=lambda l: l.mean_loss)


def main():
    args = parse_args()

    data = parse_file(args.file)

    run_name = args.name if args.name else get_run_name(args.file)

    visualize(data, run_name, last=args.last)

    plot_lr(data)

    print(best_validation_epoch(data))


if __name__ == "__main__":
    main()
