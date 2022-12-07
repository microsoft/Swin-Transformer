import argparse
import os
import pathlib
import yaml
import tomli
import tomli_w
from tqdm.auto import tqdm
# import sys
# sys.path.append("..")

# from .. import config, logging, templating, util

import logger, config, utils
logger = logger.init("experiments.generate")


from . import templating #, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General .toml files from template .toml files. I kept all my templates in experiments/templates and my generated experiment configs in experiments/generated, which I then removed from version control.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy to use to combine multiple lists in a template.",
        default="grid",
        choices=["grid", "paired", "random"],
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Number of configs to generate when using --strategy random. Required.",
        default=-1,
    )
    parser.add_argument(
        "--no-expand",
        type=str,
        nargs="+",
        default=[],
        help=".-separated fields to not expand",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="generated-",
        help="Prefix to add to generated templates",
    )
    parser.add_argument(
        "templates",
        nargs="+",
        type=str,
        help="Template .toml files or directories containing template .toml files.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output directory to write the generated .toml files to.",
    )
    return parser.parse_args()


def generate(args: argparse.Namespace) -> None:

    strategy = templating.Strategy.new(args.strategy)

    count = args.count

    if strategy is templating.Strategy.random:
        assert count > 0, "Need to include --count!"

    for template_toml in utils.files_with_extension(args.templates, ".yaml"):

        with open(template_toml, "rb") as template_file:
            try:
                template_dict = yaml.safe_load(template_file)
            except tomli.TOMLDecodeError as err: #ToDo have to replace this for yaml file
                logger.warning(
                    "Error parsing template file. [file: %s, err: %s]",
                    template_toml,
                    err,
                )
                continue
        

        template_name = pathlib.Path(template_toml).stem


        logger.info("Opened template file. [file: %s]", template_toml)
        # print ("\n\n")
        # print (template_dict)

        experiment_dicts = templating.generate(
            template_dict, strategy, count=count, no_expand=set(args.no_expand)
        )


        logger.info(
            "Loaded experiment dictionaries. [count: %s]", len(experiment_dicts)
        )

        for i, experiment_dict in enumerate(tqdm(experiment_dicts)):
            outputfile=experiment_dict["EXPERIMENT"]["NAME"]
            experiment_dict["EXPERIMENT"]["NAME"]=os.path.join(outputfile, str(i))
            filename = f"{args.prefix}{template_name}-{i}.yaml"
            filepath = os.path.join(args.output, filename)
            with open(filepath, "w") as file:
                yaml.dump(experiment_dict, file)



            #exit()
            # Verifies that the configs are correctly loaded.
            #list(config.load_configs(filepath))
            


def main() -> None:
    args = parse_args()

    # print (args.no_expand)

    generate(args)


if __name__ == "__main__":
    main()
