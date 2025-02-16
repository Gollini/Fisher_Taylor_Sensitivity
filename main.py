"""
Main script to run the experiment(s) in the specified path.
"""
#Imports
import os
import argparse

from batch_exp import batch
# from batch_exp import batch_lth

def main():
    parser = argparse.ArgumentParser(
        description="Run the experiment(s) in the specified path."
    )
    parser.add_argument(
        "--experiment",
        default="pbt",
        type=str,
        help="""Name of the experiment to be run.""",
    )
    parser.add_argument(
        "--config",
        default="./exp_configs/",
        type=str,
        help="""Path to the json parameter files for the experiment(s) to be run.""",
    )
    parser.add_argument(
        "-D",
        "--debug",
        default=False,
        action="store_true",
        help="""Flag to set the experiment to debug mode.""",
    )
    args = parser.parse_args()

    if args.experiment == "pbt":
        exprs_batch = batch.ExperimentBatch(args.config, args.debug)
        exprs_batch.run()

if __name__ == "__main__":
    main()    