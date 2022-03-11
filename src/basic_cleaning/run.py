#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """Executes this MLflow run"""
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)
    logger.info(f"Read input artifact with shape {df.shape}")

    logger.info("Running basic cleaning...")
    # Drop outliers
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info(f"Saving output artifact with shape {df.shape}")
    # Geolocation
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    # Save to csv
    df.to_csv(args.output_artifact, index=False)
    # Upload to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    logger.info("Logging output artifact in W&B")
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    logger.info("Run end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The W&B input artifact (data to be cleaned)",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The W&B output artifact (cleaned data)",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact (a human-readable name)",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Min price for renting to be considered",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Max price for renting to be considered",
        required=True
    )

    args = parser.parse_args()

    go(args)
