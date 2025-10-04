import glob
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import typer
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.table import Table
from typing_extensions import Annotated

from . import __version__
from .__init__ import console
from .scripts.convert_to_sdf import app as convert_to_sdf_app
from .transformer.model import InstaNovo
from .predict import get_preds as transformer_get_preds
from .train import _set_author_neptune_api_token
from .train import train as train_transformer
from .utils.colorlogging import ColorLog
from .utils.s3 import S3FileHandler

# Filter out a SyntaxWarning from pubchempy
warnings.filterwarnings(
    "ignore",
    message=r'"is not" with \'int\' literal\. Did you mean "!="\?',
    category=SyntaxWarning,
    module="pubchempy",
)

logger = ColorLog(console, __name__).logger

DEFAULT_TRAIN_CONFIG_PATH = "configs"
DEFAULT_INFERENCE_CONFIG_PATH = "configs/inference"
DEFAULT_INFERENCE_CONFIG_NAME = "default"

# 创建一个主 Typer 应用
app = typer.Typer(rich_markup_mode="rich", pretty_exceptions_enable=False)
app.add_typer(convert_to_sdf_app)


def compose_config(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Compose Hydra configuration with given overrides."""
    logger.info(f"Reading config from '{config_path}' with name '{config_name}'.")
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=False)
        return cfg


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """
    InstaNovo: A transformer-based de novo peptide sequencing model.
    """
    # 如果只运行 `instanovo`，则显示帮助信息
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("predict")
def predict(
    data_path: Annotated[
        Optional[str],
        typer.Option("--data-path", "-d", help="Path to input data file"),
    ] = None,
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output-path",
            "-o",
            help="Path to output file.",
            exists=False,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    instanovo_model: Annotated[
        Optional[str],
        typer.Option(
            "--instanovo-model",
            "-i",
            help=(
                "Either a model ID (currently supported: "
                f"""{", ".join(f"'{model_id}'" for model_id in InstaNovo.get_pretrained())})"""
                " or a path to an Instanovo checkpoint file (.ckpt format)."
            ),
        ),
    ] = None,
    denovo: Annotated[
        Optional[bool],
        typer.Option(
            "--denovo/--evaluation",
            help="Do [i]de novo[/i] predictions or evaluate an annotated file "
            "with peptide sequences?",
        ),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option("--config-path", "-cp", help="Relative path to config directory."),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name",
            "-cn",
            help="The name of the config (usually the file name without the .yaml extension).",
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Run predictions with InstaNovo."""
    logger.info("Initializing InstaNovo inference.")

    s3 = S3FileHandler()

    if config_path is None:
        config_path = DEFAULT_INFERENCE_CONFIG_PATH
    if config_name is None:
        config_name = DEFAULT_INFERENCE_CONFIG_NAME

    config = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )

    # Check config inputs
    if data_path is not None:
        if "*" in data_path or "?" in data_path or "[" in data_path:
            if not glob.glob(data_path):
                raise ValueError(f"The data_path '{data_path}' doesn't correspond to any file(s).")
        config.data_path = str(data_path)

    if not config.get("data_path", None) and data_path is None:
        raise ValueError(
            "Expected 'data_path' but found None. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`--data-path='path/to/data'`."
        )

    if denovo is not None:
        config.denovo = denovo

    if output_path is not None:
        if output_path.exists():
            logger.info(f"Output path '{output_path}' already exists and will be overwritten.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path = str(output_path)
    if config.get("output_path", None) is None and config.get("denovo", False):
        raise ValueError(
            "Expected 'output_path' but found None in denovo mode. Please specify it in the "
            "config or with the cli flag `--output-path=path/to/output_file`."
        )

    if instanovo_model is not None:
        if Path(instanovo_model).is_file() and Path(instanovo_model).suffix != ".ckpt":
            raise ValueError(f"Checkpoint file '{instanovo_model}' should end with '.ckpt'.")
        if (
            not Path(instanovo_model).is_file()
            and not instanovo_model.startswith("s3://")
            and instanovo_model not in InstaNovo.get_pretrained()
        ):
            raise ValueError(
                f"InstaNovo model ID '{instanovo_model}' is not supported. "
                f"Supported: {', '.join(f'{model_id}' for model_id in InstaNovo.get_pretrained())}"
            )
        config.instanovo_model = instanovo_model

    if not config.get("instanovo_model", None):
        raise ValueError(
            "Expected 'instanovo_model' but found None. Specify it in the config or with "
            "`--instanovo_model=path/to/model.ckpt`."
        )

    logger.info(f"Loading InstaNovo model {config.instanovo_model}")
    if config.instanovo_model in InstaNovo.get_pretrained():
        model_path = s3.get_local_path(config.instanovo_model)
        assert model_path is not None
        transformer_model, transformer_config = InstaNovo.from_pretrained(model_path)
    else:
        model_path = s3.get_local_path(config.instanovo_model)
        assert model_path is not None
        transformer_model, transformer_config = InstaNovo.load(model_path)

    logger.info(f"InstaNovo config:\n{OmegaConf.to_yaml(config)}")
    logger.info(
        f"InstaNovo model params: {np.sum([p.numel() for p in transformer_model.parameters()]):,d}"
    )

    if config.get("save_beams", False) and config.get("num_beams", 1) == 1:
        logger.warning("num_beams is 1 and will override save_beams. Only use save_beams in beam search.")
        with open_dict(config):
            config["save_beams"] = False

    logger.info(f"Performing search with {config.get('num_beams', 1)} beams")
    transformer_get_preds(config, transformer_model, transformer_config, s3)


@app.command("train")
def train(
    config_path: Annotated[
        Optional[str],
        typer.Option("--config-path", "-cp", help="Relative path to config directory."),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name", "-cn", help="The name of the config (without .yaml extension)."
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Train the InstaNovo model."""
    _set_author_neptune_api_token()

    if config_path is None:
        config_path = DEFAULT_TRAIN_CONFIG_PATH
    if config_name is None:
        config_name = "instanovo"

    config = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )

    if config["n_gpu"] > 1:
        raise ValueError("n_gpu > 1 currently not supported.")

    logger.info("Initializing InstaNovo training.")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    # Unnest hydra configs
    sub_configs_list = ["model", "dataset", "residues"]
    for sub_name in sub_configs_list:
        if sub_name in config:
            with open_dict(config):
                temp = config[sub_name]
                del config[sub_name]
                config.update(temp)

    logger.info(f"InstaNovo training config:\n{OmegaConf.to_yaml(config)}")
    train_transformer(config)


@app.command()
def version() -> None:
    """Display version information for InstaNovo and its dependencies."""
    table = Table("Package", "Version")
    table.add_row("InstaNovo", __version__)
    table.add_row("NumPy", np.__version__)
    table.add_row("PyTorch", torch.__version__)
    table.add_row("Lightning", L.__version__)
    console.print(table)


def instanovo_entrypoint() -> None:
    """Main entry point for the InstaNovo CLI application."""
    app()


if __name__ == "__main__":
    app()