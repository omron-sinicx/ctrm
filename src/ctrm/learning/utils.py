"""utilities for learning
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

from .formats import (FormatInput, FormatOutput, reconstruct_format_input,
                      reconstruct_format_output)
from .model import Model, reconstruct_model


def save(
    basename: str,
    model: Model,
    format_input: FormatInput,
    format_output: FormatOutput,
) -> None:
    """save model, format_input, format_output

    Args:
        basename (str): dirname + label, e.g., xxxx/best
        model (Model): target model
        format_input (FormatInput): target format_input
        format_output (FormatOutput): target format_output
    """
    model.save(basename)
    format_input.save(basename)
    format_output.save(basename)


def reconstruct(basename: str) -> tuple[Model, FormatInput, FormatOutput]:
    """reconstruct model, format_input, format_output
    Args:
        basename (str): dirname + label, e.g., xxxx/best
            special case: len(basename)==19
            -> target will be /workspace/log/{basename}/best"

    Returns
        Model: reconstructed model
        FormatInput: reconstructed format_input
        FormatOutput: reconstructed format_output
    """
    if len(basename) == 19:  # date
        basename = f"/workspace/log/{basename}/best"

    format_input = reconstruct_format_input(basename)
    model = reconstruct_model(basename)
    return (
        model,
        format_input,
        reconstruct_format_output(basename),
    )
