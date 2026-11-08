from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .rpcr import RPCROutput


def rpcr_loss(
    output: RPCROutput,
    labels: torch.Tensor,
    auxiliary_weight: float = 0.1,
    correlation_weight: float = 0.05,
    router_weight: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    labels = labels.reshape(-1)
    main_mae = F.l1_loss(output.prediction, labels)
    auxiliary_losses = [
        F.l1_loss(prediction, labels)
        for prediction in output.auxiliary_predictions.values()
    ]
    auxiliary_mae = (
        torch.stack(auxiliary_losses).mean()
        if auxiliary_losses
        else output.prediction.new_zeros(())
    )

    centered_prediction = output.prediction - output.prediction.mean()
    centered_labels = labels - labels.mean()
    correlation = torch.sum(centered_prediction * centered_labels) / (
        torch.sqrt(torch.sum(centered_prediction.square()) + 1e-8)
        * torch.sqrt(torch.sum(centered_labels.square()) + 1e-8)
    )
    correlation_loss = 1.0 - correlation
    total_loss = (
        main_mae
        + auxiliary_weight * auxiliary_mae
        + correlation_weight * correlation_loss
        + router_weight * output.router_loss
    )
    return total_loss, {
        "main_mae": main_mae.detach(),
        "auxiliary_mae": auxiliary_mae.detach(),
        "correlation_loss": correlation_loss.detach(),
        "router_loss": output.router_loss.detach(),
    }
