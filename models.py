"""
Brevitas 4-bit quantized 1D CNN for Transportation Mode Detection (TMD).
Input shape : (batch, 3, window_size)   – 3-axis accelerometer
Output shape: (batch, num_classes)
"""

from pathlib import Path

import torch
import torch.nn as nn
from brevitas.nn import QuantConv1d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.export import export_qonnx


class TMD_1DCNN(nn.Module):
    """
    Three-block 1D-CNN with 4-bit quantised weights and activations (Brevitas).

    Architecture:
        Input  -> quant_inp (4-bit)
        Conv1(3->32, k=7, s=1) -> BN -> QReLU4
        Conv2(32->64, k=5, s=2) -> BN -> QReLU4
        Conv3(64->128, k=3, s=2) -> BN -> QReLU4
        GlobalAvgPool -> Flatten
        Linear(128->64) -> QReLU4
        Linear(64->num_classes)
    """

    def __init__(self, num_classes: int = 5, window_size: int = 128):
        super().__init__()

        # -- input quantisation -------------------------------------------------
        # return_quant_tensor=True so QuantConv1d can read the input scale;
        # conv layers themselves return plain tensors so standard BN/Pool work.
        self.quant_inp = QuantIdentity(bit_width=4, return_quant_tensor=True)

        # -- convolutional backbone ---------------------------------------------
        self.conv1 = QuantConv1d(
            3, 32, kernel_size=7, padding=3,
            weight_bit_width=4, bias=False, return_quant_tensor=False,
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = QuantReLU(bit_width=4, return_quant_tensor=True)

        self.conv2 = QuantConv1d(
            32, 64, kernel_size=5, stride=2, padding=2,
            weight_bit_width=4, bias=False, return_quant_tensor=False,
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = QuantReLU(bit_width=4, return_quant_tensor=True)

        self.conv3 = QuantConv1d(
            64, 128, kernel_size=3, stride=2, padding=1,
            weight_bit_width=4, bias=False, return_quant_tensor=False,
        )
        self.bn3 = nn.BatchNorm1d(128)
        # relu3 feeds into AdaptiveAvgPool1d (standard layer) → plain tensor
        self.relu3 = QuantReLU(bit_width=4, return_quant_tensor=False)

        self.gap = nn.AdaptiveAvgPool1d(1)

        # -- classifier head ---------------------------------------------------
        self.fc1 = QuantLinear(128, 64, weight_bit_width=4, bias=False,
                               return_quant_tensor=False)
        self.relu_fc = QuantReLU(bit_width=4, return_quant_tensor=False)
        self.fc2 = QuantLinear(64, num_classes, weight_bit_width=4, bias=True)

    # -------------------------------------------------------------------------
    def forward(self, x):
        # x: (B, 3, T)
        x = self.quant_inp(x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = self.gap(x).squeeze(-1)   # (B, 128)

        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)               # (B, num_classes)
        return x


# ---------------------------------------------------------------------------
# QONNX export helper
# ---------------------------------------------------------------------------
def export_model_to_qonnx(
    model: TMD_1DCNN,
    export_path: Path | str,
    window_size: int = 128,
    device: torch.device | None = None,
) -> None:
    """
    Export a trained TMD_1DCNN to QONNX format (.onnx) using Brevitas.

    Parameters
    ----------
    model       : trained TMD_1DCNN instance (will be moved to CPU for export)
    export_path : destination .onnx file path
    window_size : number of time-steps the model was trained on
    device      : device the model currently lives on (defaults to cpu)
    """
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Brevitas QONNX export must happen on CPU
    model_cpu = model.cpu().eval()
    dummy_input = torch.zeros(1, 3, window_size, dtype=torch.float32)  # (B, C, T)

    export_qonnx(
        model_cpu,
        args=dummy_input,
        export_path=str(export_path),
        opset_version=11,
    )

    # Move model back to original device
    if device is not None:
        model.to(device)
