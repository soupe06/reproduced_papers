"""MerLin photonic adaptation of the vertical quantum federated model (Phase 4).

This is an ADAPTATION, not the paper's circuit. Each party (client or server) is one
photonic chip: ``stages`` repetitions of ``[trainable interferometer mesh, angle
encoding of 8 inputs]`` followed by a final mesh. Clients read out the expected photon
number of every mode (8 real numbers that are sent to the server); the server turns its
output distribution into 10 class scores.

Computation space. ``UNBUNCHED`` in MerLin evolves the full Fock state exactly and then
keeps only collision-free output patterns, renormalised (post-selection; see LOG.md,
"Correction - UNBUNCHED semantics"). ``FOCK`` keeps every outcome, i.e. photon-number-
resolving detection without post-selection, and lets a phase on a mode holding k
photons contribute frequency k (Gan et al., `papers/fock_state_expressivity`).

MerLin quirk handled here: an encoding or mesh name that ends in a digit makes
``add_angle_encoding(scale=...)`` silently ignore the scale, so stage names use letters
only (``xa``, ``xb``, ...).
"""

from __future__ import annotations

import logging
import string

import merlin as ml
import torch
from torch import nn

logger = logging.getLogger(__name__)

PIXELS_PER_CLIENT = 32
N_CLASSES = 10
SERVER_ENCODINGS = {"server_wide", "server_narrow"}
SPACES = {"unbunched": ml.ComputationSpace.UNBUNCHED, "fock": ml.ComputationSpace.FOCK}
# Server readouts: 0 = fixed grouping (fully photonic), 1 = trainable linear head on the
# full output distribution (D1), 2 = trainable linear head on the per-mode
# photon-number marginals (D2; keeps the classical part small for large Fock spaces).
SERVER_READOUTS = {0: "grouped_probabilities", 1: "linear_head", 2: "marginal_head"}


def _input_state(n_modes: int, n_photons: int) -> list[int]:
    """Explicit input state: every second mode occupied (n = m/2) or every mode (n = m)."""
    if 2 * n_photons == n_modes:
        return [1 if mode % 2 == 0 else 0 for mode in range(n_modes)]
    if n_photons == n_modes:
        return [1] * n_modes
    raise ValueError(
        f"No input-state rule for {n_photons} photons in {n_modes} modes; "
        "use n_photons = n_modes / 2 (spread) or n_photons = n_modes (one per mode)"
    )


def _marginal_matrix(
    keys: list[tuple[int, ...]], n_modes: int, n_photons: int
) -> torch.Tensor:
    """0/1 matrix mapping outcome probabilities to per-mode photon-number marginals.

    Column ``mode * (n_photons + 1) + count`` sums the probabilities of every outcome
    with ``count`` photons in ``mode``.
    """
    matrix = torch.zeros(len(keys), n_modes * (n_photons + 1))
    for row, key in enumerate(keys):
        for mode, count in enumerate(key):
            matrix[row, mode * (n_photons + 1) + count] = 1.0
    return matrix


def _click_matrix(keys: list[tuple[int, ...]], n_modes: int) -> torch.Tensor:
    """0/1 matrix mapping outcome probabilities to per-mode click probabilities.

    Column ``mode`` sums the probabilities of every outcome with at least one photon
    in ``mode``, i.e. what a threshold detector on that mode reports.
    """
    matrix = torch.zeros(len(keys), n_modes)
    for row, key in enumerate(keys):
        for mode, count in enumerate(key):
            if count > 0:
                matrix[row, mode] = 1.0
    return matrix


CLIENT_READOUTS = {"mode_expectations", "click_probabilities"}


class PhotonicParty(nn.Module):
    """One party: a data re-uploading photonic chip.

    Parameters
    ----------
    n_stages : int
        Number of ``[mesh, encoding of encoded_modes inputs]`` stages; the party takes
        ``n_stages * encoded_modes`` input values.
    readout : str
        ``"mode_expectations"`` (client message: mean photon number per mode, which
        in FOCK space only reflects single-photon interference),
        ``"click_probabilities"`` (client message: P(n_k >= 1) per mode, i.e. threshold
        detection, which carries multi-photon interference),
        ``"grouped_probabilities"`` (fixed grouping of the output distribution into class
        probabilities, fully photonic), ``"linear_head"`` (trainable linear map of the
        full output distribution to class logits) or ``"marginal_head"`` (trainable
        linear map of the per-mode photon-number marginals to class logits).
    encoding_scale : float
        Multiplier of the input values before they become phase shifts.
    n_modes, n_photons : int
        Chip size and photon number.
    computation_space : str
        ``"unbunched"`` (post-selected collision-free outcomes) or ``"fock"``
        (photon-number-resolving, no post-selection).
    dtype : torch.dtype
        Real dtype of the circuit parameters. Default value is torch.float32.
    encoded_modes : int | None
        Number of leading modes carrying the angle encoding in each stage; the
        remaining modes only interfere. ``None`` encodes on every mode. Default value is
        None.
    """

    def __init__(
        self,
        n_stages: int,
        readout: str,
        encoding_scale: float,
        n_modes: int,
        n_photons: int,
        computation_space: str = "unbunched",
        dtype: torch.dtype = torch.float32,
        encoded_modes: int | None = None,
    ) -> None:
        super().__init__()
        if n_stages > len(string.ascii_lowercase):
            raise ValueError("Too many stages for letter-only parameter names")
        if computation_space not in SPACES:
            raise ValueError(f"Unknown computation_space {computation_space!r}")
        encoded_modes = n_modes if encoded_modes is None else encoded_modes
        if not 0 < encoded_modes <= n_modes:
            raise ValueError(
                f"encoded_modes must be in 1..{n_modes}, got {encoded_modes}"
            )
        self.n_inputs = n_stages * encoded_modes
        builder = ml.CircuitBuilder(n_modes=n_modes)
        for stage in range(n_stages):
            letter = string.ascii_lowercase[stage]
            builder.add_entangling_layer(name=f"mesh{letter}")
            builder.add_angle_encoding(
                modes=list(range(encoded_modes)),
                name=f"enc{letter}",
                scale=encoding_scale,
            )
        builder.add_entangling_layer(name="meshz")
        space = SPACES[computation_space]
        if readout == "mode_expectations":
            strategy = ml.MeasurementStrategy.mode_expectations(space)
        elif readout in (
            "grouped_probabilities",
            "linear_head",
            "marginal_head",
            "click_probabilities",
        ):
            strategy = ml.MeasurementStrategy.probs(space)
        else:
            raise ValueError(f"Unknown readout {readout!r}")
        self.layer = ml.QuantumLayer(
            input_size=self.n_inputs,
            builder=builder,
            input_state=_input_state(n_modes, n_photons),
            n_photons=n_photons,
            measurement_strategy=strategy,
            dtype=dtype,
        )
        self.group = None
        self.head = None
        self.register_buffer("marginals", None)
        self.register_buffer("clicks", None)
        size = self.layer.output_size
        if readout == "grouped_probabilities":
            # Consecutive blocks when the outcome count divides evenly (the original
            # 8-mode server: 70 -> 10); otherwise outcomes are grouped by index modulo 10.
            self.group = (
                ml.LexGrouping(size, N_CLASSES)
                if size % N_CLASSES == 0
                else ml.ModGrouping(size, N_CLASSES)
            )
        elif readout == "linear_head":
            self.head = nn.Linear(size, N_CLASSES, dtype=dtype)
        elif readout == "marginal_head":
            matrix = _marginal_matrix(list(self.layer.output_keys), n_modes, n_photons)
            self.marginals = matrix.to(dtype)
            self.head = nn.Linear(matrix.shape[1], N_CLASSES, dtype=dtype)
        elif readout == "click_probabilities":
            self.clicks = _click_matrix(list(self.layer.output_keys), n_modes).to(dtype)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, n_inputs)`` values to the party's output."""
        if values.shape[1] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {values.shape[1]}")
        output = self.layer(values)
        if self.clicks is not None:
            return output @ self.clicks
        if self.group is not None:
            return self.group(output)
        if self.marginals is not None:
            return self.head(output @ self.marginals)
        if self.head is not None:
            return self.head(output)
        return output


class PhotonicVerticalFederatedModel(nn.Module):
    """Clients on disjoint pixel slices plus a photonic server (MerLin adaptation).

    Parameters
    ----------
    client_slices : list of int
        For each client, the index of the 32-column slice of the permuted pixel vector.
    server_encoding : str
        ``"server_wide"`` (every client output is encoded once) or ``"server_narrow"``
        (one client's 8 outputs re-uploaded in every stage, as in the authors' E2 code).
    n_client_stages : int
        Stages per client for one pass over its 32 pixels (4 stages of 8 modes).
    server_modes : int
        Server chip size; it holds photons at the clients' photon density.
    server_head : int
        Server readout, see ``SERVER_READOUTS`` (0 is fully photonic).
    computation_space : str
        ``"unbunched"`` or ``"fock"`` for every chip. Default value is "unbunched".
    reuploads : int
        How many times every party re-uploads its whole input (data_reuploading idea).
        Default value is 1.
    client_readout : str
        ``"mode_expectations"`` or ``"click_probabilities"``; both send 8 values per
        client. Default value is "mode_expectations".
    """

    def __init__(
        self,
        client_slices: list[int],
        server_encoding: str,
        n_modes: int,
        n_photons: int,
        client_encoding_scale: float,
        server_encoding_scale: float,
        n_client_stages: int,
        server_modes: int,
        server_head: int,
        computation_space: str = "unbunched",
        reuploads: int = 1,
        client_readout: str = "mode_expectations",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if client_readout not in CLIENT_READOUTS:
            raise ValueError(f"Unknown client_readout {client_readout!r}")
        if server_encoding not in SERVER_ENCODINGS:
            raise ValueError(f"Unknown server_encoding {server_encoding!r}")
        if n_client_stages * n_modes != PIXELS_PER_CLIENT:
            raise ValueError(
                f"{n_client_stages} stages of {n_modes} modes do not encode "
                f"{PIXELS_PER_CLIENT} pixels"
            )
        if int(server_head) not in SERVER_READOUTS:
            raise ValueError(f"Unknown server_head {server_head!r}")
        if reuploads < 1:
            raise ValueError("reuploads must be at least 1")
        self.client_slices = list(client_slices)
        self.dtype = dtype
        self.reuploads = int(reuploads)
        self.clients = nn.ModuleList(
            PhotonicParty(
                n_client_stages * self.reuploads,
                client_readout,
                client_encoding_scale,
                n_modes,
                n_photons,
                computation_space,
                dtype,
            )
            for _ in self.client_slices
        )
        message_size = n_modes * len(self.client_slices)
        if server_encoding == "server_wide":
            server_stages, self.server_repeats = message_size // n_modes, 1
        else:
            # Narrow server: re-upload the single client's message in every stage.
            server_stages, self.server_repeats = n_client_stages, n_client_stages
            if len(self.client_slices) != 1:
                raise ValueError("server_narrow takes exactly one client")
        if server_modes < n_modes:
            raise ValueError("server_modes must be at least the client message width")
        if (n_photons * server_modes) % n_modes:
            raise ValueError("server_modes must keep an integer photon number")
        server_photons = n_photons * server_modes // n_modes
        self.server_readout = SERVER_READOUTS[int(server_head)]
        self.server_head = self.server_readout != "grouped_probabilities"
        # Each server stage encodes one client-message width (8 values) on its leading
        # modes; the server holds photons at the same density as the clients.
        self.server = PhotonicParty(
            server_stages * self.reuploads,
            self.server_readout,
            server_encoding_scale,
            server_modes,
            server_photons,
            computation_space,
            dtype,
            encoded_modes=n_modes,
        )
        logger.info(
            "PHOTONIC_MODEL | space=%s reuploads=%d | client modes=%d photons=%d "
            "stages=%d | server modes=%d photons=%d stages=%d outputs=%d readout=%s",
            computation_space,
            self.reuploads,
            n_modes,
            n_photons,
            n_client_stages * self.reuploads,
            server_modes,
            server_photons,
            server_stages * self.reuploads,
            self.server.layer.output_size,
            self.server_readout,
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Log class probabilities for ``(batch, 256)`` permuted pixels.

        The data pipeline stores pixels as gate-model angles in ``{0, pi/4}``; the
        photonic encoding uses the binary pixel (angle > 0) times the encoding scale.
        Returning ``log p`` makes the unchanged ``CrossEntropyLoss`` equal to the
        negative log-likelihood of the class probabilities.
        """
        binary = (pixels > 0).to(self.dtype)
        messages = []
        for client, slice_index in zip(self.clients, self.client_slices):
            start = slice_index * PIXELS_PER_CLIENT
            client_input = binary[:, start : start + PIXELS_PER_CLIENT]
            messages.append(client(client_input.repeat(1, self.reuploads)))
        message = torch.cat(messages, dim=1)
        repeats = self.server_repeats * self.reuploads
        if repeats > 1:
            message = message.repeat(1, repeats)
        if self.server_head:
            return torch.log_softmax(self.server(message), dim=1)
        probabilities = self.server(message)
        return torch.log(probabilities.clamp_min(1e-9))

    def trainable_parameter_counts(self) -> dict[str, int]:
        """Trainable phases per party and in total."""
        counts = {
            f"client_{i}": sum(p.numel() for p in c.parameters())
            for i, c in enumerate(self.clients)
        }
        counts["server"] = sum(p.numel() for p in self.server.parameters())
        # Classical parameters of the server readout head (zero when fully photonic);
        # already included in "server".
        head = self.server.head
        counts["server_classical_head"] = (
            sum(p.numel() for p in head.parameters()) if head is not None else 0
        )
        counts["total"] = sum(
            v for k, v in counts.items() if k != "server_classical_head"
        )
        return counts
