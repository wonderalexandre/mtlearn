"""Reference CFP implementation that materializes the dense Jacobian.

This module is kept as a readable baseline for validating the primary implicit
implementation. It computes the same node-wise CFP criterion as
``ConnectedFilterPreprocessingLayer`` but reconstructs pixels through an
explicit dense tree-to-pixel Jacobian. This makes the math easier to inspect
and gradcheck, at the cost of much higher memory use.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from .. import morphology
import mtlearn
from ._helpers import (
    group_name,
    to_numpy_u8,
    build_tree,
    update_ds_stats,
    normalize_with_ds_stats,
    maybe_refresh_norm_for_key,
    make_stats_payload,
    load_stats_payload,
    IndexedDatasetWrapper
)

class ConnectedFilterPreprocessingExplicitJacobianFunction(torch.autograd.Function):
    """Autograd function using a materialized tree-to-pixel Jacobian."""

    @staticmethod
    def forward(
        ctx,
        jacobian: torch.Tensor,
        residues: torch.Tensor,
        numRows: int,
        numCols: int,
        attrs2d: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        beta_f: float = 1.0,
        clamp_logits: bool = False
    ):
        """Apply CFP with a materialized tree-to-pixel Jacobian.

        Args:
            ctx: PyTorch autograd context.
            jacobian: Dense node-to-pixel Jacobian with shape
                ``(num_nodes, num_pixels)``.
            residues: Tree residues, one value per node.
            numRows: Number of rows in the reconstructed image.
            numCols: Number of columns in the reconstructed image.
            attrs2d: Normalized attributes with shape ``(num_nodes, K)``.
            weight: Learnable group weight vector with shape ``(K,)``.
            bias: Learnable scalar bias.
            beta_f: Forward sigmoid gain.
            clamp_logits: Whether to clamp ``beta_f * logits`` before sigmoid.

        Returns:
            Filtered image with shape ``(numRows, numCols)``.
        """
        assert attrs2d.dim() == 2, "attrs2d must have shape (num_nodes, K)"
        assert weight.dim() == 1, "weight must have shape (K,)"

        logits = attrs2d @ weight.view(-1) + bias.view(())
        s = beta_f * logits
        if clamp_logits:
            s = torch.clamp(s, -12.0, 12.0)

        sigmoid = torch.sigmoid(s)

        y_pred = (jacobian.T @ (residues * sigmoid)).reshape(numRows, numCols)

        ctx.jacobian = jacobian
        ctx.residues = residues
        ctx.beta_f = beta_f

        ctx.save_for_backward(attrs2d, sigmoid)
        return y_pred


    @staticmethod
    def backward(ctx, grad_output):
        """Return gradients for the explicit-Jacobian forward inputs.

        Only ``weight`` and ``bias`` receive gradients. The dense Jacobian,
        residues, dimensions, attributes, and inference options are fixed
        preprocessing data for this autograd function.
        """
        attrs2d, sigmoid = ctx.saved_tensors

        jacobian = ctx.jacobian
        residues = ctx.residues
        beta_f = ctx.beta_f

        grad_output = grad_output.flatten()

        d_sigmoid = sigmoid * (1 - sigmoid)

        d_rec_W = jacobian.T @ (residues.unsqueeze(1) * beta_f * d_sigmoid.unsqueeze(1) * attrs2d)
        d_rec_B = jacobian.T @ (residues.unsqueeze(1) * beta_f * d_sigmoid.unsqueeze(1))

        dW = (grad_output @ d_rec_W)
        dB = (grad_output @ d_rec_B)

        return None, None, None, None, None, dW, dB, None, None


class ConnectedFilterPreprocessingLayerWithExplicitJacobian(torch.nn.Module):
    """Reference CFP layer with an explicit dense Jacobian.

    For each attribute group ``g`` with ``K`` normalized attributes
    ``A_g in R[num_nodes, K]``, the layer computes
    ``sigmoid(beta_f * (A_g @ w_g + b_g))`` and reconstructs the filtered image
    with ``jacobian.T @ (residues * sigmoid)``.

    Use this implementation for debugging and mathematical comparison with the
    implicit layer, not for memory-sensitive training on larger images.

    Args:
        in_channels: Number of input channels.
        attributes_spec: Attribute groups. Each group must contain at least one
            morphology attribute enum.
        tree_type: ``"max-tree"``, ``"min-tree"``, or any other value accepted
            by the facade as a tree of shapes.
        device: Torch device used for parameters, cached tensors, and outputs.
        scale_mode: ``"minmax01"``, ``"zscore_tree"``, ``"hybrid"``, or
            ``"none"``.
        eps: Numerical floor for normalization denominators.
        beta_f: Forward sigmoid gain.
        top_hat: If true, output the tree-type-specific top-hat residual.
        clamp_logits: If true, clamp ``beta_f * logits`` to ``[-12, 12]``.
        hybrid_k: Number of standard deviations used for hybrid clipping.
        hybrid_floor_a: Lower bound used when remapping hybrid-normalized
            attributes to ``[a, 1]``.
    """
    def __init__(self,
                 in_channels,
                 attributes_spec,
                 tree_type="max-tree",
                 device="cpu",
                 scale_mode: str = "hybrid",
                 eps: float = 1e-6,
                 beta_f: float = 1.0,
                 top_hat: bool = False,
                 clamp_logits: bool = False,
                 hybrid_k: float = 3.0,
                 hybrid_floor_a: float = 0.05,
                 ):
        """Initialize dense-Jacobian CFP caches and learnable parameters.

        This constructor mirrors the primary CFP layer but stores full tree
        handles, dense Jacobians, and residues in its cache so that forward and
        backward can use the explicit formulation.
        """
        super().__init__()

        # Hybrid normalization configuration.
        self.hybrid_k = float(hybrid_k)
        self.hybrid_floor_a = float(hybrid_floor_a)

        self.in_channels = int(in_channels)
        self.tree_type   = str(tree_type)
        self.device      = torch.device(device)
        self.scale_mode  = str(scale_mode)
        self.eps         = float(eps)
        self.beta_f      = float(beta_f)
        self.top_hat     = bool(top_hat)
        self.clamp_logits = bool(clamp_logits)

        # Attribute groups and the flat set of attribute types used by them.
        self.group_defs = []
        all_attr_types_set = set()
        for item in attributes_spec:
            group = tuple(item) if isinstance(item, (list, tuple)) else (item,)
            if len(group) < 1:
                raise ValueError("Each attribute group must contain at least one attribute.")
            self.group_defs.append(group)
            for at in group:
                all_attr_types_set.add(at)
        self._all_attr_types = list(all_attr_types_set)

        self.num_groups   = len(self.group_defs)
        self.out_channels = self.in_channels * self.num_groups

        # Tree, Jacobian, residue, attribute, and normalization cache state.
        self._trees      = {}
        self._jacobians  = {}
        self._residues   = {}
        self._base_attrs = {}
        self._norm_attrs = {}
        self._stats_epoch = 0
        self._norm_epoch_by_key = {}
        self._ds_stats = {}
        self._stats_frozen = False

        # Learnable parameters: one weight vector and one bias per group.
        self._weights = torch.nn.ParameterDict()
        self._biases  = torch.nn.ParameterDict()
        for g, group in enumerate(self.group_defs):
            k = len(group)
            gname = "+".join([t.name for t in group])
            w = torch.empty(k, dtype=torch.float32, device=self.device)
            b = torch.empty(1, dtype=torch.float32, device=self.device)
            # Xavier-like initialization for a one-dimensional parameter vector.
            fan_in, fan_out = k, 1
            gain = 1.0
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std
            torch.nn.init.uniform_(w, -a, a)
            torch.nn.init.constant_(b, 0.0)
            self._weights[gname] = torch.nn.Parameter(w, requires_grad=True)
            self._biases[gname]  = torch.nn.Parameter(b, requires_grad=True)

    # ---------- helpers ----------
    def _group_name(self, group):
        """Return the stable parameter/cache name for an attribute group."""
        return group_name(group)

    def _to_numpy_u8(self, img2d_t: torch.Tensor) -> np.ndarray:
        """Convert one image channel to the backend's ``np.uint8`` format."""
        return to_numpy_u8(img2d_t)

    def _build_tree_jacobian_and_residues(self, img_np: np.ndarray):
        """Build a tree and extract the dense Jacobian plus node residues."""
        tree = build_tree(img_np, self.tree_type)
        jacobian = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_jacobian(
            tree
        ).to(self.device)
        residues = mtlearn.ConnectedFilterPreprocessingTreeTensors.get_residues(
            tree
        ).to(self.device)
        return tree, jacobian, residues

    # ---------- normalization with hybrid support ----------
    def _update_ds_stats(self, attr_type, a_raw_1d: torch.Tensor):
        """Update dataset statistics for one raw attribute vector."""
        if getattr(self, "_stats_frozen", False):
            return
        smode = self.scale_mode
        if smode == "hybrid":
            smode = "zscore_tree"
        changed = update_ds_stats(self._ds_stats, smode, attr_type, a_raw_1d)
        if changed:
            self._stats_epoch += 1

    def _normalize_with_ds_stats(self, attr_type, a_raw_1d: torch.Tensor) -> torch.Tensor:
        """Normalize a raw attribute vector according to ``scale_mode``.

        Hybrid mode applies z-score normalization, clipping, and remapping to
        ``[hybrid_floor_a, 1]``.
        """
        if self.scale_mode != "hybrid":
            return normalize_with_ds_stats(self._ds_stats, self.scale_mode, self.eps, attr_type, a_raw_1d)

        # Hybrid mode.
        st = self._ds_stats.get(attr_type, None)
        if st is None:
            # Before stats exist, preserve the raw values so early calls do not fail.
            return a_raw_1d

        # Dataset-level z-score statistics.
        count = st["count"].to(torch.float32)
        mean  = (st["sum"] / torch.clamp(count, min=1.0)) if count.item() > 0 else torch.tensor(0.0, device=a_raw_1d.device)
        var   = (st["sumsq"] / torch.clamp(count, min=1.0) - mean * mean) if count.item() > 0 else torch.tensor(0.0, device=a_raw_1d.device)
        std   = torch.sqrt(torch.clamp(var, min=self.eps))

        # 1) z-score
        x = (a_raw_1d - mean) / std
        # 2) clip em [-k, +k]
        k = torch.tensor(self.hybrid_k, dtype=x.dtype, device=x.device)
        x = torch.clamp(x, -k, k)
        # 3) rescale to [a, 1].
        a = torch.tensor(self.hybrid_floor_a, dtype=x.dtype, device=x.device)
        x01 = a + (1.0 - a) * ((x + k) / (2.0 * k))
        return x01

    def _maybe_refresh_norm_for_key(self, key: str):
        """Refresh normalized cached attributes if dataset stats changed."""
        # Nothing to refresh until raw attributes are cached.
        if key not in self._base_attrs:
            return

        # Cache is already current for this stats epoch.
        if self._norm_epoch_by_key.get(key, -1) == self._stats_epoch:
            return

        if self.scale_mode == "hybrid":
            # Reapply hybrid normalization attribute by attribute.
            per_attr_raw = self._base_attrs[key]           # dict[attr_type] -> (numNodes,1)
            per_attr_norm = {}
            for attr_type, a_raw_2d in per_attr_raw.items():
                a_raw_1d = a_raw_2d.view(-1)              # (numNodes,)
                a_norm   = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                per_attr_norm[attr_type] = a_norm
            self._norm_attrs[key] = per_attr_norm
            self._norm_epoch_by_key[key] = self._stats_epoch
        else:
            # Non-hybrid modes are shared across CFP implementations.
            maybe_refresh_norm_for_key(
                key,
                self._base_attrs,
                self._norm_attrs,
                self._all_attr_types,
                self._ds_stats,
                self.scale_mode,
                self.eps,
                self._norm_epoch_by_key,
                self._stats_epoch
            )

    def freeze_ds_stats(self):
        """Stop collecting dataset statistics for future samples."""
        self._stats_frozen = True

    def unfreeze_ds_stats(self):
        """Resume collecting dataset statistics for future samples."""
        self._stats_frozen = False

    def save_stats(self, path: str):
        """Save normalization statistics and scale mode for reproducibility."""
        payload = make_stats_payload(self._ds_stats, self.scale_mode)
        torch.save(payload, path)
        print(f"[ConnectedLinearUnit] stats saved to {path}")

    def load_stats(self, path: str, refresh_cache: bool = True, *, trusted_legacy_format: bool = False):
        """Load normalization statistics and optionally refresh cached attrs."""
        payload = load_stats_payload(path, self.device, trusted_legacy_format=trusted_legacy_format)
        self._ds_stats = payload.get("ds_stats", {})
        # Invalidate normalized values derived from older statistics.
        self._stats_epoch += 1
        if refresh_cache:
            self.refresh_cached_normalization()

    # ---------- tree and attribute construction ----------
    def _ensure_tree_and_attr(self, key: str, img_t: torch.Tensor):
        """Populate cached tree, Jacobian, residues, and attributes for ``key``."""
        # Accept a PyTorch tensor and convert it internally to np.ndarray.
        if key in self._trees:
            return

        img_np = self._to_numpy_u8(img_t.detach())
        tree, jacobian, residues = self._build_tree_jacobian_and_residues(img_np)
        self._trees[key] = tree
        self._jacobians[key] = jacobian
        self._residues[key] = residues

        per_attr_raw, per_attr_norm = {}, {}
        for attr_type in self._all_attr_types:
            attr_np  = morphology.compute_attributes(tree, [attr_type])[1]
            a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
            self._update_ds_stats(attr_type, a_raw_1d)
            a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
            per_attr_raw[attr_type]  = a_raw_1d.unsqueeze(1)
            per_attr_norm[attr_type] = a_norm

        self._base_attrs[key] = per_attr_raw
        self._norm_attrs[key] = per_attr_norm
        self._norm_epoch_by_key[key] = self._stats_epoch


    # ---------- inspection ----------
    def inspect_training_sample(self, img: torch.Tensor, channel: int = 0, idx: int | None = None, build_if_missing: bool = True):
        """Return cached or on-the-fly inspection data for one sample.

        When ``idx`` is provided, the method inspects cached data under
        ``f"{idx}_{channel}"``. Without an index, the tree, dense Jacobian, and
        attributes are computed on the fly and are not persisted.
        """
        # Normalize image layout to (C, H, W).
        if img.dim() == 2:
            imgCHW = img.unsqueeze(0)
        elif img.dim() == 3:
            imgCHW = img
        else:
            raise ValueError(f"img must be (H, W) or (C, H, W); got {tuple(img.shape)}")

        C, H, W = imgCHW.shape
        if C != self.in_channels:
            if C != 1:
                raise AssertionError(f"in_channels={self.in_channels}, input C={C}")

        c = channel if C > 1 else 0

        if idx is not None:
            key = f"{idx}_{c}"
            use_cache = True
        else:
            use_cache = False

        if use_cache:
            if (key not in self._trees) and build_if_missing:
                self._ensure_tree_and_attr(key, imgCHW[c])
            elif key not in self._trees:
                raise KeyError("Tree/attributes not found in cache. Use build_if_missing=True.")
            self._maybe_refresh_norm_for_key(key)
            tree = self._trees[key]
            base_attrs_by_group = {}
            norm_attrs_by_group = {}
            weights_by_group    = {}
            bias_by_group       = {}
            for group in self.group_defs:
                gname = self._group_name(group)
                cols_raw  = [self._base_attrs[key][attr_type].view(-1, 1) for attr_type in group]
                cols_norm = [self._norm_attrs[key][attr_type].view(-1, 1) for attr_type in group]
                A_raw  = torch.cat(cols_raw,  dim=1)
                A_norm = torch.cat(cols_norm, dim=1)
                base_attrs_by_group[gname] = A_raw
                norm_attrs_by_group[gname] = A_norm
                weights_by_group[gname]    = self._weights[gname]
                bias_by_group[gname]       = self._biases[gname]
        else:
            print("[inspect_training_sample] Running without cache; computing tree and attributes directly.")
            img_np = self._to_numpy_u8(imgCHW[c].detach())
            tree, jacobian, residues = self._build_tree_jacobian_and_residues(img_np)
            base_attrs_by_group = {}
            norm_attrs_by_group = {}
            weights_by_group    = {}
            bias_by_group       = {}
            for group in self.group_defs:
                gname = self._group_name(group)
                cols_raw, cols_norm = [], []
                for attr_type in group:
                    attr_np = morphology.compute_attributes(tree, [attr_type])[1]
                    a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
                    a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                    cols_raw.append(a_raw_1d.unsqueeze(1))
                    cols_norm.append(a_norm.view(-1, 1))
                A_raw = torch.cat(cols_raw, dim=1)
                A_norm = torch.cat(cols_norm, dim=1)
                base_attrs_by_group[gname] = A_raw
                norm_attrs_by_group[gname] = A_norm
                weights_by_group[gname]    = self._weights[gname]
                bias_by_group[gname]       = self._biases[gname]

        return {
            "tree": tree,
            "base_attrs_by_group": base_attrs_by_group,
            "norm_attrs_by_group": norm_attrs_by_group,
            "weights_by_group": weights_by_group,
            "bias_by_group": bias_by_group,
        }

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CFP to a batch and return ``(B, C * groups, H, W)`` output."""
        # Match the input conventions used by the CFP layers.
        if isinstance(x, tuple) and len(x) == 2:
            x, idx = x
            use_cache = True
        elif isinstance(x, list) and len(x) == 2 and isinstance(x[1], torch.Tensor) and x[1].dim() == 1:
            x, idx = x[0], x[1]
            use_cache = True
        else:
            if isinstance(x, list):
                x = torch.stack(x, dim=0)
            idx = torch.arange(x.size(0), device=x.device)
            use_cache = False

        assert x.dim() == 4, f"expected (B, C, H, W), got {tuple(x.shape)}"
        B, C, H, W = x.shape
        assert C == self.in_channels, f"in_channels={self.in_channels}, input C={C}"

        out = torch.empty((B, self.out_channels, H, W), dtype=torch.float32, device=self.device)

        for b in range(B):
            for c in range(C):
                if use_cache:
                    # Use idx as part of the persistent per-channel cache key.
                    key = f"{int(idx[b])}_{c}"
                    self._ensure_tree_and_attr(key, x[b, c])
                    tree = self._trees[key]
                    self._maybe_refresh_norm_for_key(key)
                    for g, group in enumerate(self.group_defs):
                        gname = self._group_name(group)
                        # Build A_norm by stacking one normalized column per group attribute.
                        cols = [self._norm_attrs[key][attr_type].view(-1, 1) for attr_type in group]
                        A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                        y_ch = ConnectedFilterPreprocessingExplicitJacobianFunction.apply(
                            self._jacobians[key],
                            self._residues[key],
                            tree.numRows,
                            tree.numCols,
                            A_norm,
                            self._weights[gname],
                            self._biases[gname],
                            self.beta_f,
                            self.clamp_logits
                        )
                        if self.top_hat:
                            x_bc = x[b, c].to(dtype=torch.float32, device=self.device)
                            tt = self.tree_type
                            if tt == "max-tree":
                                y_out = x_bc - y_ch
                            elif tt == "min-tree":
                                y_out = y_ch - x_bc
                            else:
                                y_out = torch.abs(y_ch - x_bc)
                        else:
                            y_out = y_ch
                        out[b, c * self.num_groups + g].copy_(y_out, non_blocking=True)
                else:
                    # No persistent key was provided; build tree and attributes directly.
                    #print("[ConnectedFilterPreprocessingLayerWithExplicitJacobian] Cache is not used because no index was provided.")
                    img_np = self._to_numpy_u8(x[b, c].detach())
                    tree, jacobian, residues = self._build_tree_jacobian_and_residues(img_np)
                    # Compute and normalize attributes directly, without storing them.
                    per_attr_norm = {}
                    for attr_type in self._all_attr_types:
                        attr_np = morphology.compute_attributes(tree, [attr_type])[1]
                        a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
                        a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                        per_attr_norm[attr_type] = a_norm
                    for g, group in enumerate(self.group_defs):
                        gname = self._group_name(group)
                        cols = [per_attr_norm[attr_type].view(-1, 1) for attr_type in group]
                        A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                        y_ch = ConnectedFilterPreprocessingExplicitJacobianFunction.apply(
                            jacobian,
                            residues,
                            tree.numRows,
                            tree.numCols,
                            A_norm,
                            self._weights[gname],
                            self._biases[gname],
                            self.beta_f,
                            self.clamp_logits
                        )
                        if self.top_hat:
                            x_bc = x[b, c].to(dtype=torch.float32, device=self.device)
                            tt = self.tree_type
                            if tt == "max-tree":
                                y_out = x_bc - y_ch
                            elif tt == "min-tree":
                                y_out = y_ch - x_bc
                            else:
                                y_out = torch.abs(y_ch - x_bc)
                        else:
                            y_out = y_ch
                        out[b, c * self.num_groups + g].copy_(y_out, non_blocking=True)

        return out

    # ---------- prediction / inference ----------
    def predict(self, x: torch.Tensor, beta_f: float = 1000.0) -> torch.Tensor:
        """Run inference with a caller-provided forward sigmoid gain."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            # Same input handling as forward.
            if isinstance(x, tuple) and len(x) == 2:
                x, idx = x
                use_cache = True
            elif isinstance(x, list) and len(x) == 2 and isinstance(x[1], torch.Tensor) and x[1].dim() == 1:
                x, idx = x[0], x[1]
                use_cache = True
            else:
                if isinstance(x, list):
                    x = torch.stack(x, dim=0)
                idx = torch.arange(x.size(0), device=x.device)
                use_cache = False
            B, C, H, W = x.shape
            out = torch.empty((B, self.out_channels, H, W), dtype=torch.float32, device=self.device)
            for b in range(B):
                for c in range(C):
                    if use_cache:
                        key = f"{int(idx[b])}_{c}"
                        self._ensure_tree_and_attr(key, x[b, c])
                        tree = self._trees[key]
                        self._maybe_refresh_norm_for_key(key)
                        for g, group in enumerate(self.group_defs):
                            gname = self._group_name(group)
                            cols = [self._norm_attrs[key][attr_type].view(-1, 1) for attr_type in group]
                            A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                            y_ch = ConnectedFilterPreprocessingExplicitJacobianFunction.apply(
                                self._jacobians[key],
                                self._residues[key],
                                tree.numRows,
                                tree.numCols,
                                A_norm,
                                self._weights[gname],
                                self._biases[gname],
                                beta_f,  # caller-provided beta_f
                                self.clamp_logits
                            )
                            if self.top_hat:
                                x_bc = x[b, c].to(dtype=torch.float32, device=self.device)
                                tt = self.tree_type
                                if tt == "max-tree":
                                    y_out = x_bc - y_ch
                                elif tt == "min-tree":
                                    y_out = y_ch - x_bc
                                else:
                                    y_out = torch.abs(y_ch - x_bc)
                            else:
                                y_out = y_ch
                            out[b, c * self.num_groups + g].copy_(y_out, non_blocking=True)
                    else:
                        #print("[ConnectedFilterPreprocessingLayerWithExplicitJacobian] Cache is not used during prediction because no index was provided.")
                        img_np = self._to_numpy_u8(x[b, c].detach())
                        tree, jacobian, residues = self._build_tree_jacobian_and_residues(img_np)
                        per_attr_norm = {}
                        for attr_type in self._all_attr_types:
                            attr_np = morphology.compute_attributes(tree, [attr_type])[1]
                            a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
                            a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                            per_attr_norm[attr_type] = a_norm
                        for g, group in enumerate(self.group_defs):
                            gname = self._group_name(group)
                            cols = [per_attr_norm[attr_type].view(-1, 1) for attr_type in group]
                            A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                            y_ch = ConnectedFilterPreprocessingExplicitJacobianFunction.apply(
                                jacobian,
                                residues,
                                tree.numRows,
                                tree.numCols,
                                A_norm,
                                self._weights[gname],
                                self._biases[gname],
                                beta_f,  # caller-provided beta_f
                                self.clamp_logits
                            )
                            if self.top_hat:
                                x_bc = x[b, c].to(dtype=torch.float32, device=self.device)
                                tt = self.tree_type
                                if tt == "max-tree":
                                    y_out = x_bc - y_ch
                                elif tt == "min-tree":
                                    y_out = y_ch - x_bc
                                else:
                                    y_out = torch.abs(y_ch - x_bc)
                            else:
                                y_out = y_ch
                            out[b, c * self.num_groups + g].copy_(y_out, non_blocking=True)
        if was_training:
            self.train()
        else:
            self.eval()
        return out

    # ---------- save / load ----------
    def save_params(self, path: str):
        """Save all group weights and biases."""
        payload = {
            "weights": { name: p.detach().cpu() for name, p in self._weights.items() },
            "biases":  { name: p.detach().cpu() for name, p in self._biases.items()  },
            "scale_mode": self.scale_mode,
        }
        torch.save(payload, path)
        print(f"[ConnectedLinearUnit] weights and biases saved to {path}")

    def get_params(self):
        """Return CPU clones of group weights and biases."""
        return (
            { name: p.detach().cpu().clone() for name, p in self._weights.items() },
            { name: p.detach().cpu().clone() for name, p in self._biases.items()  },
        )


    # ---------- cached-normalization utilities ----------
    def refresh_cached_normalization(self):
        """Recompute normalized attributes for every cached sample."""
        for key, per_attr_raw in self._base_attrs.items():
            per_attr_norm = {}
            for attr_type, a_raw_2d in per_attr_raw.items():
                a_raw_1d = a_raw_2d.view(-1)
                a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                per_attr_norm[attr_type] = a_norm
            self._norm_attrs[key] = per_attr_norm
            self._norm_epoch_by_key[key] = self._stats_epoch

    # ---------- initialization helpers ----------
    @staticmethod
    def _logit(p: float) -> float:
        """Return a numerically clipped logit for probability ``p``."""
        p = max(min(float(p), 1.0 - 1e-6), 1e-6)
        return math.log(p / (1.0 - p))

    @torch.no_grad()
    def init_identity_with_bias(self, p0: float = 0.995):
        """Initialize near identity by using only a positive bias.

        For each group, weights are set to zero and bias is set to
        ``logit(p0) / beta_f``.
        """
        L = self._logit(p0) / float(self.beta_f)
        for group in self.group_defs:
            gname = self._group_name(group)
            self._weights[gname].zero_()
            self._biases[gname].fill_(L)

    @torch.no_grad()
    def init_identity_bias_zero(self, p0: float = 0.99):
        """Initialize near identity with zero bias under hybrid normalization.

        This assumes normalized attributes live in ``[a, 1]`` where
        ``a = hybrid_floor_a``. Each group receives constant weights
        ``c = logit(p0) / (beta_f * K * a)`` and zero bias.
        """
        if self.scale_mode != "hybrid":
            print("[init_identity_bias_zero] Warning: this initializer assumes scale_mode == 'hybrid'.")
        a = max(min(self.hybrid_floor_a, 1.0), 1e-6)
        L = self._logit(p0) / float(self.beta_f)
        for group in self.group_defs:
            gname = self._group_name(group)
            K = len(group)
            c = L / (K * a)
            self._weights[gname].fill_(c)
            self._biases[gname].zero_()

    def build_dataloader_cached(self, dataloader):
        """Precompute trees, dense Jacobians, and attributes for a DataLoader.

        The returned DataLoader wraps the original dataset and emits
        ``((x, idx), y)`` batches, where ``idx`` is the original dataset index.
        """
        from torch.utils.data import DataLoader

        dataset_wrapped = IndexedDatasetWrapper(dataloader.dataset)
        new_loader = DataLoader(
            dataset_wrapped,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=False,
            collate_fn=dataloader.collate_fn,
            persistent_workers=getattr(dataloader, "persistent_workers", False),
        )

        print(f"[ConnectedFilterPreprocessingLayerWithExplicitJacobian] Preprocessing dataset using mode '{self.scale_mode}'...")
        self._stats_frozen = False
        total_batches = len(new_loader)

        with torch.no_grad():
            for batch_i, ((x, idx), y) in enumerate(new_loader):
                B, C, H, W = x.shape
                for b in range(B):
                    for c in range(C):
                        key = f"{int(idx[b])}_{c}"
                        self._ensure_tree_and_attr(key, x[b, c])
                if (batch_i + 1) % 10 == 0 or batch_i == total_batches - 1:
                    print(f"  [{batch_i+1}/{total_batches}] batches processed.")

        self.freeze_ds_stats()
        self.refresh_cached_normalization()
        print(f"[ConnectedFilterPreprocessingLayerWithExplicitJacobian] Full and normalized cache with '{self.scale_mode}'.")
        return new_loader


CFPLayerWithExplicitJacobian = ConnectedFilterPreprocessingLayerWithExplicitJacobian
CFPExplicitJacobianFunction = ConnectedFilterPreprocessingExplicitJacobianFunction

__all__ = [
    'ConnectedFilterPreprocessingLayerWithExplicitJacobian',
    'ConnectedFilterPreprocessingExplicitJacobianFunction',
    'CFPLayerWithExplicitJacobian',
    'CFPExplicitJacobianFunction',
]
