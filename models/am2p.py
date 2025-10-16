"""
AM²P: Area-balanced Multi-scale Prototype Pooling.

This module replaces the adaptive local prototype pooling used in ALPNet with
an area-aware, multi-scale prototype builder that is friendlier to the highly
variable object morphology in remote-sensing imagery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PrototypeSet:
    """Container holding local/global prototypes and debugging metadata."""

    local_prototypes: torch.Tensor  # [M, C] or empty
    local_weights: torch.Tensor  # [M] or empty
    global_proto: torch.Tensor  # [C]
    anchor_meta: List[Dict[str, torch.Tensor]]
    fg_area: float


class AM2P(nn.Module):
    """
    Area-balanced Multi-scale Prototype Pooling.

    Args:
        radii: Window radii (in pixels) used to gather local statistics.
        alpha: Controls anchor quota growth per connected component.
        nmax_comp: Upper bound on anchors allocated to a single component.
        mmax_total: Maximum number of local prototypes kept after pruning.
        theta_min: Minimum number of foreground pixels in a window to accept it.
        tau_area: Components with area below this bypass local pooling.
        beta: Weight assigned to the global prototype when mixing scores.
        temp: Temperature applied to the final logits.
        epsilon: Numerical stability constant.
    """

    def __init__(
        self,
        radii: Iterable[int] = (4, 8, 16),
        alpha: float = 1.2,
        nmax_comp: int = 8,
        mmax_total: int = 64,
        theta_min: int = 8,
        tau_area: int = 9,
        beta: float = 0.3,
        temp: float = 0.07,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "_radii", torch.tensor(list(radii), dtype=torch.long), persistent=False
        )
        self.alpha = float(alpha)
        self.nmax_comp = int(nmax_comp)
        self.mmax_total = int(mmax_total)
        self.theta_min = int(theta_min)
        self.tau_area = int(tau_area)
        self.beta = float(beta)
        self.temp = float(temp)
        self.epsilon = float(epsilon)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def build_prototypes(
        self, support_feats: torch.Tensor, support_masks: torch.Tensor
    ) -> PrototypeSet:
        """
        Build prototypes from support features and masks.

        Args:
            support_feats: [S, C, H, W] support embeddings (S=shots*batch).
            support_masks: [S, 1, H, W] binary masks aligned to support_feats.
        """
        device = support_feats.device
        dtype = support_feats.dtype
        support_masks = support_masks.to(support_feats.dtype)

        fg_pixels = support_masks.sum(dtype=torch.float64)
        if fg_pixels < self.epsilon:
            global_proto = support_feats.mean(dim=(0, 2, 3))
        else:
            fg_sum = (support_feats * support_masks).sum(dim=(0, 2, 3))
            global_proto = fg_sum / fg_pixels

        local_protos: List[torch.Tensor] = []
        local_weights: List[torch.Tensor] = []
        anchor_meta: List[Dict[str, torch.Tensor]] = []

        for sample_idx, (feat, mask) in enumerate(zip(support_feats, support_masks), 0):
            mask_bool = mask.squeeze(0) > 0.5
            if mask_bool.sum() == 0:
                continue
            labels, num_components = self._connected_components(mask_bool)
            if num_components == 0:
                continue
            stats_per_radius = self._precompute_local_statistics(feat, mask_bool)
            for comp_id in range(1, num_components + 1):
                comp_coords = torch.nonzero(labels == comp_id, as_tuple=False)
                if comp_coords.numel() == 0:
                    continue
                anchors = self._sample_anchors(comp_coords)
                if not anchors:
                    continue
                comp_area = comp_coords.shape[0]
                for anchor in anchors:
                    proto, weight, radius_idx = self._select_best_radius(
                        anchor,
                        comp_area,
                        stats_per_radius,
                        device=device,
                    )
                    if proto is None:
                        continue
                    local_protos.append(proto.unsqueeze(0))
                    local_weights.append(weight.unsqueeze(0))
                    anchor_meta.append(
                        {
                            "sample_index": torch.tensor(
                                sample_idx, device=device, dtype=torch.long
                            ),
                            "component_id": torch.tensor(
                                comp_id, device=device, dtype=torch.long
                            ),
                            "position": anchor.to(device),
                            "radius_index": torch.tensor(
                                radius_idx, device=device, dtype=torch.long
                            ),
                        }
                    )

        if local_protos:
            local_stack = torch.cat(local_protos, dim=0)
            weight_stack = torch.cat(local_weights, dim=0)
            local_stack, weight_stack, anchor_meta = self._prune_prototypes(
                local_stack, weight_stack, anchor_meta
            )
            weight_stack = weight_stack / (weight_stack.sum() + self.epsilon)
        else:
            local_stack = torch.empty((0, support_feats.shape[1]), device=device, dtype=dtype)
            weight_stack = torch.empty((0,), device=device, dtype=dtype)

        return PrototypeSet(
            local_prototypes=local_stack,
            local_weights=weight_stack,
            global_proto=global_proto,
            anchor_meta=anchor_meta,
            fg_area=float(fg_pixels.item()),
        )

    def forward(
        self,
        support_feats: torch.Tensor,
        support_masks: torch.Tensor,
        query_feats: torch.Tensor,
        proto_cache: Optional[PrototypeSet] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute logits for query features using AM²P prototypes.

        Args:
            support_feats: [S, C, H, W]
            support_masks: [S, 1, H, W]
            query_feats: [B, C, H, W]
            proto_cache: Optional pre-computed prototypes.

        Returns:
            logits: [B, 2, H, W] background / foreground logits.
            debug: dictionary of diagnostic tensors.
        """
        if proto_cache is None:
            proto_cache = self.build_prototypes(support_feats, support_masks)

        logits = self._compute_logits(query_feats, proto_cache)
        debug_tensors = {
            "num_local": torch.tensor(
                proto_cache.local_prototypes.shape[0],
                device=query_feats.device,
                dtype=torch.long,
            ),
            "fg_area": torch.tensor(proto_cache.fg_area, device=query_feats.device),
        }
        return logits, debug_tensors

    # ------------------------------------------------------------------ #
    # Prototype helpers                                                  #
    # ------------------------------------------------------------------ #
    def _compute_logits(
        self, query_feats: torch.Tensor, proto_set: PrototypeSet
    ) -> torch.Tensor:
        q_norm = F.normalize(query_feats, dim=1)
        global_proto = proto_set.global_proto.to(query_feats.device)
        global_proto = F.normalize(global_proto.unsqueeze(0), dim=1).squeeze(0)

        s_global = torch.einsum("bchw,c->bhw", q_norm, global_proto)

        if proto_set.local_prototypes.numel() > 0:
            local_protos = proto_set.local_prototypes.to(query_feats.device)
            local_protos = F.normalize(local_protos, dim=1)
            sims = torch.einsum("bchw,mc->bmhw", q_norm, local_protos)
            weights = proto_set.local_weights.to(query_feats.device)
            weights = weights / (weights.sum() + self.epsilon)
            local_score = torch.sum(
                weights.view(1, -1, 1, 1) * sims, dim=1
            )
        else:
            local_score = torch.zeros_like(s_global)

        beta = self.beta
        residual = max(0.0, 1.0 - beta)
        s_fg = beta * s_global + residual * local_score

        logits = torch.stack(
            (-s_fg / self.temp, s_fg / self.temp), dim=1
        )
        return logits

    @torch.no_grad()
    def _connected_components(
        self, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        if mask.dim() != 2:
            raise ValueError("Expected a 2D mask for connected components.")
        mask_np = mask.detach().cpu().to(torch.uint8).numpy()
        h, w = mask_np.shape
        labels = torch.zeros((h, w), dtype=torch.int32)
        current = 0
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        stack: List[Tuple[int, int]] = []

        for row in range(h):
            for col in range(w):
                if mask_np[row, col] == 0 or labels[row, col] != 0:
                    continue
                current += 1
                stack.append((row, col))
                labels[row, col] = current
                while stack:
                    r, c = stack.pop()
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and mask_np[nr, nc] != 0
                            and labels[nr, nc] == 0
                        ):
                            labels[nr, nc] = current
                            stack.append((nr, nc))

        return labels.to(mask.device), current

    @torch.no_grad()
    def _precompute_local_statistics(
        self, feat: torch.Tensor, mask: torch.Tensor
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        device = feat.device
        feat = feat.unsqueeze(0)  # [1, C, H, W]
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        feat_sq = feat * feat

        stats: Dict[int, Dict[str, torch.Tensor]] = {}
        ones_scalar = torch.ones(1, 1, device=device, dtype=feat.dtype)

        for r_idx, radius in enumerate(self._radii.tolist()):
            ksize = int(2 * radius + 1)
            kernel = ones_scalar.expand(1, 1, ksize, ksize)
            conv_kwargs = {"padding": radius}

            mask_sum = F.conv2d(mask, kernel, **conv_kwargs)
            valid_count = F.conv2d(
                torch.ones_like(mask), kernel, **conv_kwargs
            )

            kernel_feat = kernel.expand(feat.shape[1], 1, ksize, ksize)
            feat_sum = F.conv2d(
                feat * mask, kernel_feat, groups=feat.shape[1], **conv_kwargs
            )
            feat_sq_sum = F.conv2d(
                feat_sq * mask,
                kernel_feat,
                groups=feat.shape[1],
                **conv_kwargs,
            )

            mean = feat_sum / (mask_sum + self.epsilon)
            variance = torch.clamp(
                feat_sq_sum / (mask_sum + self.epsilon) - mean * mean, min=0.0
            )
            rho = mask_sum / (valid_count + self.epsilon)

            stats[r_idx] = {
                "mean": mean.squeeze(0),  # [C, H, W]
                "var": variance.squeeze(0),
                "rho": rho.squeeze(0).squeeze(0),  # [H, W]
                "mask_sum": mask_sum.squeeze(0).squeeze(0),
            }

        return stats

    @torch.no_grad()
    def _sample_anchors(
        self, coords: torch.Tensor
    ) -> List[torch.Tensor]:
        area = coords.shape[0]
        quota = math.ceil(self.alpha * math.log1p(area))
        quota = max(1, min(quota, self.nmax_comp))

        if area <= quota:
            return [coord.to(torch.long) for coord in coords]

        coords = coords.to(torch.float32)
        centroid = coords.mean(dim=0, keepdim=True)
        dist_to_centroid = torch.sum((coords - centroid) ** 2, dim=1)
        first_idx = torch.argmin(dist_to_centroid).item()

        anchors: List[torch.Tensor] = []
        anchors.append(coords[first_idx])

        if quota == 1:
            return [anchors[0].round().to(torch.long)]

        dists = torch.full(
            (coords.shape[0],), float("inf"), device=coords.device
        )

        for _ in range(1, quota):
            last_anchor = anchors[-1]
            current_dist = torch.sum((coords - last_anchor) ** 2, dim=1)
            dists = torch.minimum(dists, current_dist)
            dists[first_idx] = -1  # already used
            candidate_idx = torch.argmax(dists).item()
            anchors.append(coords[candidate_idx])
            first_idx = candidate_idx

        return [anchor.round().to(torch.long) for anchor in anchors]

    @torch.no_grad()
    def _select_best_radius(
        self,
        anchor: torch.Tensor,
        comp_area: int,
        stats_per_radius: Dict[int, Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:
        row, col = anchor.tolist()
        best_score = None
        best_proto = None
        best_weight = None
        best_radius_idx = None

        fallback_candidate: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None
        fallback_score = None

        for r_idx, stats in stats_per_radius.items():
            mask_sum_at_anchor = stats["mask_sum"][row, col]
            rho = stats["rho"][row, col]
            mean_vec = stats["mean"][:, row, col]
            var_vec = stats["var"][:, row, col]
            if mask_sum_at_anchor >= self.theta_min:
                variance_score = torch.mean(var_vec)
                score = rho / (variance_score + self.epsilon)
                if best_score is None or score > best_score:
                    best_score = score
                    best_proto = mean_vec
                    best_weight = rho.clamp(min=self.epsilon)
                    best_radius_idx = r_idx

            if fallback_score is None or mask_sum_at_anchor > fallback_score:
                fallback_candidate = (
                    mean_vec,
                    rho.clamp(min=self.epsilon),
                    r_idx,
                )
                fallback_score = mask_sum_at_anchor

        if best_proto is None:
            if comp_area < self.tau_area and fallback_candidate is not None:
                proto, weight, r_idx = fallback_candidate
                return proto.to(device), weight.to(device), r_idx
            return None, None, None

        return best_proto.to(device), best_weight.to(device), best_radius_idx

    @torch.no_grad()
    def _prune_prototypes(
        self,
        protos: torch.Tensor,
        weights: torch.Tensor,
        anchor_meta: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        if protos.shape[0] <= self.mmax_total:
            return protos, weights, anchor_meta

        device = protos.device
        normed = F.normalize(protos, dim=1)
        num = normed.shape[0]

        comp_ids = torch.tensor(
            [meta["component_id"] for meta in anchor_meta],
            device=device,
            dtype=torch.long,
        )

        selected_indices: List[int] = []
        unique_comps = comp_ids.unique(sorted=True)
        for comp_id in unique_comps.tolist():
            comp_mask = comp_ids == comp_id
            comp_indices = torch.nonzero(comp_mask, as_tuple=False).squeeze(1)
            if comp_indices.numel() == 0:
                continue
            best_idx = comp_indices[torch.argmax(weights[comp_indices])]
            selected_indices.append(best_idx.item())

        if len(selected_indices) > self.mmax_total:
            selected_indices = sorted(
                selected_indices, key=lambda idx: weights[idx].item(), reverse=True
            )[: self.mmax_total]

        selected_set = set(selected_indices)
        min_dists = None
        if selected_indices:
            stack = normed[selected_indices]
            min_dists = torch.cdist(normed, stack, p=2).min(dim=1).values
        else:
            min_dists = torch.full((num,), float("inf"), device=device)

        available = [
            idx for idx in range(num) if idx not in selected_set
        ]
        while len(selected_indices) < self.mmax_total and available:
            candidates = torch.tensor(available, device=device, dtype=torch.long)
            scores = min_dists[candidates] * (weights[candidates] + self.epsilon)
            best_local_idx = torch.argmax(scores).item()
            best_idx = candidates[best_local_idx].item()
            selected_indices.append(best_idx)
            available.remove(best_idx)
            new_dists = torch.cdist(
                normed, normed[best_idx : best_idx + 1], p=2
            ).squeeze(1)
            min_dists = torch.minimum(min_dists, new_dists)

        selected_indices = sorted(selected_indices)
        protos = protos[selected_indices]
        weights = weights[selected_indices]
        anchor_meta = [anchor_meta[idx] for idx in selected_indices]
        return protos, weights, anchor_meta
