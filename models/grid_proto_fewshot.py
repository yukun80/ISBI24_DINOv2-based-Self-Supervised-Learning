"""Few-shot segmentation with AM²P prototypes."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .am2p import AM2P
from util.consts import DEFAULT_FEATURE_SIZE
from util.lora import inject_trainable_lora

# from util.utils import load_config_from_url, plot_dinov2_fts
import math


class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """

    def __init__(self, image_size, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.image_size = image_size
        self.pretrained_path = pretrained_path
        self.config = cfg or {"align": False, "debug": False}
        self.get_encoder()
        self.get_cls()
        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path), strict=True)
            print(f"###### Pre-trained model f{self.pretrained_path} has been loaded ######")

    def get_encoder(self):
        self.config["feature_hw"] = [DEFAULT_FEATURE_SIZE, DEFAULT_FEATURE_SIZE]  # default feature map size
        if self.config["which_model"] == "dinov2_l14":
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            self.config["feature_hw"] = [
                max(self.image_size // 14, DEFAULT_FEATURE_SIZE),
                max(self.image_size // 14, DEFAULT_FEATURE_SIZE),
            ]
        elif self.config["which_model"] == "dinov2_l14_reg":
            try:
                self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
            except RuntimeError as e:
                self.encoder = torch.hub.load("facebookresearch/dino", "dinov2_vitl14_reg", force_reload=True)
            self.config["feature_hw"] = [
                max(self.image_size // 14, DEFAULT_FEATURE_SIZE),
                max(self.image_size // 14, DEFAULT_FEATURE_SIZE),
            ]
        elif self.config["which_model"] == "dinov2_b14":
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.config["feature_hw"] = [
                max(self.image_size // 14, DEFAULT_FEATURE_SIZE),
                max(self.image_size // 14, DEFAULT_FEATURE_SIZE),
            ]
        elif self.config["which_model"] == "dinov2_s14":
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.config["feature_hw"] = [
                max(self.image_size // 14, DEFAULT_FEATURE_SIZE),
                max(self.image_size // 14, DEFAULT_FEATURE_SIZE),
            ]
        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.config["lora"] > 0:
            self.encoder.requires_grad_(False)
            print(f'Injecting LoRA with rank:{self.config["lora"]}')
            encoder_lora_params = inject_trainable_lora(self.encoder, r=self.config["lora"])

    def get_features(self, imgs_concat):
        if "dino" in self.config["which_model"]:
            # resize imgs_concat to the closest size that is divisble by 14
            imgs_concat = F.interpolate(
                imgs_concat, size=(self.image_size // 14 * 14, self.image_size // 14 * 14), mode="bilinear"
            )
            dino_fts = self.encoder.forward_features(imgs_concat)
            img_fts = dino_fts["x_norm_patchtokens"]  # B, HW, C
            img_fts = img_fts.permute(0, 2, 1)  # B, C, HW
            C, HW = img_fts.shape[-2:]
            img_fts = img_fts.view(-1, C, int(HW**0.5), int(HW**0.5))  # B, C, H, W
            if HW < DEFAULT_FEATURE_SIZE**2:
                img_fts = F.interpolate(
                    img_fts, size=(DEFAULT_FEATURE_SIZE, DEFAULT_FEATURE_SIZE), mode="bilinear"
                )  # this is if h,w < (32,32)
        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        return img_fts

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        if self.config["cls_name"] == "grid_proto":
            am2p_cfg = self.config.get("am2p", {})
            self.cls_unit = AM2P(**am2p_cfg)
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    def forward_resolutions(
        self, resolutions, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False, supp_fts=None
    ):
        predictions = []
        for res in resolutions:
            supp_imgs_resized = (
                [[F.interpolate(supp_img[0], size=(res, res), mode="bilinear") for supp_img in supp_imgs]]
                if supp_imgs[0][0].shape[-1] != res
                else supp_imgs
            )
            fore_mask_resized = (
                [
                    [
                        F.interpolate(fore_mask_way[0].unsqueeze(0), size=(res, res), mode="bilinear")[0]
                        for fore_mask_way in fore_mask
                    ]
                ]
                if fore_mask[0][0].shape[-1] != res
                else fore_mask
            )
            back_mask_resized = (
                [
                    [
                        F.interpolate(back_mask_way[0].unsqueeze(0), size=(res, res), mode="bilinear")[0]
                        for back_mask_way in back_mask
                    ]
                ]
                if back_mask[0][0].shape[-1] != res
                else back_mask
            )
            qry_imgs_resized = (
                [F.interpolate(qry_img, size=(res, res), mode="bilinear") for qry_img in qry_imgs]
                if qry_imgs[0][0].shape[-1] != res
                else qry_imgs
            )

            pred = self.forward(
                supp_imgs_resized,
                fore_mask_resized,
                back_mask_resized,
                qry_imgs_resized,
                isval,
                val_wsize,
                show_viz,
                supp_fts,
            )[0]
            predictions.append(pred)

    def resize_inputs_to_image_size(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        supp_imgs = [
            [
                F.interpolate(supp_img, size=(self.image_size, self.image_size), mode="bilinear")
                for supp_img in supp_imgs_way
            ]
            for supp_imgs_way in supp_imgs
        ]
        fore_mask = (
            [
                [
                    F.interpolate(
                        fore_mask_way[0].unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear"
                    )[0]
                    for fore_mask_way in fore_mask
                ]
            ]
            if fore_mask[0][0].shape[-1] != self.image_size
            else fore_mask
        )
        back_mask = (
            [
                [
                    F.interpolate(
                        back_mask_way[0].unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear"
                    )[0]
                    for back_mask_way in back_mask
                ]
            ]
            if back_mask[0][0].shape[-1] != self.image_size
            else back_mask
        )
        qry_imgs = (
            [F.interpolate(qry_img, size=(self.image_size, self.image_size), mode="bilinear") for qry_img in qry_imgs]
            if qry_imgs[0][0].shape[-1] != self.image_size
            else qry_imgs
        )
        return supp_imgs, fore_mask, back_mask, qry_imgs

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False, supp_fts=None):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W]
            fore_mask: foreground masks for support images
                way x shot x [B x H x W]
            back_mask: unused (kept for API compatibility)
            qry_imgs: query images
                N x [B x 3 x H x W]
        Returns:
            logits: [N*B, 2, H, W]
            align_loss: scalar tensor
            debug_info: dict with prototype metadata
        """
        del back_mask, isval, val_wsize, show_viz  # retained for API compatibility

        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "AM²P currently supports single-way episodes."

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)], dim=0)

        img_fts = self.get_features(imgs_concat)
        if img_fts.dim() == 5:
            raise NotImplementedError("3D features are not yet supported by AM²P.")
        fts_size = img_fts.shape[-2:]

        if supp_fts is None:
            supp_fts = img_fts[: n_ways * n_shots * sup_bsize].view(n_ways, n_shots, sup_bsize, -1, *fts_size)
            qry_fts = img_fts[n_ways * n_shots * sup_bsize :].view(n_queries, qry_bsize, -1, *fts_size)
        else:
            qry_fts = img_fts.view(n_queries, qry_bsize, -1, *fts_size)

        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)

        res_fg_msk = torch.stack(
            [F.interpolate(fore_mask_w, size=fts_size, mode="nearest") for fore_mask_w in fore_mask],
            dim=0,
        )

        support_feats = supp_fts.view(n_ways, n_shots * sup_bsize, -1, *fts_size)[0]
        support_masks = res_fg_msk.view(n_ways, n_shots * sup_bsize, 1, *fts_size)[0]
        query_feats = qry_fts.view(n_queries * qry_bsize, -1, *fts_size)

        proto_set = self.cls_unit.build_prototypes(support_feats, support_masks)
        logits, am2p_debug = self.cls_unit(support_feats, support_masks, query_feats, proto_cache=proto_set)

        logits = logits.view(n_queries, qry_bsize, 2, *fts_size).view(-1, 2, *fts_size)
        logits = F.interpolate(logits, size=img_size, mode="bilinear", align_corners=False)

        align_loss = (
            self._alignment_loss(proto_set, support_feats, support_masks)
            if self.config.get("align", False) and self.training
            else logits.new_tensor(0.0)
        )

        debug_info = {
            "am2p": am2p_debug,
            "num_support_shots": support_feats.shape[0],
        }

        return logits, align_loss, debug_info

    def _alignment_loss(self, proto_set, support_feats: torch.Tensor, support_masks: torch.Tensor) -> torch.Tensor:
        """
        Encourage prototypes to reproduce the annotated support masks.
        """
        logits, _ = self.cls_unit(support_feats, support_masks, support_feats, proto_cache=proto_set)
        targets = (support_masks.squeeze(1) > 0.5).long()
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def dino_cls_loss(self, teacher_cls_tokens, student_cls_tokens):
        cls_loss_weight = 0.1
        student_temp = 1
        teacher_cls_tokens = self.sinkhorn_knopp_teacher(teacher_cls_tokens)
        lsm = F.log_softmax(student_cls_tokens / student_temp, dim=-1)
        cls_loss = torch.sum(teacher_cls_tokens * lsm, dim=-1)

        return -cls_loss.mean() * cls_loss_weight

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp=1, n_iterations=3):
        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Q is K-by-B for consistency with notations from our paper
        Q = torch.exp(teacher_output / teacher_temp).t()
        # B = Q.shape[1] * world_size # number of samples to assign
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def dino_patch_loss(self, features, masked_features, masks):
        # for both supp and query features perform the patch wise loss
        loss = 0.0
        weight = 0.1
        B = features.shape[0]
        for f, mf, mask in zip(features, masked_features, masks):
            # TODO sinkhorn knopp center features
            f = f[mask]
            f = self.sinkhorn_knopp_teacher(f)
            mf = mf[mask]
            loss += torch.sum(f * F.log_softmax(mf / 1, dim=-1), dim=-1) / mask.sum()

        return -loss.sum() * weight / B
