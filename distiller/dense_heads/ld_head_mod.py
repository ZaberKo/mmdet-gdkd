# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import Tensor
import torch.distributed as dist

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import ConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import multi_apply, unpack_gt_instances
from mmdet.models.dense_heads import GFLHead
from mmengine import MessageHub

from ..loss import DistillLoss


def reduce_sum(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


@MODELS.register_module()
class LDHeadMod(GFLHead):
    """Localization distillation Head. (Short description)

    It utilizes the learned bbox distributions to transfer the localization
    dark knowledge from teacher to student. Original paper: `Localization
    Distillation for Object Detection. <https://arxiv.org/abs/2102.12252>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss_ld (:obj:`ConfigDict` or dict): Config of Localization
            Distillation Loss (LD), T is the temperature for distillation.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_ld: ConfigType = dict(
                     type='KnowledgeDistillationKDLoss',
                     loss_weight=0.25,
                     T=10),
                 # "weighted_sum" or "weighted_sum_nonorm" or "mean"
                 loss_ld_avg_mode: str = "weighted_sum",
                 loss_cls_kd: ConfigType = None,  # default disabled cls_kd
                 **kwargs) -> dict:

        super().__init__(
            num_classes=num_classes, in_channels=in_channels, **kwargs)

        assert loss_ld_avg_mode in [
            "weighted_sum", 'weighted_sum_nonorm', "mean"],  "unsupported loss_ld_avg_mode"

        self.loss_ld_avg_mode = loss_ld_avg_mode
        self.loss_ld = MODELS.build(loss_ld)
        assert isinstance(
            self.loss_ld, DistillLoss
        ), "loss_ld must be subclass of DistillLoss"

        if loss_cls_kd is not None:
            self.loss_cls_kd = MODELS.build(loss_cls_kd)
            assert isinstance(
                self.loss_cls_kd, DistillLoss
            ), "loss_cls_kd must be subclass of DistillLoss"

    def loss(self, x: List[Tensor], out_teacher: Tuple[Tensor],
             batch_data_samples: SampleList) -> dict:
        """
        Args:
            x (list[Tensor]): Features from FPN.
            out_teacher (tuple[Tensor]): The output of teacher.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        cls_scores, bbox_preds = self(x)
        soft_cls_targets, soft_bbox_targets = out_teacher

        losses = self.loss_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            soft_cls_targets=soft_cls_targets,
            soft_bbox_targets=soft_bbox_targets,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        return losses

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_target: Tensor,
                            stride: Tuple[int], soft_cls_target: Tensor,
                            soft_bbox_target: Tensor, num_pos: int):
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            soft_targets (Tensor): Soft BBox regression targets.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[tuple, Tensor]: Loss components and weight targets.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        # [B, 4]
        anchors = anchors.reshape(-1, 4)
        # [B, num_classes-1] only include fg cls_score
        # B=B*H*W
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        soft_cls_target = soft_cls_target.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        # [B, (reg_max+1)*4]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, 4 * (self.reg_max + 1))
        soft_bbox_target = soft_bbox_target.permute(0, 2, 3, 1).reshape(
            -1, 4 * (self.reg_max + 1))

        bbox_target = bbox_target.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        # score is the normal IoU
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_target[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            # [B_p,2]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            # [B_p], maximum sigmoid pred score of each prediction
            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)

            # pred_corners, soft_corners: [B_p*4, reg_max+1]
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            pos_soft_targets = soft_bbox_target[pos_inds]
            soft_corners = pos_soft_targets.reshape(-1, self.reg_max + 1)

            # target_corners: [B_p*4]
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

            # ld loss
            if self.loss_ld_avg_mode in ["weighted_sum", 'weighted_sum_nonorm']:
                # delay divied by sum of weight_targets
                loss_ld = self.loss_ld(
                    pred_corners,
                    soft_corners,
                    target_corners,
                    weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                    avg_factor=4.0)
                ld_train_info = self.loss_ld.train_info
            elif self.loss_ld_avg_mode == "mean":
                loss_ld = self.loss_ld(
                    pred_corners,
                    soft_corners,
                    target_corners
                )/4.0
                ld_train_info = {k: v/4.0
                                 for k, v in self.loss_ld.train_info.items()}

            else:
                raise NotImplementedError

            # cls_kd loss
            if hasattr(self, "loss_cls_kd"):
                # for fg objects:
                loss_cls_kd = self.loss_cls_kd(
                    cls_score[pos_inds],
                    soft_cls_target[pos_inds],
                    # should be all 1 in default
                    weight=label_weights[pos_inds]
                )
            else:
                loss_cls_kd = cls_score.sum() * 0

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_ld = bbox_pred.sum() * 0
            loss_cls_kd = cls_score.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)
            ld_train_info = {}

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_pos)

        # TODO: add dIoU-based vlr support:

        return loss_cls, loss_bbox, loss_dfl, loss_cls_kd, loss_ld, ld_train_info, weight_targets.sum()

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            soft_cls_targets: List[Tensor],
            soft_bbox_targets: List[Tensor],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            soft_targets (list[Tensor]): Soft BBox regression targets.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_pos) = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        # here gfl use PseudoSampler, and avg_factor = #number of positive samples

        num_pos = reduce_sum(
            torch.tensor(num_pos, dtype=torch.float, device=device)).item()

        (losses_cls, losses_bbox, losses_dfl, losses_cls_kd,
         losses_ld, ld_train_infos, weight_sums) = multi_apply(
            self.loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides,
            soft_cls_targets,
            soft_bbox_targets,
            num_pos=num_pos)

        # Note: above losses_bbox|dfl|ld use weighted sum.
        # To make the weight across batch-level on DDP,
        # use reduced sum of weights: avg_factor
        weight_sum = sum(weight_sums)
        # align with new official GFLHead (#4978)
        weight_sum = reduce_sum(weight_sum).clamp(min=1).item()
        # Caution: every FPN layer must have same number of samples,
        # Otherwise using sum(avg_factor) will be wrong

        losses_bbox = [x / weight_sum for x in losses_bbox]
        losses_dfl = [x / weight_sum for x in losses_dfl]

        # Note: losses_ld is intended not being divided by avg_factor in ori impl
        # for better performance (by increased kd impact during traing).
        # Here we do not follow this impl for simplicity.
        if (self.loss_ld_avg_mode == "weighted_sum" and
                type(self.loss_ld).__name__ != "KnowledgeDistillationDISTLoss"):
            losses_ld = [x / weight_sum for x in losses_ld]

        # Average ld_train_infos:
        ld_train_info_summary = {}

        for k in ld_train_infos[0].keys():
            ld_train_info_summary[k] = sum(
                [ld_train_info[k] for ld_train_info in ld_train_infos
                 if len(ld_train_info) > 0]  # avoid empty ld_train_info
            )
            if (self.loss_ld_avg_mode == "weighted_sum" and
                    type(self.loss_ld).__name__ != "KnowledgeDistillationDISTLoss"):
                ld_train_info_summary[k] /= weight_sum

        # Then add them to message_hub
        self.record_ld_train_info(ld_train_info_summary)

        losses_dict = dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_ld=losses_ld,
            losses_cls_kd=losses_cls_kd
        )

        return losses_dict

    def record_ld_train_info(self, train_info):
        message_hub = MessageHub.get_current_instance()

        message_hub.update_scalars({
            f"train/ld/{k}": v
            for k, v in train_info.items()
        })
