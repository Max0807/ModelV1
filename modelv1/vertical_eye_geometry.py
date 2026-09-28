"""High-resolution probabilistic eye landmarks for vertical gaze geometry."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modelv1.data.eye_geometry_pseudo_labels import (
    EYE_PSEUDO_IRIS_CENTER_INDEX,
    EYE_PSEUDO_POINT_COUNT,
)
from modelv1.geometry import unpack_table_frame7


DEFAULT_LANDMARK_HEATMAP_SIZE = (28, 45)
OUTER_CANTHUS_INDEX = 0
INNER_CANTHUS_INDEX = 5
IRIS_RIM_SLICE = slice(EYE_PSEUDO_IRIS_CENTER_INDEX + 1, EYE_PSEUDO_POINT_COUNT)


class ProbabilisticEyeLandmarkHead(nn.Module):
    """Fuse ResNet layer2/layer3 and predict landmark distributions.

    The input maps remain substantially finer than the final gaze tokens.
    For a 56x90 eye crop, layer2/layer3 are 14x23 and 7x12. They are fused at
    layer2 resolution and decoded to the configured high-resolution heatmap.
    """

    def __init__(
        self,
        *,
        layer2_channels: int = 128,
        layer3_channels: int = 256,
        hidden_channels: int = 128,
        keypoint_count: int = EYE_PSEUDO_POINT_COUNT,
        output_size: tuple[int, int] = DEFAULT_LANDMARK_HEATMAP_SIZE,
        temperature: float = 1.0,
        predict_variance_correction: bool = True,
        max_abs_log_std_correction: float = 1.5,
        shape_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        # if min(
        #     layer2_channels,
        #     layer3_channels,
        #     hidden_channels,
        #     keypoint_count,
        #     shape_embedding_dim,
        # ) <= 0:
        #     raise ValueError("Eye-landmark head dimensions must be positive.")
        # if keypoint_count != EYE_PSEUDO_POINT_COUNT:
        #     raise ValueError(
        #         "keypoint_count must match the 15-point pseudo-label schema."
        #     )
        # if len(output_size) != 2 or min(output_size) <= 1:
        #     raise ValueError("output_size must be a two-item spatial size > 1.")
        # if temperature <= 0:
        #     raise ValueError("temperature must be positive.")
        # if max_abs_log_std_correction <= 0:
        #     raise ValueError("max_abs_log_std_correction must be positive.")
        self.keypoint_count = int(keypoint_count)
        self.output_size = tuple(int(value) for value in output_size)
        self.temperature = float(temperature)
        self.predict_variance_correction = bool(predict_variance_correction)
        self.max_abs_log_std_correction = float(max_abs_log_std_correction)

        self.layer2_projection = _conv_norm_activation(
            layer2_channels, hidden_channels, kernel_size=1
        )
        self.layer3_projection = _conv_norm_activation(
            layer3_channels, hidden_channels, kernel_size=1
        )
        self.fusion = _conv_norm_activation(
            2 * hidden_channels, hidden_channels, kernel_size=3
        )
        self.heatmap_predictor = nn.Sequential(
            _conv_norm_activation(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, self.keypoint_count, kernel_size=1),
        )
        self.visibility_predictor = nn.Linear(hidden_channels, self.keypoint_count)
        self.variance_predictor = (
            nn.Linear(hidden_channels, 2 * self.keypoint_count)
            if self.predict_variance_correction
            else None
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(2 * self.keypoint_count, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, shape_embedding_dim),
        )

    def forward(self, layer2: Tensor, layer3: Tensor) -> dict[str, Tensor]:
        """
        输入一只眼睛的 ResNet 中层特征 layer2/layer3，预测 15 个眼部伪关键点的概率热力图，
        然后从热力图里解析出每个关键点的坐标均值、协方差、不确定性、可见性，以及一个整体眼形 embedding。
        heatmap 本身给出一个基础不确定性；variance_predictor 再根据整只眼睛的状态，对这个不确定性做放大或缩小
        """
        layer2_features = self.layer2_projection(layer2)  # 投影到同一个通道数128；  -->[B,128,14,23]
        layer3_projected = self.layer3_projection(layer3)
        layer3_features = F.interpolate(
            layer3_projected.float(),
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(dtype=layer3_projected.dtype)  # 投影到同一个通道数128，并上采样；-->[B,128,14,23]
        fused = self.fusion(torch.cat((layer2_features, layer3_features), dim=1))  # -->[B,256,14,23] -->[B,128,14,23]
        decoded = F.interpolate(
            fused.float(),
            size=self.output_size,
            mode="bilinear",
            align_corners=False,  # 控制插值输入和输出网格的角点是否对齐
        ).to(dtype=fused.dtype)  # 插值到高宽为28x45的尺寸；-->[B,128,28,45]
        logits = self.heatmap_predictor(decoded)  # 预测每个关键点的 heatmap logits； -->[B,15,28,45]
        pooled = F.adaptive_avg_pool2d(fused, output_size=1).flatten(start_dim=1)  # 池化到特征维度为128的向量； -->[B,128]
        visibility_logits = self.visibility_predictor(pooled)  # 预测每个关键点的可见性 logits； -->[B,15]
        moments = spatial_probability_moments(logits, self.temperature)  # 计算概率分布的均值、协方差、不确定性
        covariance_heatmap = moments["covariance"]  # 获取概率分布的协方差  
        if self.variance_predictor is None:
            log_std_correction = logits.new_zeros(
                (logits.shape[0], self.keypoint_count, 2), dtype=torch.float32
            )
            covariance = covariance_heatmap
        else:
            raw_correction = self.variance_predictor(pooled).reshape(
                logits.shape[0], self.keypoint_count, 2
            )  # 预测每个关键点的协方差； -->[B,15,2]
            log_std_correction = self.max_abs_log_std_correction * torch.tanh(
                raw_correction.float()
            )  # 限制范围
            std_scale = torch.exp(log_std_correction)  # 变成尺度
            covariance = (
                covariance_heatmap
                * std_scale.unsqueeze(-1)
                * std_scale.unsqueeze(-2)
            )  # 修正协方差
        canonical_shape = canonicalize_eye_landmark_shape(moments["mean_xy"])  # 用关键点的平均坐标 mean_xy 构造一个“标准眼睛坐标系”。
        shape_embedding = F.normalize(
            self.shape_encoder(canonical_shape.flatten(start_dim=1)),
            dim=-1,
            eps=1e-8,
        )  # 用一个向量表示一个“标准眼睛坐标系”，并归一化
        return {
            "logits": logits,  #关键点heatmap; -->[B,15,28,45]
            **moments,  # 关键点概率分布的均值、协方差、不确定性；
            "covariance_heatmap": covariance_heatmap,  # 关键点概率分布的协方差（热力图自己算出来的原始不确定性）；-->[64, 15, 2, 2]
            "covariance": covariance,  # 获取概率分布的协方差（经过全局特征预测的方差修正之后的不确定性）；-->[64, 15, 2, 2]
            "variance_xy": torch.diagonal(covariance, dim1=-2, dim2=-1),  # 提取协方差矩阵的对角线元素（方差）；-->[64, 15, 2]
            "log_std_correction": log_std_correction,  # 获取概率分布的协方差修正参数；-->[64, 15, 2]
            "visibility_logits": visibility_logits,  # 关键点可见性 logits；-->[B,15]
            "visibility_probability": torch.sigmoid(visibility_logits.float()),  # 获取关键点可见性（转换为概率）；-->[B,15]
            "canonical_shape": canonical_shape,  # 用关键点的平均坐标 mean_xy 构造一个“标准眼睛坐标系”；存储了标准眼睛局部坐标系下的坐标；-->[B,15,2]
            "shape_embedding": shape_embedding,  # 用一个向量表示一个“标准眼睛坐标系”，并归一化；-->[64, 32]
        }


class ProbabilisticLandmarkTokenizer(nn.Module):
    """Turn landmark posteriors and shallow eye maps into anatomical tokens."""

    _STATISTIC_DIM = 7

    def __init__(
        self,
        *,
        token_dim: int,
        keypoint_count: int = EYE_PSEUDO_POINT_COUNT,
        layer2_channels: int = 128,
        layer3_channels: int = 256,
        uncertainty_scale_norm: float = 0.25,
        min_quality: float = 0.1,
        detach_probability_for_sampling: bool = True,
        detach_landmark_statistics: bool = True,
    ) -> None:
        super().__init__()
        if min(token_dim, keypoint_count, layer2_channels, layer3_channels) <= 0:
            raise ValueError("Landmark-tokenizer dimensions must be positive.")
        if keypoint_count != EYE_PSEUDO_POINT_COUNT:
            raise ValueError(
                "keypoint_count must match the 15-point pseudo-label schema."
            )
        if uncertainty_scale_norm <= 0:
            raise ValueError("uncertainty_scale_norm must be positive.")
        if not 0 <= min_quality < 1:
            raise ValueError("min_quality must lie in [0, 1).")
        self.keypoint_count = int(keypoint_count)
        self.token_dim = int(token_dim)
        self.uncertainty_scale_norm = float(uncertainty_scale_norm)
        self.min_quality = float(min_quality)
        self.detach_probability_for_sampling = bool(
            detach_probability_for_sampling
        )
        self.detach_landmark_statistics = bool(detach_landmark_statistics)

        self.layer2_projection = nn.Conv2d(
            layer2_channels, token_dim, kernel_size=1, bias=False
        )
        self.layer3_projection = nn.Conv2d(
            layer3_channels, token_dim, kernel_size=1, bias=False
        )
        self.statistics_projection = nn.Sequential(
            nn.LayerNorm(self._STATISTIC_DIM),
            nn.Linear(self._STATISTIC_DIM, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.token_fusion = nn.Sequential(
            nn.Linear(3 * token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_embedding = nn.Parameter(
            torch.zeros(1, self.keypoint_count, token_dim)
        )
        self.eye_type_embedding = nn.Parameter(torch.zeros(2, 1, token_dim))
        self.output_norm = nn.LayerNorm(token_dim)
        nn.init.trunc_normal_(self.point_embedding, std=0.02)
        nn.init.trunc_normal_(self.eye_type_embedding, std=0.02)

    @staticmethod
    def _normalized_probability_at_size(
        probability_maps: Tensor,
        spatial_size: tuple[int, int],
        *,
        detach: bool,
        dtype: torch.dtype,
    ) -> Tensor:
        probabilities = torch.nan_to_num(
            F.interpolate(
                probability_maps.float(),
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            ),  # 插值缩放到spatial_size，后面要和 projected 特征图逐位置相乘、加权求和
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)  # 插值后可能出现 NaN 或 Inf，或者负数，全部置为 0
        probabilities = probabilities / probabilities.sum(
            dim=(-1, -2), keepdim=True
        ).clamp_min(1e-8)  # 经过 F.interpolate 缩放后，总和不一定还等于 1；这里重新做加权平均
        if detach:
            probabilities = probabilities.detach()
        return probabilities.to(dtype=dtype)

    def _appearance_tokens(
        self,
        probability_maps: Tensor,
        feature_map: Tensor,
        projection: nn.Module,
    ) -> Tensor:
        '''
        先从ResNet 特征空间 -> 1x1 Conv -> 关键点采样用的 token 特征空间；再用关键点 heatmap 当作空间权重，在 layer2 特征图上做加权采样
        '''
        projected = projection(feature_map)  # 经过 1x1 Conv，进行投影;1x1 Conv 可以学习重新加权和组合这些通道,让模型学习“哪些通道适合关键点采样”
        probabilities = self._normalized_probability_at_size(
            probability_maps,  # 原始关键点概率图，--> [B,15,28,45]
            tuple(projected.shape[-2:]),  # 当前特征图的空间尺寸
            detach=self.detach_probability_for_sampling,  # 是否切断 probability_maps 到采样过程的梯度
            dtype=projected.dtype,  # 输出 dtype 和 projected 保持一致，比如 float16/bfloat16/float32
        )  # 把关键点 heatmap 概率图缩放到当前特征图的空间大小，并重新整理成一个合法的概率分布，然后用它去做关键点位置的特征采样。
        return torch.einsum("bkhw,bchw->bkc", probabilities, projected)  # 对每个关键点 k，用它的概率热力图，在 feature_map 上做加权平均。

    def _tokenize_eye(
        self,
        *,
        layer2: Tensor,
        layer3: Tensor,
        probability_maps: Tensor,
        mean_xy: Tensor,
        covariance: Tensor,
        visibility_probability: Tensor,
        entropy: Tensor,
        eye_index: int,
    ) -> tuple[Tensor, Tensor]:
        layer2_appearance = self._appearance_tokens(
            probability_maps, layer2, self.layer2_projection
        )  # 从 layer2 提取关键点外观; [64, 128, 14, 23]-->[64, 15, 128]
        layer3_appearance = self._appearance_tokens(
            probability_maps, layer3, self.layer3_projection
        )  # 从 layer3 提取关键点外观; [64, 256, 14, 23]-->[64, 15, 128]
        covariance = 0.5 * (
            covariance.float() + covariance.float().transpose(-1, -2)
        )  # 将协方差矩阵转换为对称矩阵
        statistics = torch.cat(
            (
                mean_xy.float(),  # [64, 15, 2]     ← 期望坐标 (x, y)
                covariance[..., 0, 0:1],  # [64, 15, 2, 2]  方差 σx²  ← 协方差矩阵
                covariance[..., 0, 1:2],  # [64, 15, 1]  协方差 σxy
                covariance[..., 1, 1:2],  # [64, 15, 1]  方差 σy²
                visibility_probability.float().unsqueeze(-1),  # [64, 15]   ← 可见性概率
                entropy.float().unsqueeze(-1),   # [64, 15]    ← 归一化熵
            ),
            dim=-1,
        )  # 将位置、不确定性和置信度拼接成一个统一的一维特征向量，作为每个关键点的完整统计描述; -->[64, 15, 7]
        statistics = torch.nan_to_num(
            statistics, nan=0.0, posinf=1.0, neginf=-1.0
        )  # [64, 15, 7]
        if self.detach_landmark_statistics:
            statistics = statistics.detach()
        statistics_features = self.statistics_projection(statistics).to(
            dtype=layer2_appearance.dtype
        )  # [64, 15, 7] --一个小MLP--> [64, 15, 128]
        tokens = self.token_fusion(
            torch.cat(
                (layer2_appearance, layer3_appearance, statistics_features),
                dim=-1,
            )
        )  # [B,15,128] + [B,15,128] + [B,15,128] --> [B,15,384] --> [64, 15, 128]
        tokens = self.output_norm(
            tokens
            + self.point_embedding.to(dtype=tokens.dtype)  # 表示第几个关键点；[1,15,128]
            + self.eye_type_embedding[eye_index : eye_index + 1].to(
                dtype=tokens.dtype
            )  # 表示左眼还是右眼;[2,1,128] --> [1,1,128]
        )  # 向特征向量中添加位置编码和类型编码；[B,15,128] + [1,15,128] + [1,1,128] -->[64, 15, 128]
        posterior_std = torch.diagonal(
            covariance, dim1=-2, dim2=-1
        ).clamp_min(0.0).sum(dim=-1).sqrt()  # 将对角线（cov_xx, cov_yy）方差求和后开平方根,计算后验标准差；[64, 15]
        visibility = visibility_probability.float().clamp(0.0, 1.0)  # 关键点可见性概率（0~1），来自可见性分类头;[64, 15]
        normalized_entropy = entropy.float().clamp(0.0, 1.0)  # 归一化熵（0~1）;[64, 15]
        raw_quality = (
            visibility
            * (1.0 - 0.5 * normalized_entropy)
            * torch.exp(-posterior_std / self.uncertainty_scale_norm)
        ).clamp(0.0, 1.0)  # 计算原始质量分数;[64, 15]
        quality = self.min_quality + (1.0 - self.min_quality) * raw_quality
        quality = torch.nan_to_num(
            quality,
            nan=self.min_quality,
            posinf=1.0,
            neginf=self.min_quality,
        )  # 映射到最终质量分数;[64, 15]
        return tokens, quality

    def forward(
        self,
        *,
        left_layer2: Tensor,
        left_layer3: Tensor,
        right_layer2: Tensor,
        right_layer3: Tensor,
        predictions: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        '''
        它把左右眼关键点预测结果和 ResNet 的 layer2/layer3 眼部特征，转换成可以给交叉注意力使用的 landmark tokens。
        '''
        left_tokens, left_quality = self._tokenize_eye(
            layer2=left_layer2,
            layer3=left_layer3,
            probability_maps=predictions["left_eye_keypoint_probability_maps"],
            mean_xy=predictions["left_eye_keypoint_mean_xy"],
            covariance=predictions["left_eye_keypoint_covariance"],
            visibility_probability=predictions[
                "left_eye_keypoint_visibility_probability"
            ],
            entropy=predictions["left_eye_keypoint_entropy"],
            eye_index=0,
        )
        right_tokens, right_quality = self._tokenize_eye(
            layer2=right_layer2,
            layer3=right_layer3,
            probability_maps=predictions["right_eye_keypoint_probability_maps"],
            mean_xy=predictions["right_eye_keypoint_mean_xy"],
            covariance=predictions["right_eye_keypoint_covariance"],
            visibility_probability=predictions[
                "right_eye_keypoint_visibility_probability"
            ],
            entropy=predictions["right_eye_keypoint_entropy"],
            eye_index=1,
        )
        return {
            "tokens": torch.cat((left_tokens, right_tokens), dim=1),
            "quality": torch.cat((left_quality, right_quality), dim=1),
            "left_tokens": left_tokens,
            "right_tokens": right_tokens,
            "left_quality": left_quality,
            "right_quality": right_quality,
        }


class LandmarkGuidedEyeFusion(nn.Module):
    """Inject probabilistic eye anatomy before face fusion."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        keypoint_count: int = EYE_PSEUDO_POINT_COUNT,
        gate_init: float = 0.0,
        gate_max: float = 1.0,
        quality_logit_scale: float = 1.0,
        uncertainty_scale_norm: float = 0.25,
        min_quality: float = 0.1,
        detach_probability_for_sampling: bool = True,
        detach_landmark_statistics: bool = True,
        detach_quality: bool = True,
    ) -> None:
        super().__init__()
        if min(token_dim, num_heads) <= 0:
            raise ValueError("Fusion token/head dimensions must be positive.")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if gate_max <= 0 or not -gate_max < gate_init < gate_max:
            raise ValueError(
                "gate_init must lie strictly inside (-gate_max, gate_max)."
            )
        if quality_logit_scale < 0:
            raise ValueError("quality_logit_scale must be non-negative.")
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.gate_init = float(gate_init)
        self.gate_max = float(gate_max)
        self.quality_logit_scale = float(quality_logit_scale)
        self.detach_quality = bool(detach_quality)

        self.tokenizer = ProbabilisticLandmarkTokenizer(
            token_dim=token_dim,
            keypoint_count=keypoint_count,
            uncertainty_scale_norm=uncertainty_scale_norm,
            min_quality=min_quality,
            detach_probability_for_sampling=detach_probability_for_sampling,
            detach_landmark_statistics=detach_landmark_statistics,
        )
        self.quality_embedding = nn.Linear(1, token_dim, bias=False)
        self.query_norm = nn.LayerNorm(token_dim)
        self.landmark_norm = nn.LayerNorm(token_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.delta_norm = nn.LayerNorm(token_dim)
        self.delta_dropout = nn.Dropout(dropout)
        self.raw_gate = nn.Parameter(torch.empty(()))
        self.reset_gate()

    def reset_gate(self) -> None:
        normalized = self.gate_init / self.gate_max
        with torch.no_grad():
            self.raw_gate.fill_(math.atanh(normalized))

    def forward(
        self,
        eye_tokens: Tensor,
        *,
        left_layer2: Tensor,
        left_layer3: Tensor,
        right_layer2: Tensor,
        right_layer3: Tensor,
        predictions: Mapping[str, Tensor],
        return_attention: bool = False,
    ) -> dict[str, Tensor]:
        # if eye_tokens.ndim != 3 or eye_tokens.shape[-1] != self.token_dim:
        tokenized = self.tokenizer(
            left_layer2=left_layer2,
            left_layer3=left_layer3,
            right_layer2=right_layer2,
            right_layer3=right_layer3,
            predictions=predictions,
        )
        landmark_tokens = tokenized["tokens"]  # 关键点 Token;[64, 30, 128]
        quality = tokenized["quality"]  # [64, 30]
        quality_for_fusion = quality.detach() if self.detach_quality else quality  # 切断沿着 quality 的反向传播路径;[64, 30]
        # 质量是一个控制信号，用来告诉融合模块该信任哪些关键点；避免主任务为了获得更好的融合结果，反过来任意操纵质量分数
        query = self.query_norm(eye_tokens)  # Query：眼图 Token;[64, 168, 128]
        landmark_key = self.landmark_norm(  
            landmark_tokens
            + self.quality_embedding(
                quality_for_fusion.to(dtype=landmark_tokens.dtype).unsqueeze(-1) 
            )  # 关键点质量嵌入;[64, 30] --> [64, 30, 1]
        )  # Key：关键点身份和质量
        landmark_value = landmark_tokens * quality_for_fusion.to(
            dtype=landmark_tokens.dtype
        ).unsqueeze(-1)  # Value：被取出的关键点内容;[64, 30, 128]

        attention_bias = self.quality_logit_scale * torch.log(
            quality_for_fusion.clamp_min(1e-4)
        )  # 关键点质量对权重的缩放;[64, 30]
        attention_bias = (
            attention_bias[:, None, None, :]  # [64, 1, 1, 30]
            .expand(-1, self.num_heads, eye_tokens.shape[1], -1)  # expand[-1, 4, 168, -1] --> [64, 4, 168, 30]
            .reshape(
                eye_tokens.shape[0] * self.num_heads,  # 64*4=256
                eye_tokens.shape[1],  # 168
                landmark_tokens.shape[1],  # 30
            )
            .to(dtype=query.dtype)
        )  # [256, 168, 30]
        attended, attention_weights = self.cross_attention(
            query=query,  # [64, 168, 128]  眼图空间 Token
            key=landmark_key,  # [64, 30,128]  关键点匹配信息
            value=landmark_value,  # [64, 30, 128]  关键点内容
            attn_mask=attention_bias,  # 不希望查询（168个空间位置）关注的位置（30个关键点中的某些）,掩码值会设为 -inf 或一个极大的负数，使得 Softmax 后这些位置的权重变为 0
            need_weights=return_attention,  # False；对 scores 的最后一个维度（S=30）做 Softmax，得到注意力权重 attn_weights
            average_attn_weights=False,
        )  # [64, 168, 128], None
        delta = self.delta_dropout(self.delta_norm(attended))  # 修正量 delta；不直接加到眼图 Token 上，需要经过门控；[64, 168, 128]
        global_gate = self.gate_max * torch.tanh(self.raw_gate)  # 可训练参数，关键点修正眼图特征
        sample_quality = quality_for_fusion.mean(dim=-1, keepdim=True)  # 将 30 个关键点的质量取平均;[64, 1]
        applied_gate = global_gate * sample_quality  # [64, 1];关键点修正眼图特征最终门控同时取决于：模型全局是否愿意使用关键点信息;当前样本的关键点是否可靠
        applied_delta = applied_gate.unsqueeze(-1) * delta  # [64, 168, 128]
        refined_eye_tokens = eye_tokens + applied_delta

        output = {
            "refined_eye_tokens": refined_eye_tokens,
            "landmark_tokens": landmark_tokens,
            "landmark_quality": quality,
            "sample_quality": sample_quality,
            "global_gate": global_gate.expand(eye_tokens.shape[0], 1),
            "applied_gate": applied_gate,
            "delta_norm": torch.linalg.vector_norm(
                applied_delta.float(), dim=-1  # 沿着最后一维计算 L2范数； [64, 168, 128] --> [64, 168]
            ).mean(dim=-1, keepdim=True),  # 沿着关键点维度求平均值，保留维度；[64, 168] --> [64, 1]
            # 代表该批次中所有关键点位移/特征向量的平均长度
        }  
        if return_attention and attention_weights is not None:
            output["attention_weights"] = attention_weights
        return output


def _conv_norm_activation(
    input_channels: int,
    output_channels: int,
    *,
    kernel_size: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(output_channels),
        nn.SiLU(inplace=True),
    )

def spatial_probability_moments(
    logits: Tensor,
    temperature: float,
) -> dict[str, Tensor]:
    """Return spatial probabilities, normalized means, and full covariance."""


    # if logits.ndim != 4:
    #     raise ValueError("Landmark logits must have shape [B, K, H, W].")
    # if temperature <= 0:
    #     raise ValueError("temperature must be positive.")
    batch_size, keypoint_count, height, width = logits.shape
    probabilities = torch.softmax(
        logits.float().flatten(start_dim=2) / float(temperature), dim=-1
    )  # 对每个关键点的 28x45 热力图做 softmax，得到概率分布; --> [B,15,28,45]
    y_coordinates, x_coordinates = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=logits.device),
        torch.linspace(0.0, 1.0, width, device=logits.device),
        indexing="ij",
    )  # 创建一个坐标网格
    coordinate_grid = torch.stack((x_coordinates, y_coordinates), dim=-1).reshape(
        height * width, 2
    )  # 堆叠成坐标表； --> [28*45,2]
    mean_xy = torch.einsum("bkn,nc->bkc", probabilities, coordinate_grid)  # 用概率作权重，对所有像素坐标做加权求和；归一化坐标； -->[B,15,2]
    centered = coordinate_grid[None, None] - mean_xy[:, :, None]
    covariance = torch.einsum(
        "bkn,bknc,bknd->bkcd", probabilities, centered, centered
    )  # 计算协方差矩阵
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))  # 计算对数概率
    entropy = -(probabilities * log_probabilities).sum(dim=-1)  # 计算熵
    if height * width > 1:
        entropy = entropy / math.log(height * width)
    return {
        "probability_maps": probabilities.reshape(
            batch_size, keypoint_count, height, width
        ),
        "mean_xy": mean_xy,
        "covariance": covariance,
        "variance_xy": torch.diagonal(covariance, dim1=-2, dim2=-1),
        "entropy": entropy,
        "confidence": (1.0 - entropy).clamp(0.0, 1.0),  # 获取关键点置信度
    }


def canonicalize_eye_landmark_shape(points_xy: Tensor) -> Tensor:
    """用外眼角到内眼角的方向作为局部 x 轴，用两眼角中点作为原点，用眼角距离作为尺度。
       去掉一部分平移、尺度、左右眼方向差异，让模型更关注“眼形结构本身”"""
    if points_xy.ndim != 3 or points_xy.shape[1:] != (EYE_PSEUDO_POINT_COUNT, 2):
        raise ValueError("points_xy must have shape [B, 15, 2].")
    outer = points_xy[:, OUTER_CANTHUS_INDEX]  # 第 0 个点：outer canthus，外眼角
    inner = points_xy[:, INNER_CANTHUS_INDEX]  # 第 5 个点：inner canthus，内眼角
    origin = 0.5 * (outer + inner)
    x_vector = inner - outer
    scale = torch.linalg.vector_norm(x_vector, dim=-1, keepdim=True).clamp_min(1e-4)
    x_axis = x_vector / scale
    y_axis = torch.stack((-x_axis[:, 1], x_axis[:, 0]), dim=-1)
    upper = points_xy[:, 2]
    lower = points_xy[:, 8]
    orientation = torch.sign(((lower - upper) * y_axis).sum(dim=-1, keepdim=True))
    orientation = torch.where(orientation == 0, torch.ones_like(orientation), orientation)
    y_axis = y_axis * orientation
    centered = points_xy - origin[:, None]
    local_x = (centered * x_axis[:, None]).sum(dim=-1) / scale
    local_y = (centered * y_axis[:, None]).sum(dim=-1) / scale
    return torch.stack((local_x, local_y), dim=-1)


def probabilistic_landmark_supervision_losses(
    *,
    predicted_mean_xy: Tensor,
    heatmap_logits: Tensor,
    visibility_logits: Tensor,
    target_xy: Tensor,
    valid_mask: Tensor,
    teacher_quality: Tensor,
    visibility_supervision_weight: Tensor,
    huber_delta_norm: float,
    heatmap_sigma_min_px: float,
    heatmap_sigma_max_px: float,
) -> dict[str, Tensor]:
    """Coordinate, Gaussian-distribution, and visibility supervision."""

    if huber_delta_norm <= 0:
        raise ValueError("huber_delta_norm must be positive.")
    if not 0 < heatmap_sigma_min_px <= heatmap_sigma_max_px:
        raise ValueError("Heatmap sigma bounds must satisfy 0 < min <= max.")
    batch_size, eye_count, keypoint_count, _ = target_xy.shape
    if eye_count != 2 or keypoint_count != EYE_PSEUDO_POINT_COUNT:
        raise ValueError("target_xy must have shape [B, 2, 15, 2].")
    if predicted_mean_xy.shape != target_xy.shape:
        raise ValueError("predicted_mean_xy must match target_xy.")
    if valid_mask.shape != target_xy.shape[:-1]:
        raise ValueError("valid_mask must have shape [B, 2, 15].")
    if teacher_quality.shape != (batch_size, 2):
        raise ValueError("teacher_quality must have shape [B, 2].")
    if visibility_supervision_weight.shape != (batch_size, 2):
        raise ValueError("visibility_supervision_weight must have shape [B, 2].")
    if heatmap_logits.ndim != 5 or heatmap_logits.shape[:3] != (
        batch_size,
        2,
        keypoint_count,
    ):
        raise ValueError("heatmap_logits must have shape [B, 2, 15, H, W].")
    if visibility_logits.shape != valid_mask.shape:
        raise ValueError("visibility_logits must have shape [B, 2, 15].")

    targets = target_xy.to(device=predicted_mean_xy.device, dtype=torch.float32)
    valid = valid_mask.to(device=predicted_mean_xy.device, dtype=torch.float32)
    quality = teacher_quality.to(device=predicted_mean_xy.device, dtype=torch.float32)
    visibility_weight = visibility_supervision_weight.to(
        device=predicted_mean_xy.device, dtype=torch.float32
    )
    point_weight = valid * quality.unsqueeze(-1)
    denominator = point_weight.sum().clamp_min(1.0)
    coordinate_loss = F.smooth_l1_loss(
        predicted_mean_xy.float(),
        targets,
        beta=huber_delta_norm,
        reduction="none",
    ).sum(dim=-1)
    coordinate_loss = (coordinate_loss * point_weight).sum() / denominator

    _, _, _, height, width = heatmap_logits.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=heatmap_logits.device, dtype=torch.float32),
        torch.arange(width, device=heatmap_logits.device, dtype=torch.float32),
        indexing="ij",
    )
    centers_px = targets.clone()
    centers_px[..., 0] *= max(width - 1, 1)
    centers_px[..., 1] *= max(height - 1, 1)
    sigma = heatmap_sigma_min_px + (1.0 - quality) * (
        heatmap_sigma_max_px - heatmap_sigma_min_px
    )
    distance_sq = (
        (x_grid[None, None, None] - centers_px[..., 0, None, None]).square()
        + (y_grid[None, None, None] - centers_px[..., 1, None, None]).square()
    )
    target_distribution = torch.exp(
        -0.5 * distance_sq / sigma[:, :, None, None, None].square()
    )
    target_distribution = target_distribution / target_distribution.sum(
        dim=(-1, -2), keepdim=True
    ).clamp_min(1e-8)
    log_probability = F.log_softmax(
        heatmap_logits.float().flatten(start_dim=-2), dim=-1
    ).reshape_as(heatmap_logits)
    heatmap_cross_entropy = -(
        target_distribution * log_probability
    ).sum(dim=(-1, -2))
    heatmap_loss = (heatmap_cross_entropy * point_weight).sum() / denominator

    visibility_bce = F.binary_cross_entropy_with_logits(
        visibility_logits.float(), valid, reduction="none"
    )
    visibility_point_weight = visibility_weight.unsqueeze(-1).expand_as(valid)
    visibility_loss = (
        visibility_bce * visibility_point_weight
    ).sum() / visibility_point_weight.sum().clamp_min(1.0)
    return {
        "coordinate_loss": coordinate_loss,
        "heatmap_loss": heatmap_loss,
        "visibility_loss": visibility_loss,
        "effective_point_weight": point_weight.sum(),
    }


def mirror_shape_consistency_loss(
    left_shape_embedding: Tensor,
    right_shape_embedding: Tensor,
    eye_weight: Tensor,
) -> Tensor:
    """Match canonical shared shape embeddings, never raw left/right coordinates."""

    if left_shape_embedding.shape != right_shape_embedding.shape:
        raise ValueError("Left/right shape embeddings must share shape.")
    if eye_weight.shape != (left_shape_embedding.shape[0], 2):
        raise ValueError("eye_weight must have shape [B, 2].")
    sample_weight = eye_weight.to(
        device=left_shape_embedding.device, dtype=torch.float32
    ).prod(dim=-1)
    cosine_distance = 1.0 - F.cosine_similarity(
        left_shape_embedding.float(), right_shape_embedding.float(), dim=-1
    )
    return (cosine_distance * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)


def sample_eye_affine_matrices(
    batch_size: int,
    *,
    device: torch.device,
    max_rotation_deg: float,
    max_translation_norm: float,
    scale_min: float,
    scale_max: float,
    deterministic_sign: int | None = None,
) -> Tensor:
    """Sample forward transforms from original to augmented crop coordinates."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_rotation_deg < 0 or not 0 <= max_translation_norm < 0.5:
        raise ValueError("Invalid affine rotation/translation range.")
    if not 0 < scale_min <= scale_max:
        raise ValueError("Affine scale bounds must satisfy 0 < min <= max.")
    if deterministic_sign not in (None, -1, 1):
        raise ValueError("deterministic_sign must be None, -1, or 1.")
    if deterministic_sign is None:
        angle_deg = torch.empty(batch_size, device=device).uniform_(
            -max_rotation_deg, max_rotation_deg
        )
        translation = torch.empty(batch_size, 2, device=device).uniform_(
            -max_translation_norm, max_translation_norm
        )
        scale = torch.empty(batch_size, device=device).uniform_(scale_min, scale_max)
    else:
        sign = float(deterministic_sign)
        angle_deg = torch.full(
            (batch_size,), sign * 0.7 * max_rotation_deg, device=device
        )
        translation = torch.tensor(
            [sign * 0.5 * max_translation_norm, -sign * 0.35 * max_translation_norm],
            device=device,
        ).expand(batch_size, -1)
        scale = torch.ones(batch_size, device=device)
    angle = torch.deg2rad(angle_deg)
    cosine = torch.cos(angle) * scale
    sine = torch.sin(angle) * scale
    linear = torch.stack(
        (
            torch.stack((cosine, -sine), dim=-1),
            torch.stack((sine, cosine), dim=-1),
        ),
        dim=-2,
    )
    center = torch.full((batch_size, 2), 0.5, device=device)
    offset = center + translation - torch.bmm(
        linear, center.unsqueeze(-1)
    ).squeeze(-1)
    matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    matrix[:, :2, :2] = linear
    matrix[:, :2, 2] = offset
    return matrix


def warp_eye_images(images: Tensor, forward_matrix: Tensor) -> Tensor:
    """Warp normalized eye images using a known original-to-augmented matrix."""

    if images.ndim != 4 or forward_matrix.shape != (images.shape[0], 3, 3):
        raise ValueError("Expected images [B,C,H,W] and matrices [B,3,3].")
    batch_size, _, height, width = images.shape
    y_coordinates, x_coordinates = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=images.device),
        torch.linspace(0.0, 1.0, width, device=images.device),
        indexing="ij",
    )
    output_grid = torch.stack(
        (x_coordinates, y_coordinates, torch.ones_like(x_coordinates)), dim=-1
    ).reshape(1, height * width, 3).expand(batch_size, -1, -1)
    inverse = torch.linalg.inv(forward_matrix.float())
    source = torch.bmm(output_grid, inverse.transpose(1, 2))[..., :2]
    sampling_grid = (2.0 * source - 1.0).reshape(batch_size, height, width, 2)
    return F.grid_sample(
        images,
        sampling_grid.to(dtype=images.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def transform_normalized_points(points_xy: Tensor, matrix: Tensor) -> Tensor:
    if points_xy.ndim < 3 or points_xy.shape[0] != matrix.shape[0]:
        raise ValueError("Points and matrices must share a batch dimension.")
    homogeneous = torch.cat((points_xy, torch.ones_like(points_xy[..., :1])), dim=-1)
    return torch.einsum("bij,b...j->b...i", matrix.float(), homogeneous.float())[..., :2]


def equivariant_landmark_consistency_loss(
    view1_mean_xy: Tensor,
    view2_mean_xy: Tensor,
    view1_matrix: Tensor,
    view2_matrix: Tensor,
    eye_weight: Tensor,
    *,
    huber_delta_norm: float,
) -> dict[str, Tensor]:
    """Compare two augmented predictions after mapping both back to the crop."""

    if view1_mean_xy.shape != view2_mean_xy.shape:
        raise ValueError("Equivariant landmark predictions must share shape.")
    if view1_mean_xy.ndim != 4 or view1_mean_xy.shape[1:] != (
        2,
        EYE_PSEUDO_POINT_COUNT,
        2,
    ):
        raise ValueError("Equivariant means must have shape [B, 2, 15, 2].")
    if eye_weight.shape != (view1_mean_xy.shape[0], 2):
        raise ValueError("eye_weight must have shape [B, 2].")
    inverse1 = torch.linalg.inv(view1_matrix.float())
    inverse2 = torch.linalg.inv(view2_matrix.float())
    canonical1 = transform_normalized_points(view1_mean_xy, inverse1)
    canonical2 = transform_normalized_points(view2_mean_xy, inverse2)
    in_bounds = (
        (canonical1 >= 0.0).all(dim=-1)
        & (canonical1 <= 1.0).all(dim=-1)
        & (canonical2 >= 0.0).all(dim=-1)
        & (canonical2 <= 1.0).all(dim=-1)
    ).float()
    weight = eye_weight.to(device=canonical1.device, dtype=torch.float32).unsqueeze(-1)
    weight = weight * in_bounds
    per_point = F.smooth_l1_loss(
        canonical1,
        canonical2,
        beta=huber_delta_norm,
        reduction="none",
    ).sum(dim=-1)
    denominator = weight.sum().clamp_min(1.0)
    return {
        "loss": (per_point * weight).sum() / denominator,
        "mean_error_norm": (
            torch.linalg.vector_norm(canonical1 - canonical2, dim=-1) * weight
        ).sum()
        / denominator,
    }


def eye_landmark_validation_metrics(
    predicted_mean_xy: Tensor,
    target_xy: Tensor,
    valid_mask: Tensor,
    teacher_quality: Tensor,
    *,
    eye_image_size: tuple[int, int],
) -> dict[str, Tensor]:
    """Return validation-only pseudo-label accuracy and calibration metrics."""

    if predicted_mean_xy.shape != target_xy.shape or predicted_mean_xy.ndim != 4:
        raise ValueError("Predicted/target landmarks must share shape [B,2,K,2].")
    width, height = eye_image_size
    pixel_scale = predicted_mean_xy.new_tensor((width, height))
    predicted_px = predicted_mean_xy.float() * pixel_scale
    target_px = target_xy.float() * pixel_scale
    valid = valid_mask.float()
    point_error_px = torch.linalg.vector_norm(predicted_px - target_px, dim=-1)
    canthus_distance_px = torch.linalg.vector_norm(
        target_px[:, :, INNER_CANTHUS_INDEX]
        - target_px[:, :, OUTER_CANTHUS_INDEX],
        dim=-1,
    ).clamp_min(1.0)
    nme_per_point = point_error_px / canthus_distance_px.unsqueeze(-1)
    scale_valid = (
        valid[:, :, OUTER_CANTHUS_INDEX]
        * valid[:, :, INNER_CANTHUS_INDEX]
    )
    nme_valid = valid * scale_valid.unsqueeze(-1)
    nme_denominator = nme_valid.sum().clamp_min(1.0)
    nme = (nme_per_point * nme_valid).sum() / nme_denominator

    point_denominator = valid.sum().clamp_min(1.0)

    iris_valid = valid[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
    iris_error = point_error_px[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
    iris_center_error_px = (iris_error * iris_valid).sum() / iris_valid.sum().clamp_min(1.0)
    vertical_error_px = (
        (predicted_px[..., 1] - target_px[..., 1]).abs() * valid
    ).sum() / point_denominator

    rim_valid = valid[:, :, IRIS_RIM_SLICE].prod(dim=-1) * iris_valid
    teacher_center = target_px[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
    teacher_rim = target_px[:, :, IRIS_RIM_SLICE]
    predicted_rim = predicted_px[:, :, IRIS_RIM_SLICE]
    teacher_delta = teacher_rim - teacher_center.unsqueeze(-2)
    teacher_covariance = torch.einsum(
        "beqi,beqj->beij", teacher_delta, teacher_delta
    ) / teacher_delta.shape[-2]
    identity = torch.eye(2, device=target_px.device).reshape(1, 1, 2, 2)
    teacher_inverse = torch.linalg.inv(teacher_covariance + identity * 1e-3)
    predicted_delta = predicted_rim - teacher_center.unsqueeze(-2)
    predicted_radius_sq = 0.5 * torch.einsum(
        "beqi,beij,beqj->beq",
        predicted_delta,
        teacher_inverse,
        predicted_delta,
    ).clamp_min(0.0)
    teacher_radius_sq = 0.5 * torch.einsum(
        "beqi,beij,beqj->beq",
        teacher_delta,
        teacher_inverse,
        teacher_delta,
    ).clamp_min(0.0)
    ellipse_error = (
        torch.sqrt(predicted_radius_sq + 1e-12)
        - torch.sqrt(teacher_radius_sq + 1e-12)
    ).abs().mean(dim=-1)
    ellipse_reprojection_error = (
        ellipse_error * rim_valid
    ).sum() / rim_valid.sum().clamp_min(1.0)

    eye_valid_count = nme_valid.sum(dim=-1)
    per_eye_error = (
        (nme_per_point * nme_valid).sum(dim=-1)
        / eye_valid_count.clamp_min(1.0)
    )
    correlation_mask = eye_valid_count > 0
    quality = teacher_quality.float()[correlation_mask]
    errors = per_eye_error[correlation_mask]
    if quality.numel() >= 2:
        centered_quality = quality - quality.mean()
        centered_error = errors - errors.mean()
        correlation = (
            (centered_quality * centered_error).mean()
            / (
                centered_quality.square().mean().sqrt()
                * centered_error.square().mean().sqrt()
            ).clamp_min(1e-8)
        )
    else:
        correlation = predicted_mean_xy.new_zeros(())
    return {
        "eye_landmark_nme": nme,
        "eye_landmark_iris_center_error_px": iris_center_error_px,
        "eye_landmark_vertical_error_px": vertical_error_px,
        "eye_landmark_ellipse_reprojection_error": ellipse_reprojection_error,
        "eye_landmark_teacher_quality_error_corr": correlation,
    }


def stack_binocular_outputs(
    outputs: Mapping[str, Tensor],
    suffix: str,
) -> Tensor:
    """Stack left/right model outputs into anatomical eye dimension 1."""

    return torch.stack(
        (outputs[f"left_eye_keypoint_{suffix}"], outputs[f"right_eye_keypoint_{suffix}"]),
        dim=1,
    )
