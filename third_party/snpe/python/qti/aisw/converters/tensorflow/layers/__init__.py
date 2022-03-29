# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np


from qti.aisw.converters.tensorflow.layers.batch_to_space import (
    BatchToSpaceLayerResolver,
    BatchToSpaceLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.fullyconnected import (
    FullyConnectedLayerResolver,
    FullyConnectedLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.cast import (
    CastLayerResolver,
    CastLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.convolution import (
    ConvolutionLayerResolver,
    DilatedConvolutionLayerResolver,
    ConvolutionLayerBuilder,
    DepthwiseConvolutionLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver,
    DepthwiseConvolutionLayerBuilder,
    GroupedConvolutionLayerResolver
)

from qti.aisw.converters.tensorflow.layers.concat import (
    ConcatLayerResolver,
    ConcatLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.relu import (
    ReluLayerResolver,
    ReluLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.relu_min_max import (
    ReluMinMaxLayerResolver,
    ReluMinMaxLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.relu6 import (
    Relu6LayerResolver
)
from qti.aisw.converters.tensorflow.layers.sigmoid import (
    SigmoidLayerResolver,
    SigmoidLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.tanh import (
    TanhLayerResolver,
    TanhLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.softmax import (
    SoftmaxLayerResolver,
    SoftmaxLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.lrn import (
    LrnLayerResolver,
    LrnLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.gather import (
    GatherLayerResolver,
    GatherLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.deconvolution import (
    DeconvolutionLayerResolver,
    DeconvolutionLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.batchnorm import (
    BatchNormLayerResolver,
    BatchNormWithEltwiseLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    BatchNormLayerBuilder,
    FusedBatchNormNormLayerResolver,
    GenericBatchNormLayerResolver
)

from qti.aisw.converters.tensorflow.layers.instance_norm import (
    InstanceNormLayerBuilder,
    InstanceNormLayerResolver
)

from qti.aisw.converters.tensorflow.layers.pooling import (
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    PoolingLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.eltwise import (
    EltWiseAndLayerResolver,
    EltWiseAndLayerBuilder,
    EltWiseEqualLayerResolver,
    EltWiseEqualLayerBuilder,
    EltWiseFloorDivLayerResolver,
    EltWiseFloorDivLayerBuilder,
    EltWiseGreaterLayerResolver,
    EltWiseGreaterLayerBuilder,
    EltWiseGreaterEqualLayerResolver,
    EltWiseGreaterEqualLayerBuilder,
    EltWiseLessLayerResolver,
    EltWiseLessLayerBuilder,
    EltWiseLessEqualLayerResolver,
    EltWiseLessEqualLayerBuilder,
    EltWiseNotEqualLayerResolver,
    EltWiseNotEqualLayerBuilder,
    EltWiseOrLayerResolver,
    EltWiseOrLayerBuilder,
    EltWisePowLayerResolver,
    EltWisePowLayerBuilder,
    EltWiseSelectLayerResolver,
    EltWiseSelectLayerBuilder,
    EltWiseSumLayerResolver,
    EltWiseSumLayerBuilder,
    EltWiseBiasaddLayerResolver,
    EltWiseBiasaddLayerBuilder,
    EltWiseSubLayerResolver,
    EltWiseSubLayerBuilder,
    EltWiseMulLayerResolver,
    EltWiseMulLayerBuilder,
    EltWiseMaxLayerResolver,
    EltWiseMaxLayerBuilder,
    EltWiseMinLayerResolver,
    EltWiseMinLayerBuilder,
    EltWiseDivLayerResolver,
    EltWiseDivLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.add_n import (
    AddNLayerResolver,
    AddNLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.slice import (
    SliceLayerResolver,
    SliceLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.prelu import (
    PReLuLayerResolver,
    LeakyReLuLayerResolver,
    PReLuLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.reshape import (
    ReshapeLayerResolver,
    ReshapeLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.resize import (
    ResizeNearestNeighborLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.l2_normalize import (
    L2NormLayerResolver,
    L2NormLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.logsoftmax import (
    LogSoftmaxLayerResolver,
    LogSoftmaxLayerBuilder
)


from qti.aisw.converters.tensorflow.layers.lstm import (
    MergedWeightsLstmLayerResolver,
    SplitWeightsLstmLayerResolver,
    LstmLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.ignored_patterns import (
    IgnoredLayersResolver,
    IgnoredLayersBuilder
)

from qti.aisw.converters.tensorflow.layers.fill import (
    FillLayerResolver,
    FillLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.ssd import (
    SSDDecoderResolver,
    SSDDecoderLayersBuilder,
    SSDNmsResolver,
    SSDNmsLayersBuilder,
    SSDAnchorGeneratorResolver,
    Tf2SSDNmsResolver,
)

from qti.aisw.converters.tensorflow.layers.space_to_batch import (
    SpaceToBatchLayerResolver,
    SpaceToBatchLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.crop import (
    CropLayerResolver,
    CropLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.constant import (
    ConstantLayerResolver,
    ConstantLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.pad import (
    PadLayerResolver,
    PadLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.strided_slice import (
    StridedSliceLayerResolver,
    StridedSliceLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.permute import (
    PermuteLayerResolver,
    PermuteLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.argmax import (
    ArgMaxLayerResolver,
    ArgMaxLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.argmin import (
    ArgMinLayerResolver,
    ArgMinLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.channel_shuffle import (
    ChannelShuffleLayerResolver,
    ChannelShuffleLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.elu import (
    EluLayerResolver,
    EluLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.reduction import (
    ReductionMeanLayerResolver,
    ReductionMeanLayerBuilder,
    ReductionProdLayerResolver,
    ReductionProdLayerBuilder,
    ReductionSumLayerResolver,
    ReductionSumLayerBuilder,
    ReductionMinLayerResolver,
    ReductionMinLayerBuilder,
    ReductionMaxLayerResolver,
    ReductionMaxLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.eltwise_unary import (
    EltWiseUnaryAbsLayerResolver,
    EltWiseUnaryAbsLayerBuilder,
    EltWiseUnaryCeilLayerResolver,
    EltWiseUnaryCeilLayerBuilder,
    EltWiseUnaryExpLayerResolver,
    EltWiseUnaryExpLayerBuilder,
    EltWiseUnaryFloorLayerResolver,
    EltWiseUnaryFloorLayerBuilder,
    EltWiseUnaryLogLayerResolver,
    EltWiseUnaryLogLayerBuilder,
    EltWiseUnaryLogicalNotLayerResolver,
    EltWiseUnaryLogicalNotLayerBuilder,
    EltWiseUnaryNegLayerResolver,
    EltWiseUnaryNegLayerBuilder,
    EltWiseUnaryRoundLayerResolver,
    EltWiseUnaryRoundLayerBuilder,
    EltWiseUnaryRsqrtLayerResolver,
    EltWiseUnaryRsqrtLayerBuilder,
    EltWiseUnarySinLayerResolver,
    EltWiseUnarySinLayerBuilder,
    EltWiseUnarySqrtLayerResolver,
    EltWiseUnarySqrtLayerBuilder,
    EltWiseUnarySquareLayerResolver,
    EltWiseUnarySquareLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.tile import (
    TileLayerResolver,
    TileLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.extract_glimpse import (
    ExtractGlimpseLayerResolver,
    ExtractGlimpseLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.image_projective_transform import (
    ImageProjectiveTransformLayerResolver,
    ImageProjectiveTransformLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.fake_quant import (
    FakeQuantLayerResolver,
    FakeQuantPerChannelResolver,
    FakeQuantLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.matmul import(
    MatMulLayerResolver,
    MatMulLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.pixel_shuffle import (
    PixelShuffleLayerResolver,
    PixelShuffleLayerBuilder
)


from qti.aisw.converters.tensorflow.layers.crop_and_resize import (
    CropAndResizeLayerResolver,
    CropAndResizeLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.non_max_suppression import (
    NonMaxSuppressionLayerResolver,
    NonMaxSuppressionLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.moments import (
    MomentsLayerResolver,
    MomentsLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.space_to_depth import (
    SpaceToDepthLayerResolver,
    SpaceToDepthLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.caffe_ssd import (
    CaffeSsdLayerResolver,
    CaffeSsdLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.pack import (
    PackLayerResolver,
    PackLayerBuilder,
    UnPackLayerResolver,
    UnpackLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.topk import (
    TopKLayerResolver,
    TopKLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.identity_n import (
    IdentityNLayerResolver,
    IdentityNLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.one_hot import (
    OneHotLayerResolver,
    OneHotLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.custom import (
    CustomLayerResolver,
    CustomLayerBuilder,
)

from qti.aisw.converters.tensorflow.common import (
    LayerDescriptor,
    LayerResolver,
    LayerBuilder
)

from qti.aisw.converters.tensorflow.layers.gelu import (
    GeLuLayerResolver,
    GeLuLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.layer_norm import (
    LayerNormLayerResolver,
    LayerNormLayerBuilder
)

layer_resolvers = [
    IgnoredLayersResolver,
    CastLayerResolver,
    FakeQuantLayerResolver,
    Tf2SSDNmsResolver,
    CaffeSsdLayerResolver,
    SSDAnchorGeneratorResolver,
    SSDNmsResolver,
    ConvolutionLayerResolver,
    ReshapeLayerResolver,
    ConcatLayerResolver,
    FullyConnectedLayerResolver,
    ReluLayerResolver,
    Relu6LayerResolver,
    ReluMinMaxLayerResolver,
    SigmoidLayerResolver,
    TanhLayerResolver,
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    NonMaxSuppressionLayerResolver,
    SoftmaxLayerResolver,
    L2NormLayerResolver,
    LrnLayerResolver,
    DeconvolutionLayerResolver,
    LayerNormLayerResolver,
    InstanceNormLayerResolver,
    EltWiseAndLayerResolver,
    EltWiseEqualLayerResolver,
    EltWiseGreaterLayerResolver,
    EltWiseGreaterEqualLayerResolver,
    EltWiseLessLayerResolver,
    EltWiseLessEqualLayerResolver,
    EltWiseNotEqualLayerResolver,
    EltWiseOrLayerResolver,
    EltWisePowLayerResolver,
    EltWiseSelectLayerResolver,
    EltWiseSumLayerResolver,
    EltWiseBiasaddLayerResolver,
    EltWiseSubLayerResolver,
    EltWiseMulLayerResolver,
    EltWiseMaxLayerResolver,
    EltWiseMinLayerResolver,
    EltWiseDivLayerResolver,
    GeLuLayerResolver,
    BatchNormWithEltwiseLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    BatchToSpaceLayerResolver,
    GenericBatchNormLayerResolver,
    GroupedConvolutionLayerResolver,
    SliceLayerResolver,
    PackLayerResolver,
    UnPackLayerResolver,
    PReLuLayerResolver,
    LeakyReLuLayerResolver,
    DilatedConvolutionLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeNearestNeighborLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver,
    DepthwiseConvolutionLayerResolver,
    AddNLayerResolver,
    MergedWeightsLstmLayerResolver,
    SplitWeightsLstmLayerResolver,
    FillLayerResolver,
    SSDDecoderResolver,
    CropLayerResolver,
    FusedBatchNormNormLayerResolver,
    PadLayerResolver,
    PixelShuffleLayerResolver,
    StridedSliceLayerResolver,
    PermuteLayerResolver,
    ArgMaxLayerResolver,
    ArgMinLayerResolver,
    ChannelShuffleLayerResolver,
    EluLayerResolver,
    TileLayerResolver,
    GatherLayerResolver,
    ReductionMeanLayerResolver,
    ReductionProdLayerResolver,
    ReductionSumLayerResolver,
    ReductionMinLayerResolver,
    ReductionMaxLayerResolver,
    SpaceToBatchLayerResolver,
    EltWiseUnaryAbsLayerResolver,
    EltWiseUnaryCeilLayerResolver,
    EltWiseUnaryExpLayerResolver,
    EltWiseUnaryFloorLayerResolver,
    EltWiseUnaryLogLayerResolver,
    EltWiseUnaryLogicalNotLayerResolver,
    EltWiseUnaryNegLayerResolver,
    EltWiseUnaryRoundLayerResolver,
    EltWiseUnaryRsqrtLayerResolver,
    EltWiseUnarySinLayerResolver,
    EltWiseUnarySqrtLayerResolver,
    EltWiseUnarySquareLayerResolver,
    ExtractGlimpseLayerResolver,
    ImageProjectiveTransformLayerResolver,
    CropAndResizeLayerResolver,
    MomentsLayerResolver,
    MatMulLayerResolver,
    SpaceToDepthLayerResolver,
    LogSoftmaxLayerResolver,
    TopKLayerResolver,
    IdentityNLayerResolver,
    OneHotLayerResolver,
    ConstantLayerResolver  # final resolution to add leftover static sub-graph/ops
]
"""
type: list[type(LayerResolver)]
"""

layer_builders = {
    GeLuLayerResolver.Descriptor: GeLuLayerBuilder,
    LayerNormLayerResolver.Descriptor: LayerNormLayerBuilder,
    BatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    BatchNormWithGlobalNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    BatchToSpaceLayerResolver.Descriptor: BatchToSpaceLayerBuilder,
    GenericBatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    CaffeSsdLayerResolver.Descriptor: CaffeSsdLayerBuilder,
    GatherLayerResolver.Descriptor: GatherLayerBuilder,
    CastLayerResolver.Descriptor: CastLayerBuilder,
    ConcatLayerResolver.Descriptor: ConcatLayerBuilder,
    ConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    DeconvolutionLayerResolver.Descriptor: DeconvolutionLayerBuilder,
    EltWiseAndLayerResolver.Descriptor: EltWiseAndLayerBuilder,
    EltWiseEqualLayerResolver.Descriptor: EltWiseEqualLayerBuilder,
    EltWiseFloorDivLayerResolver.Descriptor: EltWiseFloorDivLayerBuilder,
    EltWiseGreaterLayerResolver.Descriptor: EltWiseGreaterLayerBuilder,
    EltWiseGreaterEqualLayerResolver.Descriptor: EltWiseGreaterEqualLayerBuilder,
    EltWiseLessLayerResolver.Descriptor: EltWiseLessLayerBuilder,
    EltWiseLessEqualLayerResolver.Descriptor: EltWiseLessEqualLayerBuilder,
    EltWiseNotEqualLayerResolver.Descriptor: EltWiseNotEqualLayerBuilder,
    EltWiseOrLayerResolver.Descriptor: EltWiseOrLayerBuilder,
    EltWiseSelectLayerResolver.Descriptor: EltWiseSelectLayerBuilder,
    EltWiseMaxLayerResolver.Descriptor: EltWiseMaxLayerBuilder,
    EltWiseMinLayerResolver.Descriptor: EltWiseMinLayerBuilder,
    EltWiseMulLayerResolver.Descriptor: EltWiseMulLayerBuilder,
    EltWisePowLayerResolver.Descriptor: EltWisePowLayerBuilder,
    EltWiseSumLayerResolver.Descriptor: EltWiseSumLayerBuilder,
    EltWiseBiasaddLayerResolver.Descriptor: EltWiseBiasaddLayerBuilder,
    EltWiseSubLayerResolver.Descriptor: EltWiseSubLayerBuilder,
    EltWiseDivLayerResolver.Descriptor: EltWiseDivLayerBuilder,
    InstanceNormLayerResolver.Descriptor: InstanceNormLayerBuilder,
    AddNLayerResolver.Descriptor: AddNLayerBuilder,
    TileLayerResolver.Descriptor: TileLayerBuilder,
    FullyConnectedLayerResolver.Descriptor: FullyConnectedLayerBuilder,
    FakeQuantLayerResolver.Descriptor: FakeQuantLayerBuilder,
    FakeQuantPerChannelResolver.Descriptor: FakeQuantLayerBuilder,
    L2NormLayerResolver.Descriptor: L2NormLayerBuilder,
    LrnLayerResolver.Descriptor: LrnLayerBuilder,
    ReluLayerResolver.Descriptor: ReluLayerBuilder,
    Relu6LayerResolver.Descriptor: ReluMinMaxLayerBuilder,
    ReluMinMaxLayerResolver.Descriptor: ReluMinMaxLayerBuilder,
    SigmoidLayerResolver.Descriptor: SigmoidLayerBuilder,
    SoftmaxLayerResolver.Descriptor: SoftmaxLayerBuilder,
    SpaceToBatchLayerResolver.Descriptor: SpaceToBatchLayerBuilder,
    TanhLayerResolver.Descriptor: TanhLayerBuilder,
    AvgPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    MaxPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    NonMaxSuppressionLayerResolver.Descriptor: NonMaxSuppressionLayerBuilder,
    GroupedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    SliceLayerResolver.Descriptor: SliceLayerBuilder,
    PixelShuffleLayerResolver.Descriptor: PixelShuffleLayerBuilder,
    PReLuLayerResolver.Descriptor: PReLuLayerBuilder,
    LeakyReLuLayerResolver.Descriptor: PReLuLayerBuilder,
    DilatedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    DepthwiseConvolutionLayerResolver.Descriptor: DepthwiseConvolutionLayerBuilder,
    DilatedDepthwiseConvolutionLayerResolver.Descriptor: DepthwiseConvolutionLayerBuilder,
    ReshapeLayerResolver.Descriptor: ReshapeLayerBuilder,
    ResizeBilinearLayerResolver.Descriptor: ResizeLayerBuilder,
    ResizeNearestNeighborLayerResolver.Descriptor: ResizeLayerBuilder,
    SplitWeightsLstmLayerResolver.Descriptor: LstmLayerBuilder,
    MergedWeightsLstmLayerResolver.Descriptor: LstmLayerBuilder,
    IgnoredLayersResolver.Descriptor: IgnoredLayersBuilder,
    FillLayerResolver.Descriptor: FillLayerBuilder,
    SSDDecoderResolver.Descriptor: SSDDecoderLayersBuilder,
    CropLayerResolver.Descriptor: CropLayerBuilder,
    SSDNmsResolver.Descriptor: SSDNmsLayersBuilder,
    ConstantLayerResolver.Descriptor: ConstantLayerBuilder,
    FusedBatchNormNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    PackLayerResolver.Descriptor: PackLayerBuilder,
    PadLayerResolver.Descriptor: PadLayerBuilder,
    UnPackLayerResolver.Descriptor: UnpackLayerBuilder,
    StridedSliceLayerResolver.Descriptor: StridedSliceLayerBuilder,
    PermuteLayerResolver.Descriptor: PermuteLayerBuilder,
    ArgMaxLayerResolver.Descriptor: ArgMaxLayerBuilder,
    ArgMinLayerResolver.Descriptor: ArgMinLayerBuilder,
    ChannelShuffleLayerResolver.Descriptor: ChannelShuffleLayerBuilder,
    EluLayerResolver.Descriptor: EluLayerBuilder,
    ReductionMeanLayerResolver.Descriptor: ReductionMeanLayerBuilder,
    ReductionProdLayerResolver.Descriptor: ReductionProdLayerBuilder,
    ReductionSumLayerResolver.Descriptor: ReductionSumLayerBuilder,
    ReductionMinLayerResolver.Descriptor: ReductionMinLayerBuilder,
    ReductionMaxLayerResolver.Descriptor: ReductionMaxLayerBuilder,
    EltWiseUnaryAbsLayerResolver.Descriptor: EltWiseUnaryAbsLayerBuilder,
    EltWiseUnaryCeilLayerResolver.Descriptor: EltWiseUnaryCeilLayerBuilder,
    EltWiseUnaryExpLayerResolver.Descriptor: EltWiseUnaryExpLayerBuilder,
    EltWiseUnaryFloorLayerResolver.Descriptor: EltWiseUnaryFloorLayerBuilder,
    EltWiseUnaryLogLayerResolver.Descriptor: EltWiseUnaryLogLayerBuilder,
    EltWiseUnaryLogicalNotLayerResolver.Descriptor: EltWiseUnaryLogicalNotLayerBuilder,
    EltWiseUnaryNegLayerResolver.Descriptor: EltWiseUnaryNegLayerBuilder,
    EltWiseUnaryRoundLayerResolver.Descriptor: EltWiseUnaryRoundLayerBuilder,
    EltWiseUnaryRsqrtLayerResolver.Descriptor: EltWiseUnaryRsqrtLayerBuilder,
    EltWiseUnarySinLayerResolver.Descriptor: EltWiseUnarySinLayerBuilder,
    EltWiseUnarySqrtLayerResolver.Descriptor: EltWiseUnarySqrtLayerBuilder,
    EltWiseUnarySquareLayerResolver.Descriptor: EltWiseUnarySquareLayerBuilder,
    ExtractGlimpseLayerResolver.Descriptor: ExtractGlimpseLayerBuilder,
    ImageProjectiveTransformLayerResolver.Descriptor: ImageProjectiveTransformLayerBuilder,
    CropAndResizeLayerResolver.Descriptor: CropAndResizeLayerBuilder,
    MomentsLayerResolver.Descriptor: MomentsLayerBuilder,
    SpaceToDepthLayerResolver.Descriptor: SpaceToDepthLayerBuilder,
    LogSoftmaxLayerResolver.Descriptor: LogSoftmaxLayerBuilder,
    MatMulLayerResolver.Descriptor: MatMulLayerBuilder,
    TopKLayerResolver.Descriptor: TopKLayerBuilder,
    CustomLayerResolver.Descriptor: CustomLayerBuilder,
    Tf2SSDNmsResolver.Descriptor: SSDNmsLayersBuilder,
    IdentityNLayerResolver.Descriptor: IdentityNLayerBuilder,
    OneHotLayerResolver.Descriptor: OneHotLayerBuilder,
}

"""
type: dict[type(LayerDescriptor), type(LayerBuilder)]
"""
