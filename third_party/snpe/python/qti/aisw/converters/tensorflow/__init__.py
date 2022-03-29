# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except AttributeError:
    tf_compat_v1 = tf

    # import contrib ops since they are not imported as part of TF by default
    import tensorflow.contrib
