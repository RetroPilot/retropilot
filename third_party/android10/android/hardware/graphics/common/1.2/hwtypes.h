#ifndef HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_COMMON_V1_2_HWTYPES_H
#define HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_COMMON_V1_2_HWTYPES_H

#include <android/hardware/graphics/common/1.2/types.h>

#include <android/hardware/graphics/common/1.0/hwtypes.h>
#include <android/hardware/graphics/common/1.1/hwtypes.h>

#include <hidl/Status.h>
#include <hwbinder/IBinder.h>
#include <hwbinder/Parcel.h>

namespace android {
namespace hardware {
namespace graphics {
namespace common {
namespace V1_2 {
::android::status_t readEmbeddedFromParcel(
        const ::android::hardware::graphics::common::V1_2::HardwareBuffer &obj,
        const ::android::hardware::Parcel &parcel,
        size_t parentHandle,
        size_t parentOffset);

::android::status_t writeEmbeddedToParcel(
        const ::android::hardware::graphics::common::V1_2::HardwareBuffer &obj,
        ::android::hardware::Parcel *parcel,
        size_t parentHandle,
        size_t parentOffset);

}  // namespace V1_2
}  // namespace common
}  // namespace graphics
}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_COMMON_V1_2_HWTYPES_H
