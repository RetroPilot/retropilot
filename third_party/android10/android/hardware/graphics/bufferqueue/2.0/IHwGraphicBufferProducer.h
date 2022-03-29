#ifndef HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V2_0_IHWGRAPHICBUFFERPRODUCER_H
#define HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V2_0_IHWGRAPHICBUFFERPRODUCER_H

#include <android/hardware/graphics/bufferqueue/2.0/IGraphicBufferProducer.h>

#include <android/hardware/graphics/bufferqueue/2.0/BnHwProducerListener.h>
#include <android/hardware/graphics/bufferqueue/2.0/BpHwProducerListener.h>
#include <android/hardware/graphics/bufferqueue/2.0/hwtypes.h>
#include <android/hardware/graphics/common/1.2/hwtypes.h>
#include <android/hidl/base/1.0/BnHwBase.h>
#include <android/hidl/base/1.0/BpHwBase.h>

#include <hidl/Status.h>
#include <hwbinder/IBinder.h>
#include <hwbinder/Parcel.h>

namespace android {
namespace hardware {
namespace graphics {
namespace bufferqueue {
namespace V2_0 {
::android::status_t readEmbeddedFromParcel(
        const ::android::hardware::graphics::bufferqueue::V2_0::IGraphicBufferProducer::DequeueBufferOutput &obj,
        const ::android::hardware::Parcel &parcel,
        size_t parentHandle,
        size_t parentOffset);

::android::status_t writeEmbeddedToParcel(
        const ::android::hardware::graphics::bufferqueue::V2_0::IGraphicBufferProducer::DequeueBufferOutput &obj,
        ::android::hardware::Parcel *parcel,
        size_t parentHandle,
        size_t parentOffset);

::android::status_t readEmbeddedFromParcel(
        const ::android::hardware::graphics::bufferqueue::V2_0::IGraphicBufferProducer::QueueBufferInput &obj,
        const ::android::hardware::Parcel &parcel,
        size_t parentHandle,
        size_t parentOffset);

::android::status_t writeEmbeddedToParcel(
        const ::android::hardware::graphics::bufferqueue::V2_0::IGraphicBufferProducer::QueueBufferInput &obj,
        ::android::hardware::Parcel *parcel,
        size_t parentHandle,
        size_t parentOffset);

}  // namespace V2_0
}  // namespace bufferqueue
}  // namespace graphics
}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V2_0_IHWGRAPHICBUFFERPRODUCER_H
