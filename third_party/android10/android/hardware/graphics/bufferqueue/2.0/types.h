#ifndef HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V2_0_TYPES_H
#define HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V2_0_TYPES_H

#include <hidl/HidlSupport.h>
#include <hidl/MQDescriptor.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>

namespace android {
namespace hardware {
namespace graphics {
namespace bufferqueue {
namespace V2_0 {

// Forward declaration for forward reference support:
enum class Status : int32_t;
enum class SlotIndex : int32_t;
enum class ConnectionType : int32_t;

/**
 * Possible return values from a function call.
 */
enum class Status : int32_t {
    /**
     * The call succeeds.
     */
    OK = 0,
    /**
     * The function fails allocate memory.
     */
    NO_MEMORY = -12 /* (-12) */,
    /**
     * The buffer queue has been abandoned, no consumer is connected, or no
     * producer is connected at the time of the call.
     */
    NO_INIT = -19 /* (-19) */,
    /**
     * Some of the provided input arguments are invalid.
     */
    BAD_VALUE = -22 /* (-22) */,
    /**
     * An unexpected death of some object prevents the operation from
     * continuing.
     * 
     * @note This status value is different from a transaction failure, which
     * should be detected by isOk().
     */
    DEAD_OBJECT = -32 /* (-32) */,
    /**
     * The internal state of the buffer queue does not permit the operation.
     */
    INVALID_OPERATION = -38 /* (-38) */,
    /**
     * The call fails to finish within the specified time limit.
     */
    TIMED_OUT = -110 /* (-110) */,
    /**
     * The buffer queue is operating in a non-blocking mode, but the call cannot
     * be completed without blocking.
     */
    WOULD_BLOCK = -5 /* 0xfffffffb */,
    /**
     * The call fails because of a reason not listed above.
     */
    UNKNOWN_ERROR = -1 /* 0xffffffff */,
};

/**
 * Special values for a slot index.
 */
enum class SlotIndex : int32_t {
    /**
     * Invalid/unspecified slot index. This may be returned from a function that
     * returns a slot index if the call is unsuccessful.
     */
    INVALID = -1 /* (-1) */,
    UNSPECIFIED = -1 /* (-1) */,
};

/**
 * An "empty" fence can be an empty handle (containing no fds and no ints) or a
 * fence with one fd that is equal to -1 and no ints.
 * 
 * A valid fence is an empty fence or a native handle with exactly one fd and no
 * ints.
 */
typedef ::android::hardware::hidl_handle Fence;

/**
 * How buffers shall be produced. One of these values must be provided in a call
 * to IGraphicBufferProducer::connect() and
 * IGraphicBufferProducer::disconnect().
 */
enum class ConnectionType : int32_t {
    /**
     * This value can be used only as an input to
     * IGraphicBufferProducer::disconnect().
     */
    CURRENTLY_CONNECTED = -1 /* (-1) */,
    /**
     * Buffers shall be queued by EGL via `eglSwapBuffers()` after being filled
     * using OpenGL ES.
     */
    EGL = 1,
    /**
     * Buffers shall be queued after being filled using the CPU.
     */
    CPU = 2,
    /**
     * Buffers shall be queued by Stagefright after being filled by a video
     * decoder. The video decoder can either be a software or hardware decoder.
     */
    MEDIA = 3,
    /**
     * Buffers shall be queued by the camera HAL.
     */
    CAMERA = 4,
};

//
// type declarations for package
//

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::graphics::bufferqueue::V2_0::Status o);

constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V2_0::Status lhs, const ::android::hardware::graphics::bufferqueue::V2_0::Status rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V2_0::Status rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V2_0::Status lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V2_0::Status lhs, const ::android::hardware::graphics::bufferqueue::V2_0::Status rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V2_0::Status rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V2_0::Status lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V2_0::Status e) {
    v |= static_cast<int32_t>(e);
    return v;
}
constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V2_0::Status e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::graphics::bufferqueue::V2_0::SlotIndex o);

constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex lhs, const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex lhs, const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex e) {
    v |= static_cast<int32_t>(e);
    return v;
}
constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::graphics::bufferqueue::V2_0::ConnectionType o);

constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType lhs, const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType lhs, const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
constexpr int32_t &operator|=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType e) {
    v |= static_cast<int32_t>(e);
    return v;
}
constexpr int32_t &operator&=(int32_t& v, const ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType e) {
    v &= static_cast<int32_t>(e);
    return v;
}

//
// type header definitions for package
//

template<>
inline std::string toString<::android::hardware::graphics::bufferqueue::V2_0::Status>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::bufferqueue::V2_0::Status> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::Status::OK) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::Status::OK)) {
        os += (first ? "" : " | ");
        os += "OK";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::Status::OK;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::Status::NO_MEMORY) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::Status::NO_MEMORY)) {
        os += (first ? "" : " | ");
        os += "NO_MEMORY";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::Status::NO_MEMORY;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::Status::NO_INIT) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::Status::NO_INIT)) {
        os += (first ? "" : " | ");
        os += "NO_INIT";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::Status::NO_INIT;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::Status::BAD_VALUE) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::Status::BAD_VALUE)) {
        os += (first ? "" : " | ");
        os += "BAD_VALUE";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::Status::BAD_VALUE;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::Status::DEAD_OBJECT) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::Status::DEAD_OBJECT)) {
        os += (first ? "" : " | ");
        os += "DEAD_OBJECT";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::Status::DEAD_OBJECT;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::Status::INVALID_OPERATION) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::Status::INVALID_OPERATION)) {
        os += (first ? "" : " | ");
        os += "INVALID_OPERATION";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::Status::INVALID_OPERATION;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::Status::TIMED_OUT) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::Status::TIMED_OUT)) {
        os += (first ? "" : " | ");
        os += "TIMED_OUT";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::Status::TIMED_OUT;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::Status::WOULD_BLOCK) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::Status::WOULD_BLOCK)) {
        os += (first ? "" : " | ");
        os += "WOULD_BLOCK";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::Status::WOULD_BLOCK;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::Status::UNKNOWN_ERROR) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::Status::UNKNOWN_ERROR)) {
        os += (first ? "" : " | ");
        os += "UNKNOWN_ERROR";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::Status::UNKNOWN_ERROR;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::bufferqueue::V2_0::Status o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::Status::OK) {
        return "OK";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::Status::NO_MEMORY) {
        return "NO_MEMORY";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::Status::NO_INIT) {
        return "NO_INIT";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::Status::BAD_VALUE) {
        return "BAD_VALUE";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::Status::DEAD_OBJECT) {
        return "DEAD_OBJECT";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::Status::INVALID_OPERATION) {
        return "INVALID_OPERATION";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::Status::TIMED_OUT) {
        return "TIMED_OUT";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::Status::WOULD_BLOCK) {
        return "WOULD_BLOCK";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::Status::UNKNOWN_ERROR) {
        return "UNKNOWN_ERROR";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

template<>
inline std::string toString<::android::hardware::graphics::bufferqueue::V2_0::SlotIndex>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::bufferqueue::V2_0::SlotIndex> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::INVALID) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::INVALID)) {
        os += (first ? "" : " | ");
        os += "INVALID";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::INVALID;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::UNSPECIFIED) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::UNSPECIFIED)) {
        os += (first ? "" : " | ");
        os += "UNSPECIFIED";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::UNSPECIFIED;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::bufferqueue::V2_0::SlotIndex o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::INVALID) {
        return "INVALID";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::UNSPECIFIED) {
        return "UNSPECIFIED";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

template<>
inline std::string toString<::android::hardware::graphics::bufferqueue::V2_0::ConnectionType>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::graphics::bufferqueue::V2_0::ConnectionType> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CURRENTLY_CONNECTED) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CURRENTLY_CONNECTED)) {
        os += (first ? "" : " | ");
        os += "CURRENTLY_CONNECTED";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CURRENTLY_CONNECTED;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::EGL) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::EGL)) {
        os += (first ? "" : " | ");
        os += "EGL";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::EGL;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CPU) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CPU)) {
        os += (first ? "" : " | ");
        os += "CPU";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CPU;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::MEDIA) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::MEDIA)) {
        os += (first ? "" : " | ");
        os += "MEDIA";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::MEDIA;
    }
    if ((o & ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CAMERA) == static_cast<int32_t>(::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CAMERA)) {
        os += (first ? "" : " | ");
        os += "CAMERA";
        first = false;
        flipped |= ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CAMERA;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::graphics::bufferqueue::V2_0::ConnectionType o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CURRENTLY_CONNECTED) {
        return "CURRENTLY_CONNECTED";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::EGL) {
        return "EGL";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CPU) {
        return "CPU";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::MEDIA) {
        return "MEDIA";
    }
    if (o == ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CAMERA) {
        return "CAMERA";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}


}  // namespace V2_0
}  // namespace bufferqueue
}  // namespace graphics
}  // namespace hardware
}  // namespace android

//
// global type declarations for package
//

namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::graphics::bufferqueue::V2_0::Status, 9> hidl_enum_values<::android::hardware::graphics::bufferqueue::V2_0::Status> = {
    ::android::hardware::graphics::bufferqueue::V2_0::Status::OK,
    ::android::hardware::graphics::bufferqueue::V2_0::Status::NO_MEMORY,
    ::android::hardware::graphics::bufferqueue::V2_0::Status::NO_INIT,
    ::android::hardware::graphics::bufferqueue::V2_0::Status::BAD_VALUE,
    ::android::hardware::graphics::bufferqueue::V2_0::Status::DEAD_OBJECT,
    ::android::hardware::graphics::bufferqueue::V2_0::Status::INVALID_OPERATION,
    ::android::hardware::graphics::bufferqueue::V2_0::Status::TIMED_OUT,
    ::android::hardware::graphics::bufferqueue::V2_0::Status::WOULD_BLOCK,
    ::android::hardware::graphics::bufferqueue::V2_0::Status::UNKNOWN_ERROR,
};
}  // namespace details
}  // namespace hardware
}  // namespace android

namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::graphics::bufferqueue::V2_0::SlotIndex, 2> hidl_enum_values<::android::hardware::graphics::bufferqueue::V2_0::SlotIndex> = {
    ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::INVALID,
    ::android::hardware::graphics::bufferqueue::V2_0::SlotIndex::UNSPECIFIED,
};
}  // namespace details
}  // namespace hardware
}  // namespace android

namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::graphics::bufferqueue::V2_0::ConnectionType, 5> hidl_enum_values<::android::hardware::graphics::bufferqueue::V2_0::ConnectionType> = {
    ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CURRENTLY_CONNECTED,
    ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::EGL,
    ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CPU,
    ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::MEDIA,
    ::android::hardware::graphics::bufferqueue::V2_0::ConnectionType::CAMERA,
};
}  // namespace details
}  // namespace hardware
}  // namespace android


#endif  // HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V2_0_TYPES_H
