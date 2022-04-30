#pragma once

#include <android/native_window.h>
#include <camera/NdkCameraCaptureSession.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraError.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadata.h>
#include <camera/NdkCameraMetadataTags.h>
#include <camera/NdkCaptureRequest.h>
#include <media/NdkImageReader.h>

#include <string>

#include "selfdrive/camerad/cameras/camera_common.h"

#define FRAME_BUF_COUNT 16

// Camera Callbacks
static void CameraDeviceOnDisconnected(void* context, ACameraDevice* device) {
  LOG("Camera(id: %s) is diconnected.\n", ACameraDevice_getId(device));
}

static void CameraDeviceOnError(void* context, ACameraDevice* device, int error) {
  LOGE("Error(code: %d) on Camera(id: %s).\n", error, ACameraDevice_getId(device));
}

// Capture Callbacks
static void CaptureSessionOnReady(void* context, ACameraCaptureSession* session) {
  LOG("Session is ready.\n");
}

static void CaptureSessionOnActive(void* context, ACameraCaptureSession* session) {
  LOG("Session is active.\n");
}

struct ImageFormat {
  uint32_t width;
  uint32_t height;
};

class DisplayDimension {
public:
  DisplayDimension(uint32_t w, uint32_t h) : w_(w), h_(h), portrait_(false) {
    if (h > w) {
      // make it landscape
      w_ = h;
      h_ = w;
      portrait_ = true;
    }
  }

  DisplayDimension(const DisplayDimension& other) : w_(other.w_), h_(other.h_), portrait_(other.portrait_) {}

  DisplayDimension() : w_(0), h_(0), portrait_(false) {}

  DisplayDimension& operator=(const DisplayDimension& other) {
    w_ = other.w_;
    h_ = other.h_;
    portrait_ = other.portrait_;
    return *this;
  }

  bool operator>(const DisplayDimension& other) const {
    return (w_ > other.w_) || (h_ > other.h_);
  }
  bool operator==(const DisplayDimension& other) const {
    return (w_ == other.w_) && (h_ == other.h_) && (portrait_ == other.portrait_);
  }
  DisplayDimension operator-(const DisplayDimension& other) const {
    return DisplayDimension(w_ - other.w_, h_ - other.h_);
  }

  bool IsSameRatio(const DisplayDimension& other) const {
    return (w_ * other.h_) == (h_ * other.w_);
  }

  void Flip() {
    portrait_ = !portrait_;
  }
  bool IsPortrait() const {
    return portrait_;
  }
  uint32_t width() const {
    return w_;
  }
  uint32_t height() const {
    return h_;
  }
  uint32_t org_width() const {
    return portrait_ ? h_ : w_;
  }
  uint32_t org_height() const {
    return portrait_ ? w_ : h_;
  }

private:
    int32_t w_, h_;
    bool portrait_;
};

typedef struct CameraState {
  CameraInfo ci;
  int camera_num;
  int fps;
  float digital_gain;
  CameraBuf buf;

  ACameraDevice *camera_device;
  ACaptureRequest *capture_request;
  ACameraOutputTarget *camera_output_target;
  ACaptureSessionOutput *capture_session_output;
  ACaptureSessionOutputContainer *capture_session_output_container;
  ACameraCaptureSession *capture_session;

  ACameraDevice_StateCallbacks device_state_callbacks;
  ACameraCaptureSession_stateCallbacks capture_session_state_callbacks;

  int32_t camera_orientation;
  // android camera id
  const char *camera_id;
} CameraState;


typedef struct MultiCameraState {
  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
