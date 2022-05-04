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

#include <unistd.h>

#include <cassert>
#include <cstring>

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/camerad/cameras/camera_common.h"

#define FRAME_BUF_COUNT 4

struct ImageFormat {
  int32_t width;
  int32_t height;
};

/**
 * A helper class to assist image size comparison, by comparing the absolute
 * size
 * regardless of the portrait or landscape mode.
 */
class DisplayDimension {
 public:
  DisplayDimension(int32_t w, int32_t h) : w_(w), h_(h), portrait_(false) {
    if (h > w) {
      // make it landscape
      w_ = h;
      h_ = w;
      portrait_ = true;
    }
  }

  DisplayDimension(const DisplayDimension& other) {
    w_ = other.w_;
    h_ = other.h_;
    portrait_ = other.portrait_;
  }

  DisplayDimension() {
    w_ = 0;
    h_ = 0;
    portrait_ = false;
  }

  DisplayDimension& operator=(const DisplayDimension& other) {
    w_ = other.w_;
    h_ = other.h_;
    portrait_ = other.portrait_;

    return (*this);
  }

  bool IsSameRatio(DisplayDimension& other) {
    return (w_ * other.h_ == h_ * other.w_);
  }
  bool operator>(DisplayDimension& other) {
    return (w_ >= other.w_) || (h_ >= other.h_);
  }
  bool operator==(DisplayDimension& other) {
    return w_ == other.w_ && h_ == other.h_ && portrait_ == other.portrait_;
  }
  DisplayDimension operator-(DisplayDimension& other) {
    return DisplayDimension(w_ - other.w_, h_ - other.h_);
  }

  void Flip() { portrait_ = !portrait_; }
  bool IsPortrait() { return portrait_; }
  int32_t width() { return w_; }
  int32_t height() { return h_; }
  int32_t org_width() { return (portrait_ ? h_ : w_); }
  int32_t org_height() { return (portrait_ ? w_ : h_); }

private:
  int32_t w_, h_;
  bool portrait_;
};

class CameraState {
public:
  CameraInfo ci;
  int camera_num;
  unsigned int fps;
  float digital_gain;
  CameraBuf buf;

private:
  MultiCameraState *multi_cam_state;

  ACameraDevice *camera_device;
  int32_t camera_orientation;
  const char *camera_id;

  ImageFormat view{0, 0};
  AImageReader *yuv_reader;
  ANativeWindow *yuv_window;

  ACaptureSessionOutputContainer *capture_session_output_container;
  ACaptureRequest *capture_request;
  ACaptureSessionOutput *capture_session_output;
  ACameraOutputTarget *camera_output_target;
  ACameraCaptureSession *capture_session;

public:
  void camera_init(MultiCameraState *multi_cam_state_, VisionIpcServer *v, int camera_index, int camera_id_, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type);
  void camera_open();
  void camera_run(float *ts);
  void camera_close();

  void match_camera_size(ImageFormat *view, int32_t width, int32_t height, enum AIMAGE_FORMATS desired_format);
  void create_session(ANativeWindow *window, ACameraDevice *device);
  void start_preview(bool start);

  ACameraDevice_StateCallbacks *get_device_listener();
  ACameraCaptureSession_stateCallbacks *get_session_listener();
};


typedef struct MultiCameraState {
  ACameraManager *camera_manager;

  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
