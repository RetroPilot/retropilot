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
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/camerad/cameras/android/image_reader.h"

#define FRAME_BUF_COUNT 16

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
  ACaptureRequest *capture_request;
  ACameraOutputTarget *camera_output_target;
  ACaptureSessionOutput *capture_session_output;
  ACaptureSessionOutputContainer *capture_session_output_container;
  ACameraCaptureSession *capture_session;

  int32_t camera_orientation;
  const char *camera_id;

  ImageFormat *image_format;
  ImageReader *image_reader;

public:
  void camera_init(MultiCameraState *multi_cam_state_, VisionIpcServer *v, int camera_index, int camera_id_, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type);
  void camera_open();
  void camera_run(float *ts);
  void camera_close();

  ACameraDevice_StateCallbacks *get_device_listener();
  ACameraCaptureSession_stateCallbacks *get_capture_session_listener();

  // ** Camera Callbacks **
  static void CameraDeviceOnDisconnected(void *context, ACameraDevice *device);
  static void CameraDeviceOnError(void *context, ACameraDevice *device, int error);

  // ** Capture Callbacks **
  static void CaptureSessionOnReady(void *context, ACameraCaptureSession *session);
  static void CaptureSessionOnActive(void *context, ACameraCaptureSession *session);
  static void CaptureSessionOnClosed(void *context, ACameraCaptureSession *session);
};


typedef struct MultiCameraState {
  ACameraManager *camera_manager;

  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
