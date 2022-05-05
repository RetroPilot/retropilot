#pragma once

#include <android/native_window.h>
#include <camera/NdkCameraCaptureSession.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCaptureRequest.h>
#include <media/NdkImageReader.h>

#include <unistd.h>

#include <cassert>
#include <cstring>

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/camerad/cameras/android/image_reader.h"
#include "selfdrive/camerad/cameras/android/native_camera.h"

#define FRAME_BUF_COUNT 4

class CameraState {
public:
  CameraInfo ci;
  int camera_num;
  unsigned int fps;
  float digital_gain;
  CameraBuf buf;

private:
  MultiCameraState *multi_cam_state;

  ImageFormat view{0, 0};
  NativeCamera *native_camera;
  ImageReader *image_reader;

public:
  void camera_init(MultiCameraState *multi_cam_state_, VisionIpcServer *v, int camera_index, int camera_id_, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type);
  void camera_open();
  void camera_run(float *ts);
  void camera_close();
};


typedef struct MultiCameraState {
  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
