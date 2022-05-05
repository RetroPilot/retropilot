#pragma once

#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImageReader.h>
#include <android/native_window.h>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/camerad/cameras/android/image_format.h"

class NativeCamera {
private:
  uint32_t camera_orientation;

  /** camera device **/
  ACameraManager *camera_manager;
  ACameraIdList *camera_id_list;
  const char *camera_id;
  ACameraDevice *camera_device;

  /** capture session **/
  ACaptureSessionOutputContainer *capture_session_output_container;
  ACaptureRequest *capture_request;
  ACaptureSessionOutput *capture_session_output;
  ACameraOutputTarget *camera_output_target;
  ACameraCaptureSession *capture_session;

public:
  explicit NativeCamera(int camera_index);

  ~NativeCamera();

  void match_capture_size_request(ImageFormat *view, int32_t width, int32_t height, enum AIMAGE_FORMATS desired_format = AIMAGE_FORMAT_YUV_420_888);

  void create_capture_session(ANativeWindow *window);

  void start_preview(bool start);

  int32_t get_camera_count() { return camera_id_list->numCameras; }
  uint32_t get_orientation() { return camera_orientation; }

private:
  ACameraDevice_StateCallbacks *get_device_listener();
  ACameraCaptureSession_stateCallbacks *get_session_listener();
};
