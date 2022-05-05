#include "selfdrive/camerad/cameras/android/native_camera.h"

#include <cassert>
#include <string>

#include "selfdrive/camerad/cameras/android/display_dimension.h"

NativeCamera::NativeCamera(int camera_index) {
  camera_manager = ACameraManager_create();

  camera_status_t status = ACameraManager_getCameraIdList(camera_manager, &camera_id_list);
  assert(status == ACAMERA_OK && camera_id_list);
  assert(camera_id_list->numCameras > 0);

  camera_id = camera_id_list->cameraIds[camera_index];

  ACameraMetadata *metadata;
  status = ACameraManager_getCameraCharacteristics(camera_manager, camera_id, &metadata);
  assert(status == ACAMERA_OK && metadata);
  ACameraMetadata_free(metadata);

  status = ACameraManager_openCamera(camera_manager, camera_id,
                                     get_device_listener(), &camera_device);
  assert(status == ACAMERA_OK && camera_device);
}

NativeCamera::~NativeCamera() {
  if (capture_request) {
    ACaptureRequest_free(capture_request);
  }

  if (camera_output_target) {
    ACameraOutputTarget_free(camera_output_target);
  }

  if (camera_device) {
    ACameraDevice_close(camera_device);
  }

  ACaptureSessionOutputContainer_remove(capture_session_output_container,
                                        capture_session_output);

  if (capture_session_output) {
    ACaptureSessionOutput_free(capture_session_output);
  }

  if (capture_session_output_container) {
    ACaptureSessionOutputContainer_free(capture_session_output_container);
  }

  ACameraManager_deleteCameraIdList(camera_id_list);
  ACameraManager_delete(camera_manager);
}

void NativeCamera::match_capture_size_request(ImageFormat *view, int32_t width, int32_t height, enum AIMAGE_FORMATS desired_format) {
  LOGD("match_camera_size: w=%d, h=%d, format=0x%X", width, height, desired_format);

  DisplayDimension disp(width, height);

  ACameraMetadata *metadata = nullptr;
  ACameraManager_getCameraCharacteristics(camera_manager, camera_id, &metadata);
  ACameraMetadata_const_entry entry;
  ACameraMetadata_getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &entry);
  // format of the data: format, width, height, input?, type int32

  bool foundIt = false;
  DisplayDimension foundRes(1920, 1080);

  for (int i = 0; i < entry.count; i++) {
    int32_t input = entry.data.i32[i * 4 + 3];
    int32_t format = entry.data.i32[i * 4 + 0];

    if (input) continue;

    DisplayDimension res(entry.data.i32[i * 4 + 1],
                         entry.data.i32[i * 4 + 2]);

#if false
    if (format) {
      LOGD("found format 0x%X, w: %d, h: %d", format, res.width(), res.height());
    }
#endif

    if (format == desired_format && width == res.width() && height == res.height()) {
      foundIt = true;
      foundRes = res;
      break;
    }
  }

  if (foundIt) {
    view->width = foundRes.org_width();
    view->height = foundRes.org_height();
    LOGD("found width=%d, height=%d", view->width, view->height);
  } else {
    LOGW("could not find a matching resolution");
  }
}

void NativeCamera::create_capture_session(ANativeWindow *window) {
  LOGD("NativeCamera::create_capture_session");

  camera_status_t status;

  // create output container
  status = ACaptureSessionOutputContainer_create(&capture_session_output_container);
  assert(status == ACAMERA_OK);

  // create output to native window
  ANativeWindow_acquire(window);
  status = ACaptureSessionOutput_create(window, &capture_session_output);
  assert(status == ACAMERA_OK);
  status = ACaptureSessionOutputContainer_add(capture_session_output_container, capture_session_output);
  assert(status == ACAMERA_OK);
  status = ACameraOutputTarget_create(window, &camera_output_target);
  assert(status == ACAMERA_OK);

  // create capture request and add output target to it
  status = ACameraDevice_createCaptureRequest(camera_device, TEMPLATE_RECORD, &capture_request);
  assert(status == ACAMERA_OK);
  status = ACaptureRequest_addTarget(capture_request, camera_output_target);
  assert(status == ACAMERA_OK);

  // create capture session
  status = ACameraDevice_createCaptureSession(camera_device, capture_session_output_container,
                                              get_session_listener(), &capture_session);
  assert(status == ACAMERA_OK);
  LOGD("create_capture_session: created capture session");

  // TODO: manual mode
}

void NativeCamera::start_preview(bool start) {
  if (start) {
    camera_status_t status = ACameraCaptureSession_setRepeatingRequest(capture_session, nullptr, 1,
                                                                       &capture_request, nullptr);
    assert(status == ACAMERA_OK);
  } else {
    camera_status_t status = ACameraCaptureSession_stopRepeating(capture_session);
    assert(status == ACAMERA_OK);
  }
}

// ** CameraDevice callbacks **

static void OnDeviceDisconnect(void* /* ctx */, ACameraDevice* dev) {
  std::string id(ACameraDevice_getId(dev));
  LOGW("Device \"%s\" disconnected", id.c_str());
}

static void OnDeviceError(void* /* ctx */, ACameraDevice* dev, int err) {
  std::string id(ACameraDevice_getId(dev));
  LOGE("Camera Device Error: %#x, Device \"%s\"", err, id.c_str());

  switch (err) {
    case ERROR_CAMERA_IN_USE:
      LOGE("Camera in use");
      break;
    case ERROR_CAMERA_SERVICE:
      LOGE("Fatal Error occured in Camera Service");
      break;
    case ERROR_CAMERA_DEVICE:
      LOGE("Fatal Error occured in Camera Device");
      break;
    case ERROR_CAMERA_DISABLED:
      LOGE("Camera disabled");
      break;
    case ERROR_MAX_CAMERAS_IN_USE:
      LOGE("System limit for maximum concurrent cameras used was exceeded");
      break;
    default:
      LOGE("Unknown Camera Device Error: %#x", err);
  }
}

ACameraDevice_StateCallbacks *NativeCamera::get_device_listener() {
  static ACameraDevice_stateCallbacks device_listener = {
    .context = this,
    .onDisconnected = ::OnDeviceDisconnect,
    .onError = ::OnDeviceError,
  };
  return &device_listener;
}

// ** CaptureSession callbacks **

void OnSessionClosed(void *context, ACameraCaptureSession *session) {
  LOGW("session %p closed", session);
}

void OnSessionReady(void *context, ACameraCaptureSession *session) {
  LOGD("session %p ready", session);
}

void OnSessionActive(void *context, ACameraCaptureSession *session) {
  LOGD("session %p active", session);
}

ACameraCaptureSession_stateCallbacks *NativeCamera::get_session_listener() {
  static ACameraCaptureSession_stateCallbacks session_listener = {
    .context = this,
    .onActive = ::OnSessionActive,
    .onReady = ::OnSessionReady,
    .onClosed = ::OnSessionClosed,
  };
  return &session_listener;
}
