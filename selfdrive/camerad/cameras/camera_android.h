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

  ACameraManager *camera_manager;
  int32_t camera_orientation;
  ACameraIdList *camera_id_list;
  const char *camera_id;
  bool camera_ready;
} CameraState;


typedef struct MultiCameraState {
  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm;
  PubMaster *pm;
} MultiCameraState;
