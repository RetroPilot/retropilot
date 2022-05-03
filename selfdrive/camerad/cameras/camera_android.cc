#include "selfdrive/camerad/cameras/camera_android.h"

#include "selfdrive/common/swaglog.h"

// id of the video capturing device
const int ROAD_CAMERA_INDEX = util::getenv("ROADCAM_ID", 0);
const int DRIVER_CAMERA_INDEX = util::getenv("DRIVERCAM_ID", 1);

#define FRAME_WIDTH  1280
#define FRAME_HEIGHT 720
#define FRAME_WIDTH_FRONT  1280
#define FRAME_HEIGHT_FRONT 720

extern ExitHandler do_exit;

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  // road facing
  [CAMERA_ID_IMX363] = {
      .frame_width = FRAME_WIDTH,
      .frame_height = FRAME_HEIGHT,
      .frame_stride = FRAME_WIDTH*3,
      .bayer = false,
      .bayer_flip = false,
  },
  // driver facing
  [CAMERA_ID_IMX355] = {
      .frame_width = FRAME_WIDTH_FRONT,
      .frame_height = FRAME_HEIGHT_FRONT,
      .frame_stride = FRAME_WIDTH_FRONT*3,
      .bayer = false,
      .bayer_flip = false,
  },
};

void CameraState::camera_init(MultiCameraState *multi_cam_state_, VisionIpcServer *v, int camera_index, int camera_id_, unsigned int fps_, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type) {
  LOGD("camera_init: camera_index %d, camera_id_ %d", camera_index, camera_id_);
  multi_cam_state = multi_cam_state_;
  assert(camera_id_ < std::size(cameras_supported));
  ci = cameras_supported[camera_id_];
  assert(ci.frame_width != 0);

  camera_num = camera_index;
  fps = fps_;
  // TODO: fix me
  buf.init(device_id, ctx, this, v, FRAME_BUF_COUNT, rgb_type, yuv_type);

  // ASSUMPTION: IXM363 (road) is index[0] and IMX355 (driver) is index[1]
  // TODO: check that we actually need to rotate
  if (camera_id_ == CAMERA_ID_IMX363) {
    camera_orientation = 90;
  } else if (camera_id_ == CAMERA_ID_IMX355) {
    camera_orientation = 270;
  }

  // ** get android camera id **
  ACameraManager *camera_manager = multi_cam_state->camera_manager;
  ACameraIdList *camera_id_list = NULL;

  LOGD("camera_init: getting camera list");
  camera_status_t camera_status = ACameraManager_getCameraIdList(camera_manager, &camera_id_list);
  assert(camera_status == ACAMERA_OK); // failed to get camera id list

  LOGD("camera_init: found %d cameras", camera_id_list->numCameras);
  assert(camera_id_list->numCameras > 0); // no cameras found

  camera_id = camera_id_list->cameraIds[camera_index];

  // TODO: after we figure out how to copy the camera_id
  // ACameraManager_deleteCameraIdList(camera_id_list);

  // ** create image reader **
  image_format = new ImageFormat();
  image_format->width = ci.frame_width;
  image_format->height = ci.frame_height;
  image_format->format = AIMAGE_FORMAT_YUV_420_888;
  image_reader = new ImageReader(image_format, AIMAGE_FORMAT_YUV_420_888);
}

void CameraState::camera_open() {
  LOGD("camera_open camera_num=%d camera_id=\"%s\"", camera_num, camera_id);

  ACameraManager *camera_manager = multi_cam_state->camera_manager;

  camera_status_t status = ACameraManager_openCamera(camera_manager, camera_id,
                                                     get_device_listener(), &camera_device);
  LOGD("camera_open: open camera_id=\"%s\" status=%d", camera_id, status);
  assert(status == ACAMERA_OK);

  ANativeWindow *window = image_reader->GetNativeWindow();

  status = ACaptureSessionOutputContainer_create(&capture_session_output_container);
  assert(status == ACAMERA_OK);
  ANativeWindow_acquire(window);
  status = ACaptureSessionOutput_create(window, &capture_session_output);
  assert(status == ACAMERA_OK);
  status = ACaptureSessionOutputContainer_add(capture_session_output_container,
                                              capture_session_output);
  assert(status == ACAMERA_OK);
  status = ACameraOutputTarget_create(window, &camera_output_target);
  assert(status == ACAMERA_OK);

  // use TEMPLATE_RECORD for good quality and OK frame rate
  status = ACameraDevice_createCaptureRequest(camera_device, TEMPLATE_RECORD, &capture_request);
  assert(status == ACAMERA_OK); // failed to create preview capture request

  status = ACaptureRequest_addTarget(capture_request, camera_output_target);
  assert(status == ACAMERA_OK);

  status = ACameraDevice_createCaptureSession(camera_device, capture_session_output_container,
                                              get_capture_session_listener(), &capture_session);
  assert(status == ACAMERA_OK);

  status = ACameraCaptureSession_setRepeatingRequest(capture_session, NULL, 1, &capture_request, NULL);
  assert(status == ACAMERA_OK);
}

void CameraState::camera_run(float *ts) {
  LOGD("camera_run %d", camera_num);

  // TODO: implement transform
  // cv::Size size(ci.frame_width, ci.frame_height);
  // const cv::Mat transform = cv::Mat(3, 3, CV_32F, ts);

  uint32_t frame_id = 0;
  size_t buf_idx = 0;

  while (!do_exit) {
    // ** get image **
    AImage *image = image_reader->GetLatestImage();
    if (image == NULL) continue;
    LOGD("camera_run: image=%p", image);

    // ** debug **
    int32_t format = -1;
    AImage_getFormat(image, &format);
    assert(format == AIMAGE_FORMAT_YUV_420_888);

    int32_t planes = 0;
    AImage_getNumberOfPlanes(image, &planes);
    assert(planes == 3);

    // cv::warpPerspective(frame_mat, transformed_mat, transform, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    buf.camera_bufs_metadata[buf_idx] = {.frame_id = frame_id};

    // ** copy image data to cl buffer **
    uint8_t *data = NULL;
    int size = 0;
    media_status_t status = AImage_getPlaneData(image, 0, &data, &size);
    assert(status == AMEDIA_OK);  // failed to get image data

    auto &buffer = buf.camera_bufs[buf_idx];
    LOGD("camera_run: clEnqueueWriteBuffer size=%d", size);
    CL_CHECK(clEnqueueWriteBuffer(buffer.copy_q, buffer.buf_cl, CL_TRUE, 0, size, data, 0, NULL, NULL));
    LOGD("camera_run: clEnqueueWriteBuffer done");

    buf.queue(buf_idx);

    ++frame_id;
    buf_idx = (buf_idx + 1) % FRAME_BUF_COUNT;
  }
}

void CameraState::camera_close() {
  LOGD("camera_close %d", camera_num);

  if (capture_request) {
    ACaptureRequest_free(capture_request);
    capture_request = NULL;
  }

  if (camera_output_target) {
    ACameraOutputTarget_free(camera_output_target);
    camera_output_target = NULL;
  }

  if (camera_device) {
    ACameraDevice_close(camera_device);
    camera_device = NULL;
  }

  if (capture_session_output_container) {
    if (capture_session_output) {
      ACaptureSessionOutputContainer_remove(capture_session_output_container,
                                            capture_session_output);

      ACaptureSessionOutput_free(capture_session_output);
      capture_session_output = NULL;
    }

    ACaptureSessionOutputContainer_free(capture_session_output_container);
    capture_session_output_container = NULL;
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

ACameraDevice_StateCallbacks *CameraState::get_device_listener() {
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

ACameraCaptureSession_stateCallbacks *CameraState::get_capture_session_listener() {
  static ACameraCaptureSession_stateCallbacks session_listener = {
    .context = this,
    .onActive = ::OnSessionActive,
    .onReady = ::OnSessionReady,
    .onClosed = ::OnSessionClosed,
  };
  return &session_listener;
}

static void road_camera_thread(CameraState *s) {
  util::set_thread_name("android_road_camera_thread");

  // transforms calculation see tools/webcam/warp_vis.py
  float ts[9] = {1.50330396, 0.0, -59.40969163,
                  0.0, 1.50330396, 76.20704846,
                  0.0, 0.0, 1.0};
  // if camera upside down:
  // float ts[9] = {-1.50330396, 0.0, 1223.4,
  //                 0.0, -1.50330396, 797.8,
  //                 0.0, 0.0, 1.0};
  s->camera_run(ts);
}

void driver_camera_thread(CameraState *s) {
  util::set_thread_name("android_driver_camera_thread");

  // transforms calculation see tools/webcam/warp_vis.py
  float ts[9] = {1.42070485, 0.0, -30.16740088,
                  0.0, 1.42070485, 91.030837,
                  0.0, 0.0, 1.0};
  // if camera upside down:
  // float ts[9] = {-1.42070485, 0.0, 1182.2,
  //                 0.0, -1.42070485, 773.0,
  //                 0.0, 0.0, 1.0};
  s->camera_run(ts);
}

const char *ParseFormat(enum AIMAGE_FORMATS format) {
  switch (format) {
    case AIMAGE_FORMAT_RGBA_8888:
      return "RGBA_8888";
    case AIMAGE_FORMAT_RGBX_8888:
      return "RGBX_8888";
    case AIMAGE_FORMAT_RGB_888:
      return "RGB_888";
    case AIMAGE_FORMAT_RGB_565:
      return "RGB_565";
    case AIMAGE_FORMAT_RGBA_FP16:
      return "RGBA_FP16";
    case AIMAGE_FORMAT_YUV_420_888:
      return "YUV_420_888";
    case AIMAGE_FORMAT_JPEG:
      return "JPEG";
    case AIMAGE_FORMAT_RAW16:
      return "RAW16";
    case AIMAGE_FORMAT_RAW12:
      return "RAW12";
    case AIMAGE_FORMAT_RAW10:
      return "RAW10";
    case AIMAGE_FORMAT_RAW_PRIVATE:
      return "RAW_PRIVATE";
    case AIMAGE_FORMAT_DEPTH16:
      return "DEPTH16";
    case AIMAGE_FORMAT_DEPTH_POINT_CLOUD:
      return "DEPTH_POINT_CLOUD";
    case AIMAGE_FORMAT_PRIVATE:
      return "PRIVATE";
    default:
      return "unknown";
  }
}

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  s->camera_manager = ACameraManager_create();

  ACameraIdList *camera_id_list = NULL;
  camera_status_t status = ACameraManager_getCameraIdList(s->camera_manager, &camera_id_list);
  assert(status == ACAMERA_OK);  // failed to get camera id list
  LOG("Found %d cameras", camera_id_list->numCameras);

  // loop over cameras and print info
  for (int i = 0; i < camera_id_list->numCameras; i++) {
    const char* id = camera_id_list->cameraIds[i];
    LOG("Camera index %d: id=\"%s\"", i, id);

    ACameraMetadata *metadata = NULL;
    status = ACameraManager_getCameraCharacteristics(s->camera_manager,
                                                     id,
                                                     &metadata);
    assert(status == ACAMERA_OK);  // failed to get camera characteristics

    ACameraMetadata_const_entry entry;
    status = ACameraMetadata_getConstEntry(metadata,
                                           ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS,
                                           &entry);

    // format of the data: format, width, height, input?, type int 32
    for (int j = 0; j < entry.count; j += 4) {
      int32_t input = entry.data.i32[j + 3];
      int32_t format = entry.data.i32[j + 0];
      if (input) continue;

      const char *format_name = ParseFormat(static_cast<AIMAGE_FORMATS>(format));
      int32_t width = entry.data.i32[j + 1];
      int32_t height = entry.data.i32[j + 2];
      LOG("Camera %s supports format %s (%d): %dx%d", id, format_name, format, width, height);
    }

    int32_t count = 0;
    const uint32_t* tags = nullptr;
    status = ACameraMetadata_getAllTags(metadata, &count, &tags);
    assert(status == ACAMERA_OK);  // failed to get camera metadata

    for (int idx = 0; idx < count; idx++) {
      if (ACAMERA_LENS_FACING == tags[idx]) {
        ACameraMetadata_const_entry lensInfo = { 0 };

        status = ACameraMetadata_getConstEntry(metadata, tags[idx], &lensInfo);
        assert(status == ACAMERA_OK);  // failed to get camera metadata
        LOG("Camera id=\"%s\": lens facing=%d", id, lensInfo.data.i32[0]);
      }
    }

    // free metadata
    ACameraMetadata_free(metadata);
  }

  // free camera id list
  ACameraManager_deleteCameraIdList(camera_id_list);

  LOG("*** init road camera ***");
  s->road_cam.camera_init(s, v, ROAD_CAMERA_INDEX, CAMERA_ID_IMX363, 20, device_id, ctx,
                          VISION_STREAM_RGB_ROAD, VISION_STREAM_ROAD);
  LOG("*** init driver camera ***");
  s->driver_cam.camera_init(s, v, DRIVER_CAMERA_INDEX, CAMERA_ID_IMX355, 10, device_id, ctx,
                            VISION_STREAM_RGB_DRIVER, VISION_STREAM_DRIVER);

  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "thumbnail"});
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(MultiCameraState *s) {
  LOG("*** open road camera ***");
  s->road_cam.camera_open();
  LOG("*** open driver camera ***");
  s->driver_cam.camera_open();
}

void cameras_close(MultiCameraState *s) {
  LOG("*** close road camera ***");
  s->road_cam.camera_close();
  LOG("*** close driver camera ***");
  s->driver_cam.camera_close();
  delete s->pm;
}

void process_driver_camera(MultiCameraState *s, CameraState *c, int cnt) {
  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverCameraState();
  framed.setFrameType(cereal::FrameData::FrameType::FRONT);
  fill_frame_data(framed, c->buf.cur_frame_data);
  s->pm->send("driverCameraState", msg);
}

void process_road_camera(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;
  MessageBuilder msg;
  auto framed = msg.initEvent().initRoadCameraState();
  fill_frame_data(framed, b->cur_frame_data);
  framed.setImage(kj::arrayPtr((const uint8_t *)b->cur_yuv_buf->addr, b->cur_yuv_buf->len));
  framed.setTransform(b->yuv_transform.v);
  s->pm->send("roadCameraState", msg);
}

void cameras_run(MultiCameraState *s) {
  LOG("-- Starting threads");
  std::vector<std::thread> threads;
  threads.push_back(start_process_thread(s, &s->road_cam, process_road_camera));
  threads.push_back(start_process_thread(s, &s->driver_cam, process_driver_camera));

  std::thread t_rear = std::thread(road_camera_thread, &s->road_cam);
  driver_camera_thread(&s->driver_cam);

  LOG(" ************** STOPPING **************");

  t_rear.join();

  for (auto &t : threads) t.join();

  cameras_close(s);
}
