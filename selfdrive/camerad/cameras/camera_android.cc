#include "selfdrive/camerad/cameras/camera_android.h"

// id of the video capturing device
const int ROAD_CAMERA_ID = util::getenv("ROADCAM_ID", 1);
const int DRIVER_CAMERA_ID = util::getenv("DRIVERCAM_ID", 2);

#define FRAME_WIDTH  4032
#define FRAME_HEIGHT 3024
#define FRAME_WIDTH_FRONT  2592
#define FRAME_HEIGHT_FRONT 1944

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

void CameraState::init(VisionIpcServer *v, int camera_num, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type) {
  assert(camera_num < std::size(cameras_supported));
  ci = cameras_supported[camera_num];
  assert(ci.frame_width != 0);

  camera_num = camera_num;
  fps = fps;
  buf.init(device_id, ctx, this, v, FRAME_BUF_COUNT, rgb_type, yuv_type);

  ACameraManager *camera_manager = multi_camera_state->camera_manager;

  // ** get camera list **
  ACameraIdList *camera_id_list = NULL;
  camera_status_t camera_status = ACameraManager_getCameraIdList(camera_manager, &camera_id_list);
  assert(camera_status == ACAMERA_OK); // failed to get camera id list

  // ** set (android) camera id **
  camera_id = camera_id_list->cameraIds[camera_num];

  // ASSUMPTION: IXM363 (road) is index[0] and IMX355 (driver) is index[1]
  // TODO: check that we actually need to rotate
  if (camera_num == CAMERA_ID_IMX363) {
    camera_orientation = 90;
  } else if (camera_num == CAMERA_ID_IMX355) {
    camera_orientation = 270;
  }

  // ** setup callbacks **
  device_state_callbacks.onDisconnected = CameraDeviceOnDisconnected;
  device_state_callbacks.onError = CameraDeviceOnError;

  // ** create image reader **
  image_reader = new ImageReader(&image_format, AIMAGE_FORMAT_YUV_420_888);
}

void CameraState::open() {
  ACameraManager *camera_manager = multi_camera_state->camera_manager;

  ACameraManager_openCamera(camera_manager, camera_id,
                                            &device_state_callbacks, &camera_device);

  ANativeWindow *window = image_reader->GetNativeWindow();

  ACaptureSessionOutputContainer_create(&capture_session_output_container);
  ANativeWindow_acquire(window);
  ACaptureSessionOutput_create(window, &capture_session_output);
  ACaptureSessionOutputContainer_add(capture_session_output_container,
                                     capture_session_output);
  ACameraOutputTarget_create(window, &camera_output_target);

  // use TEMPLATE_RECORD for good quality and OK frame rate
  camera_status_t status = ACameraDevice_createCaptureRequest(camera_device,
                                                              TEMPLATE_RECORD, &capture_request);
  assert(status == ACAMERA_OK); // failed to create preview capture request

  ACaptureRequest_addTarget(capture_request, camera_output_target);

  capture_session_state_callbacks.onReady = CaptureSessionOnReady;
  capture_session_state_callbacks.onActive = CaptureSessionOnActive;
  ACameraDevice_createCaptureSession(camera_device, capture_session_output_container,
                                     &capture_session_state_callbacks, &capture_session);

  ACameraCaptureSession_setRepeatingRequest(capture_session, NULL, 1, &capture_request, NULL);
}

void CameraState::run(float *ts) {
  // TODO: implement transform
  // cv::Size size(ci.frame_width, ci.frame_height);
  // const cv::Mat transform = cv::Mat(3, 3, CV_32F, ts);

  uint32_t frame_id = 0;
  size_t buf_idx = 0;

  while (!do_exit) {
    // ** get image **
    AImage *image = image_reader->GetLatestImage();
    if (image == NULL) continue;

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
    CL_CHECK(clEnqueueWriteBuffer(buffer.copy_q, buffer.buf_cl, CL_TRUE, 0, size, data, 0, NULL, NULL));

    buf.queue(buf_idx);

    ++frame_id;
    buf_idx = (buf_idx + 1) % FRAME_BUF_COUNT;
  }
}

void CameraState::close() {
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
  s->run(ts);
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
  s->run(ts);
}

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  s->camera_manager = ACameraManager_create();

  LOG("*** init road camera ***");
  s->road_cam.init(v, ROAD_CAMERA_ID, 20, device_id, ctx,
                   VISION_STREAM_RGB_ROAD, VISION_STREAM_ROAD);
  LOG("*** init driver camera ***");
  s->driver_cam.init(v, DRIVER_CAMERA_ID, 10, device_id, ctx,
                     VISION_STREAM_RGB_DRIVER, VISION_STREAM_DRIVER);

  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "thumbnail"});
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(MultiCameraState *s) {
  LOG("*** open road camera ***");
  s->road_cam.open();
  LOG("*** open driver camera ***");
  s->driver_cam.open();
}

void cameras_close(MultiCameraState *s) {
  LOG("*** close road camera ***");
  s->road_cam.close();
  LOG("*** close driver camera ***");
  s->driver_cam.close();
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
