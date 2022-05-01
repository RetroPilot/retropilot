#include "selfdrive/camerad/cameras/camera_android.h"

#include <unistd.h>

#include <cassert>
#include <cstring>

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/camerad/cameras/android/image_reader.h"

// id of the video capturing device
const int ROAD_CAMERA_ID = util::getenv("ROADCAM_ID", 1);
const int DRIVER_CAMERA_ID = util::getenv("DRIVERCAM_ID", 2);

#define FRAME_WIDTH  4032
#define FRAME_HEIGHT 3024
#define FRAME_WIDTH_FRONT  2592
#define FRAME_HEIGHT_FRONT 1944

extern ExitHandler do_exit;

namespace {

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

void camera_open(CameraState *s) {
  ACameraManager *camera_manager = s->multi_camera_state->camera_manager;

  // ** open camera **
  camera_status_t camera_status = ACameraManager_openCamera(camera_manager, s->camera_id,
                                            &s->device_state_callbacks, &s->camera_device);
  assert(camera_status == ACAMERA_OK); // failed to open camera

  // ** create capture session **
  ANativeWindow *window = s->image_reader->GetNativeWindow();

  camera_status = ACaptureSessionOutputContainer_create(&s->capture_session_output_container);
  assert(camera_status == ACAMERA_OK); // failed to create capture session output container

  ANativeWindow_acquire(window);
  camera_status = ACaptureSessionOutput_create(window, &s->capture_session_output);
  assert(camera_status == ACAMERA_OK); // failed to create capture session output

  camera_status = ACaptureSessionOutputContainer_add(s->capture_session_output_container,
                                                     s->capture_session_output);
  assert(camera_status == ACAMERA_OK); // failed to add capture session output to container

  camera_status = ACameraOutputTarget_create(window, &s->camera_output_target);
  assert(camera_status == ACAMERA_OK); // failed to create camera output target

  camera_status = ACameraDevice_createCaptureRequest(s->camera_device,
                                                     TEMPLATE_RECORD, &s->capture_request);
  assert(camera_status == ACAMERA_OK); // failed to create preview capture request

  camera_status = ACaptureRequest_addTarget(s->capture_request, s->camera_output_target);
  assert(camera_status == ACAMERA_OK); // failed to add camera output target to preview capture request

  s->capture_session_state_callbacks.onReady = &CaptureSessionOnReady;
  s->capture_session_state_callbacks.onActive = &CaptureSessionOnActive;
  camera_status = ACameraDevice_createCaptureSession(s->camera_device, s->capture_session_output_container,
                                                     &s->capture_session_state_callbacks, &s->capture_session);
  assert(camera_status == ACAMERA_OK); // failed to create capture session

  camera_status = ACameraCaptureSession_setRepeatingRequest(s->capture_session, NULL, 1,
                                                            &s->capture_request, NULL);
  assert(camera_status == ACAMERA_OK); // failed to set repeating request
}

void camera_close(CameraState *s) {
  if (s->capture_request) {
    ACaptureRequest_free(s->capture_request);
    s->capture_request = NULL;
  }

  if (s->camera_output_target) {
    ACameraOutputTarget_free(s->camera_output_target);
    s->camera_output_target = NULL;
  }

  if (s->camera_device) {
    ACameraDevice_close(s->camera_device);
    s->camera_device = NULL;
  }

  if (s->capture_session_output_container) {
    if (s->capture_session_output) {
      ACaptureSessionOutputContainer_remove(s->capture_session_output_container,
                                            s->capture_session_output);

      ACaptureSessionOutput_free(s->capture_session_output);
      s->capture_session_output = NULL;
    }

    ACaptureSessionOutputContainer_free(s->capture_session_output_container);
    s->capture_session_output_container = NULL;
  }
}

void camera_init(VisionIpcServer *v, CameraState *s, int camera_id, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type) {
  assert(camera_id < std::size(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->camera_num = camera_id;
  s->fps = fps;
  s->buf.init(device_id, ctx, s, v, FRAME_BUF_COUNT, rgb_type, yuv_type);

  ACameraManager *camera_manager = s->multi_camera_state->camera_manager;

  // ** get camera list **
  ACameraIdList *camera_id_list = NULL;
  camera_status_t camera_status = ACameraManager_getCameraIdList(camera_manager, &camera_id_list);
  assert(camera_status == ACAMERA_OK); // failed to get camera id list

  // ** set (android) camera id **
  s->camera_id = camera_id_list->cameraIds[camera_id];

  // ASSUMPTION: IXM363 (road) is index[0] and IMX355 (driver) is index[1]
  // TODO: check that we actually need to rotate
  if (camera_id == CAMERA_ID_IMX363) {
    s->camera_orientation = 90;
  } else if (camera_id == CAMERA_ID_IMX355) {
    s->camera_orientation = 270;
  }

  // ** setup callbacks **
  s->device_state_callbacks.onDisconnected = CameraDeviceOnDisconnected;
  s->device_state_callbacks.onError = CameraDeviceOnError;

  // ** create image reader **
  s->image_reader = new ImageReader(&s->image_format, AIMAGE_FORMAT_YUV_420_888);
}

void run_camera(CameraState *s, float *ts) {
  // cv::Size size(s->ci.frame_width, s->ci.frame_height);
  // const cv::Mat transform = cv::Mat(3, 3, CV_32F, ts);
  uint32_t frame_id = 0;
  size_t buf_idx = 0;

  while (!do_exit) {
    // cv::Mat frame_mat, transformed_mat;
    // video_cap >> frame_mat;
    // if (frame_mat.empty()) continue;

    // ** get image **
    AImage *image = s->image_reader->GetLatestImage();
    if (image == NULL) continue;

    // ** debug **
    int32_t format = -1;
    AImage_getFormat(image, &format);
    assert(format == AIMAGE_FORMAT_YUV_420_888);

    int32_t planes = 0;
    AImage_getNumberOfPlanes(image, &planes);
    assert(planes == 3);

    // cv::warpPerspective(frame_mat, transformed_mat, transform, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    s->buf.camera_bufs_metadata[buf_idx] = {.frame_id = frame_id};

    // ** copy image data to cl buffer **

    int32_t width = 0, height = 0;
    AImage_getWidth(image, &width);
    AImage_getHeight(image, &height);

    auto &buf = s->buf.camera_bufs[buf_idx];
    int transformed_size = transformed_mat.total() * transformed_mat.elemSize();
    CL_CHECK(clEnqueueWriteBuffer(buf.copy_q, buf.buf_cl, CL_TRUE, 0, transformed_size, transformed_mat.data, 0, NULL, NULL));

    s->buf.queue(buf_idx);

    ++frame_id;
    buf_idx = (buf_idx + 1) % FRAME_BUF_COUNT;
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

  run_camera(s, ts);
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
  run_camera(s, ts);
}

}  // namespace

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  s->camera_manager = ACameraManager_create();

  LOG("*** init road camera ***");
  camera_init(v, &s->road_cam, ROAD_CAMERA_ID, 20, device_id, ctx,
              VISION_STREAM_RGB_ROAD, VISION_STREAM_ROAD);
  LOG("*** init driver camera ***");
  camera_init(v, &s->driver_cam, DRIVER_CAMERA_ID, 10, device_id, ctx,
              VISION_STREAM_RGB_DRIVER, VISION_STREAM_DRIVER);

  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "thumbnail"});
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(MultiCameraState *s) {
  LOG("*** open road camera ***");
  camera_open(&s->road_cam);
  LOG("*** open driver camera ***");
  camera_open(&s->driver_cam);
}

void cameras_close(MultiCameraState *s) {
  LOG("*** close road camera ***");
  camera_close(&s->road_cam);
  LOG("*** close driver camera ***");
  camera_close(&s->driver_cam);
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
