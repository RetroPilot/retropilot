#include "selfdrive/camerad/cameras/camera_android.h"

#include <camera/NdkCameraError.h>

#include "selfdrive/common/swaglog.h"

// id of the video capturing device
const int ROAD_CAMERA_INDEX = util::getenv("ROADCAM_ID", 1);
const int DRIVER_CAMERA_INDEX = util::getenv("DRIVERCAM_ID", 2);

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

void CameraState::camera_open() {
  LOGD("camera_open %d", camera_num);
}

void CameraState::camera_close() {
  LOGD("camera_close %d", camera_num);

  // if (image_reader) {
  //   delete image_reader;
  //   image_reader = nullptr;
  // }
  // if (native_camera) {
  //   delete native_camera;
  //   native_camera = nullptr;
  // }
}

void CameraState::camera_init(MultiCameraState *multi_cam_state_, VisionIpcServer *v, int camera_index, int camera_id_, unsigned int fps_, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type) {
  LOGD("camera_init: camera_index %d, camera_id_ %d", camera_index, camera_id_);

  multi_cam_state = multi_cam_state_;

  assert(camera_id_ < std::size(cameras_supported));
  ci = cameras_supported[camera_id_];
  assert(ci.frame_width != 0);

  camera_num = camera_index;
  fps = fps_;
  buf.init(device_id, ctx, this, v, FRAME_BUF_COUNT, rgb_type, yuv_type);

  // ASSUMPTION: IXM363 (road) is index[0] and IMX355 (driver) is index[1]
  // TODO: check that we actually need to rotate
  // if (camera_id_ == CAMERA_ID_IMX363) {
  //   camera_orientation = 90;
  // } else if (camera_id_ == CAMERA_ID_IMX355) {
  //   camera_orientation = 270;
  // }
}

void CameraState::camera_run(float *ts) {
  LOGD("camera_run %d", camera_num);

  uint32_t frame_id = 0;

  enum AIMAGE_FORMATS fmt = AIMAGE_FORMAT_YUV_420_888;

  native_camera = new NativeCamera(camera_num);
  native_camera->match_capture_size_request(&view, ci.frame_width, ci.frame_height, fmt);
  assert(view.width && view.height);

  image_reader = new ImageReader(&view, fmt);

  ANativeWindow *window = image_reader->GetNativeWindow();
  native_camera->create_capture_session(window);

#if false
  native_camera->start_preview(true);
#endif

  while (!do_exit) {
    AImage *image = image_reader->GetLatestImage();
    if (!image) {
      util::sleep_for(1);
      continue;
    }

    LOGD("camera_run: image=%p", image);

    media_status_t status;

    int32_t planeCount;
    int32_t format;
    status = AImage_getNumberOfPlanes(image, &planeCount);
    assert(status == AMEDIA_OK && planeCount == 3);
    status = AImage_getFormat(image, &format);
    assert(status == AMEDIA_OK && format == AIMAGE_FORMAT_YUV_420_888);

    uint8_t *y_data, *u_data, *v_data;
    int y_len, u_len, v_len;

    status = AImage_getPlaneData(image, 0, &y_data, &y_len);
    assert(status == AMEDIA_OK);
    status = AImage_getPlaneData(image, 1, &u_data, &u_len);
    assert(status == AMEDIA_OK);
    status = AImage_getPlaneData(image, 2, &v_data, &v_len);
    assert(status == AMEDIA_OK);

    LOGD("camera_run: y_data=%p, u_data=%p, v_data=%p", y_data, u_data, v_data);
    LOGD("camera_run: y_len=%d, u_len=%d, v_len=%d", y_len, u_len, v_len);

    // ** release image **
    AImage_delete(image);

    ++frame_id;
  }

#if false
  native_camera->start_preview(false);
#endif
}

#if ROAD
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
#endif

static void driver_camera_thread(CameraState *s) {
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

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
#if ROAD
  LOG("*** init road camera ***");
  s->road_cam.camera_init(s, v, ROAD_CAMERA_INDEX, CAMERA_ID_IMX363, 20, device_id, ctx,
                          VISION_STREAM_RGB_ROAD, VISION_STREAM_ROAD);
#endif
  LOG("*** init driver camera ***");
  s->driver_cam.camera_init(s, v, DRIVER_CAMERA_INDEX, CAMERA_ID_IMX355, 10, device_id, ctx,
                            VISION_STREAM_RGB_DRIVER, VISION_STREAM_DRIVER);

  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "thumbnail"});
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(MultiCameraState *s) {
#if ROAD
  LOG("*** open road camera ***");
  s->road_cam.camera_open();
#endif
  LOG("*** open driver camera ***");
  s->driver_cam.camera_open();
}

void cameras_close(MultiCameraState *s) {
#if ROAD
  LOG("*** close road camera ***");
  s->road_cam.camera_close();
#endif
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

#if ROAD
void process_road_camera(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;
  MessageBuilder msg;
  auto framed = msg.initEvent().initRoadCameraState();
  fill_frame_data(framed, b->cur_frame_data);
  framed.setImage(kj::arrayPtr((const uint8_t *)b->cur_yuv_buf->addr, b->cur_yuv_buf->len));
  framed.setTransform(b->yuv_transform.v);
  s->pm->send("roadCameraState", msg);
}
#endif

void cameras_run(MultiCameraState *s) {
  LOG("-- Starting threads");
  std::vector<std::thread> threads;
#if false
  threads.push_back(start_process_thread(s, &s->road_cam, process_road_camera));
#endif
  threads.push_back(start_process_thread(s, &s->driver_cam, process_driver_camera));

#if false
  threads.push_back(std::thread(road_camera_thread, &s->road_cam));
#endif
  driver_camera_thread(&s->driver_cam);

  LOG(" ************** STOPPING **************");

  for (auto &t : threads) t.join();

  cameras_close(s);
}
