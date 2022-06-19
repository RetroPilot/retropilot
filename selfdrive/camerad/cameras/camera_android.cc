#include "selfdrive/camerad/cameras/camera_android.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#pragma clang diagnostic pop

#include <binder/ProcessState.h>
#include <camera/NdkCameraError.h>

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"

// id of the video capturing device
const int ROAD_CAMERA_INDEX = util::getenv("ROADCAM_ID", 0);
const int DRIVER_CAMERA_INDEX = util::getenv("DRIVERCAM_ID", 1);

//TODO: get supported resolution from param

#define FRAME_WIDTH  1164
#define FRAME_HEIGHT 874
#define NDK_WIDTH 1920
#define NDK_HEIGHT 1080
#define FRAME_WIDTH_FRONT  1920
#define FRAME_HEIGHT_FRONT 1080

#define ROAD 1
//#define DRIVER 1

using namespace cv;

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

void CameraState::camera_run(CameraState *s) {
  LOGD("camera_run %d", camera_num);

  uint32_t frame_id = 0;

  enum AIMAGE_FORMATS fmt = AIMAGE_FORMAT_YUV_420_888;

  native_camera = new NativeCamera(camera_num);
  //native_camera->match_capture_size_request(&view, ci.frame_width, ci.frame_height, fmt);
  // TODO: collect supported camera resolution from a parameter
  native_camera->match_capture_size_request(&view, NDK_WIDTH, NDK_HEIGHT, fmt);
  assert(view.width && view.height);

  image_reader = new ImageReader(&view, fmt);

  ANativeWindow *window = image_reader->GetNativeWindow();
  native_camera->create_capture_session(window);

  native_camera->start_preview(true);

  double time = nanos_since_boot() * 1e-9;
  int frame_count = 0;

  size_t buf_idx = 0;

  bool snap_taken = 0;

  while (!do_exit) {
    AImage *image = image_reader->GetLatestImage();
    if (!image) {
      util::sleep_for(1);
      continue;
    }
    
    // convert NDK capture into a CV Mat

    int32_t y_stride, uv_stride;
    int32_t uv_pixel_stride;
    int32_t y_len, u_len, v_len;
    uint8_t *y_data, *u_data, *v_data;
    AImageCropRect src_rect;

    AImage_getPlaneRowStride(image, 0, &y_stride);
    AImage_getPlaneRowStride(image, 1, &uv_stride);
    AImage_getPlanePixelStride(image, 1, &uv_pixel_stride);
    AImage_getPlaneData(image, 0, &y_data, &y_len);
    AImage_getPlaneData(image, 1, &u_data, &u_len);
    AImage_getPlaneData(image, 2, &v_data, &v_len);
    AImage_getCropRect(image, &src_rect);

    // NV21 U/V interleaved format
    assert(uv_pixel_stride == 2);
    assert(u_data == v_data + 1);

    uchar buff[y_len+u_len+v_len];

    memcpy(buff+0,y_data,y_len);
    memcpy(buff+y_len,u_data,v_len);
    memcpy(buff+y_len+v_len,u_data,u_len);
    
    // shove buffer into a CV mat for later processing
    cv::Mat yuvMat(NDK_HEIGHT, NDK_WIDTH, CV_8UC1, &buff);

    // cv transform to C2 spec
    cv::Mat transformed_mat;
//    float ts[9] = {1.50330396, 0.0, -59.40969163,
//                  0.0, 1.50330396, 76.20704846,
//                  0.0, 0.0, 1.0};
    float ts[9] = {0.5803214805005242, 0.0, 34.466683147755475,
                   0.0, 0.5803214805005242, 17.137408857870753,
                   0.0, 0.0, 1.0};
    const cv::Mat transform = cv::Mat(3, 3, CV_32F, ts);
    cv::Size size(1164, 874);
    cv::warpPerspective(yuvMat, transformed_mat, transform, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    
    // DEBUG: save PNG of 100th frame
    if ((frame_count % 100 == 0) && !snap_taken ){
      imwrite("/sdcard/Download/cv_out.png", yuvMat);
      snap_taken = 1;
    }

    frame_count++;
    if (frame_count % 100 == 0) {
      LOGD("camera_run: fps %.2f", 100.0 / (nanos_since_boot() * 1e-9 - time));
      time = nanos_since_boot() * 1e-9;
      frame_count = 0;
    }

    // ** debug **
    media_status_t status;

    int32_t planeCount;
    int32_t format;
    status = AImage_getNumberOfPlanes(image, &planeCount);
    assert(status == AMEDIA_OK && planeCount == 3);
    status = AImage_getFormat(image, &format);
    assert(status == AMEDIA_OK && format == AIMAGE_FORMAT_YUV_420_888);

    // ** send frame **
    FrameMetadata frame_data = {
      .frame_id = frame_id,
      .timestamp_eof = nanos_since_boot(),
    };

    // send transformed_mat cv::Mat
    s->buf.camera_bufs_metadata[buf_idx] = {.frame_id = frame_id};
    auto &buf2 = s->buf.camera_bufs[buf_idx];
    int transformed_size = transformed_mat.total() * transformed_mat.elemSize();
    clEnqueueWriteBuffer(buf2.copy_q, buf2.buf_cl, CL_TRUE, 0, transformed_size, transformed_mat.data, 0, NULL, NULL);
    s->buf.queue(buf_idx);
    buf_idx = (buf_idx + 1) % 16;

    // send AImage
    // buf.send_yuv(image, frame_id, frame_data);

    MessageBuilder msg;
    if (camera_num == ROAD_CAMERA_INDEX) {
      auto framed = msg.initEvent().initRoadCameraState();
      fill_frame_data(framed, frame_data);
      framed.setImage(kj::arrayPtr((const uint8_t *)buf.cur_yuv_buf->addr, buf.cur_yuv_buf->len));
      framed.setTransform(buf.yuv_transform.v);

      multi_cam_state->pm->send("roadCameraState", msg);
    } else if (camera_num == DRIVER_CAMERA_INDEX) {
      auto framed = msg.initEvent().initDriverCameraState();
      framed.setImage(kj::arrayPtr((const uint8_t *)buf.cur_yuv_buf->addr, buf.cur_yuv_buf->len));
      framed.setTransform(buf.yuv_transform.v);

      multi_cam_state->pm->send("driverCameraState", msg);
    }

    // ** release image **
    AImage_delete(image);

    ++frame_id;
  }

  //release video writer
  //video.release();

  native_camera->start_preview(false);
}

#if ROAD
static void road_camera_thread(CameraState *s) {
  util::set_thread_name("android_road_camera_thread");
  s->camera_run(s);
}
#endif

#if DRIVER
static void driver_camera_thread(CameraState *s) {
  util::set_thread_name("android_driver_camera_thread");
  s->camera_run(s);
}
#endif

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
#if ROAD
  LOG("*** init road camera ***");
  s->road_cam.camera_init(s, v, ROAD_CAMERA_INDEX, CAMERA_ID_IMX363, 20, device_id, ctx,
                          VISION_STREAM_RGB_ROAD, VISION_STREAM_ROAD);
#endif
#if DRIVER
  LOG("*** init driver camera ***");
  s->driver_cam.camera_init(s, v, DRIVER_CAMERA_INDEX, CAMERA_ID_IMX355, 10, device_id, ctx,
                            VISION_STREAM_RGB_DRIVER, VISION_STREAM_DRIVER);
#endif

  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "thumbnail"});
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(MultiCameraState *s) {
#if ROAD
  LOG("*** open road camera ***");
  s->road_cam.camera_open();
#endif
#if DRIVER
  LOG("*** open driver camera ***");
  s->driver_cam.camera_open();
#endif
}

void cameras_close(MultiCameraState *s) {
#if ROAD
  LOG("*** close road camera ***");
  s->road_cam.camera_close();
#endif
#if DRIVER
  LOG("*** close driver camera ***");
  s->driver_cam.camera_close();
#endif
  delete s->pm;
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

#if DRIVER
void process_driver_camera(MultiCameraState *s, CameraState *c, int cnt) {
  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverCameraState();
  framed.setFrameType(cereal::FrameData::FrameType::FRONT);
  fill_frame_data(framed, c->buf.cur_frame_data);
  s->pm->send("driverCameraState", msg);
}
#endif

void cameras_run(MultiCameraState *s) {
  LOG("-- Starting threads");
  android::ProcessState::self()->startThreadPool();
  std::vector<std::thread> threads;

#if ROAD
  threads.push_back(start_process_thread(s, &s->road_cam, process_road_camera));
#endif
#if DRIVER
  threads.push_back(start_process_thread(s, &s->driver_cam, process_driver_camera));
#endif

#if DRIVER
#if ROAD
  std::thread t_rear = std::thread(road_camera_thread, &s->road_cam);
  driver_camera_thread(&s->driver_cam);
  t_rear.join();
#else
  driver_camera_thread(&s->driver_cam);
#endif
#else
#if ROAD
  road_camera_thread(&s->road_cam);
#endif
#endif

  LOG(" ************** STOPPING **************");

  for (auto &t : threads) t.join();

  cameras_close(s);
}
