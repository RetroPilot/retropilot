#include "selfdrive/camerad/cameras/android/image_reader.h"

ImageReader::ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format)
    : reader_(NULL) {

  media_status_t status = AImageReader_new(res->width, res->height, format,
                                          2, &reader_);
  assert(reader_ && status == AMEDIA_OK); // failed to create AImageReader

  // AImageReader_ImageListener listener{
  //   .context = this,
  //   .onImageAvailable = OnImageCallback,
  // };
  // AImageReader_setImageListener(reader_, &listener);

  // assuming 4 bit per pixel max
  LOGE("Image Buffer Size: %d", res->width * res->height * 4);
  image_buffer_ = (uint8_t *)malloc(res->width * res->height * 4);
  assert(image_buffer_); // failed to allocate image buffer
}

ImageReader::~ImageReader() {
  if (reader_) {
    AImageReader_delete(reader_);
  }

  if (image_buffer_) {
    free(image_buffer_);
  }
}

ANativeWindow *ImageReader::GetNativeWindow() {
  assert(reader_);

  ANativeWindow *native_window;
  media_status_t status = AImageReader_getWindow(reader_, &native_window);
  assert(status == AMEDIA_OK); // failed to get native window

  return native_window;
}

AImage *ImageReader::GetLatestImage() {
  AImage *image = NULL;
  media_status_t status = AImageReader_acquireLatestImage(reader_, &image);
  assert(status == AMEDIA_OK); // image is not available

  return image;
}

void ImageReader::DeleteImage(AImage *image) {
  if (image) {
    AImage_delete(image);
  }
}
