#include "selfdrive/camerad/cameras/android/image_reader.h"

#include <cassert>

#include "selfdrive/common/swaglog.h"

/**
 * ImageReader listener: called by AImageReader for every frame captured
 * We pass the event to ImageReader class, so it could do some housekeeping
 * about the loaded queue. For example, we could keep a counter to track how
 * many buffers are full and idle in the queue. If camera almost has no buffer
 * to capture we could release ( skip ) some frames by AImageReader_getNextImage()
 * and AImageReader_delete().
 */
void OnImageCallback(void *ctx, AImageReader *reader) {
  reinterpret_cast<ImageReader *>(ctx)->ImageCallback(reader);
}

ImageReader::ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format) {
  media_status_t status = AImageReader_new(res->width, res->height, format,
                                           2, &reader);
  assert(status == AMEDIA_OK);

#if false
  AImageReader_ImageListener listener {
      .context = this,
      .onImageAvailable = OnImageCallback,
  };
  AImageReader_setImageListener(reader, &listener);
#endif
}

ImageReader::~ImageReader() {
  AImageReader_delete(reader);
}

ANativeWindow *ImageReader::GetNativeWindow() {
  ANativeWindow *window;
  media_status_t status = AImageReader_getWindow(reader, &window);
  assert(status == AMEDIA_OK);
  return window;
}

AImage *ImageReader::GetLatestImage() {
  AImage *image;
  media_status_t status = AImageReader_acquireLatestImage(reader, &image);
  if (status != AMEDIA_OK) {
    LOGW("AImageReader_acquireLatestImage failed: %d", status);
    return nullptr;
  }
  return image;
}

void ImageReader::DeleteImage(AImage *image) {
  if (image) {
    AImage_delete(image);
  }
}

void ImageReader::ImageCallback(AImageReader *r) {
  LOGD("ImageReader::ImageCallback");

  int32_t format;
  media_status_t status = AImageReader_getFormat(r, &format);
  assert(status == AMEDIA_OK);
  LOGD("format: 0x%X", format);

  if (format == AIMAGE_FORMAT_JPEG) {
    AImage *image;
    status = AImageReader_acquireNextImage(r, &image);
    assert(status == AMEDIA_OK && image);

    int planeCount;
    status = AImage_getNumberOfPlanes(image, &planeCount);
    assert(status == AMEDIA_OK && planeCount == 1);

    uint8_t *data;
    int len;
    AImage_getPlaneData(image, 0, &data, &len);
    LOGD("data: %p, len: %d", data, len);

    // TODO: do we need to delete the image? we might use it somewhere else
    AImage_delete(image);
  }
}
