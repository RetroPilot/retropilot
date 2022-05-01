#pragma once

#include <media/NdkImageReader.h>

#include <cassert>
#include <stdlib.h>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/camerad/cameras/android/util.h"

class ImageReader {
public:
  explicit ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format);

  ~ImageReader();

  ANativeWindow *GetNativeWindow();

  AImage *GetLatestImage();

  void DeleteImage(AImage *image);

  void ImageCallback(AImageReader *reader);

private:
  AImageReader *reader_;

  uint32_t image_height_;
  uint32_t image_width_;

  uint8_t *image_buffer_;
};

/**
 * ImageReader listener: called by AImageReader for every frame captured
 * We pass the event to ImageReader class, so it could do some housekeeping
 * about
 * the loaded queue. For example, we could keep a counter to track how many
 * buffers are full and idle in the queue. If camera almost has no buffer to
 * capture
 * we could release ( skip ) some frames by AImageReader_getNextImage() and
 * AImageReader_delete().
 */
void OnImageCallback(void *ctx, AImageReader *reader) {
    reinterpret_cast<ImageReader *>(ctx)->ImageCallback(reader);
}
