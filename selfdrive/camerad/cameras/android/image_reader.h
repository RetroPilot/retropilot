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

  void OnImageCallback(void *ctx, AImageReader *reader);

private:
  AImageReader *reader_;

  uint8_t *image_buffer_;
};
