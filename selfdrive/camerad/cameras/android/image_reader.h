#pragma once

#include <media/NdkImageReader.h>

#include <string>

#include "selfdrive/camerad/cameras/android/util.h"


class ImageReader {
public:
  explicit ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format);

  ~ImageReader();

  AImage *GetLatestImage();

  void DeleteImage(AImage *image);

  void ImageCallback(AImageReader *reader);

private:
  AImageReader *reader_;

  uint32_t image_height_;
  uint32_t image_width_;

  uint8_t *image_buffer_;
};
