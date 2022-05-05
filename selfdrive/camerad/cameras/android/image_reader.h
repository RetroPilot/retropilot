#pragma once

#include <media/NdkImageReader.h>

#include "selfdrive/camerad/cameras/android/image_format.h"

class ImageReader {
private:
  AImageReader *reader;

public:
  explicit ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format);

  ~ImageReader();

  ANativeWindow *GetNativeWindow();

  AImage *GetLatestImage();

  void DeleteImage(AImage *image);

  void ImageCallback(AImageReader *reader);
};
