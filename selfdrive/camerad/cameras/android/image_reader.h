#pragma once

#include <media/NdkImageReader.h>

#include "selfdrive/camerad/cameras/android/image_format.h"

class ImageReader {
private:
  AImageReader *reader;

public:
  explicit ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format);

  ~ImageReader();

  /**
   * Get the Native Window object
   */
  ANativeWindow *GetNativeWindow();

  /**
   * Get the image on the top of the queue.
   */
  AImage *GetNextImage();

  /**
   * @brief Retrieve image on the bottom of the queue.
   *
   * @return AImage*
   */
  AImage *GetLatestImage();

  void DeleteImage(AImage *image);

  void ImageCallback(AImageReader *reader);
};
