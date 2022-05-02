#pragma once

#include <media/NdkImageReader.h>

#include <cassert>
#include <cstdlib>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/camerad/cameras/android/image_format.h"

class ImageReader {
public:
  explicit ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format);

  ~ImageReader();

  /**
   * Report cached ANativeWindow, which was used to create camera's capture
   * session output.
   */
  ANativeWindow *GetNativeWindow();

  /**
   * Retrieve Image on the back of Reader's queue, dropping older images
   */
  AImage *GetLatestImage();

  /**
   * Delete Image
   * @param image {@link AImage} instance to be deleted
   */
  void DeleteImage(AImage *image);

private:
  AImageReader *reader_;

  uint8_t *image_buffer_;
};
