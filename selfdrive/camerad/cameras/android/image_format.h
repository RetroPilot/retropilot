#pragma once

/*
 * ImageFormat:
 *     A Data Structure to communicate resolution between camera and ImageReader
 */
struct ImageFormat {
  int32_t width;
  int32_t height;

  int32_t format; // Through out this demo, the format is fixed to
                  // YUV_420 format
};
