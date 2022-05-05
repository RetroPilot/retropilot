#pragma once

const char *GetImageFormatName(enum AIMAGE_FORMATS format) {
  switch (format) {
    case AIMAGE_FORMAT_RGBA_8888:
      return "RGBA_8888";
    case AIMAGE_FORMAT_RGBX_8888:
      return "RGBX_8888";
    case AIMAGE_FORMAT_RGB_888:
      return "RGB_888";
    case AIMAGE_FORMAT_RGB_565:
      return "RGB_565";
    case AIMAGE_FORMAT_RGBA_FP16:
      return "RGBA_FP16";
    case AIMAGE_FORMAT_YUV_420_888:
      return "YUV_420_888";
    case AIMAGE_FORMAT_JPEG:
      return "JPEG";
    case AIMAGE_FORMAT_RAW16:
      return "RAW16";
    case AIMAGE_FORMAT_RAW12:
      return "RAW12";
    case AIMAGE_FORMAT_RAW10:
      return "RAW10";
    case AIMAGE_FORMAT_RAW_PRIVATE:
      return "RAW_PRIVATE";
    case AIMAGE_FORMAT_DEPTH16:
      return "DEPTH16";
    case AIMAGE_FORMAT_DEPTH_POINT_CLOUD:
      return "DEPTH_POINT_CLOUD";
    case AIMAGE_FORMAT_PRIVATE:
      return "PRIVATE";
    default:
      return "UNKNOWN";
  }
}
