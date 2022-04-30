#include "selfdrive/camerad/cameras/android/image_reader.h"

#include <cassert>

#include "selfdrive/common/swaglog.h"

/**
 * FRAME_BUF_COUNT:
 *   Max buffers in this ImageReader.
 */
#define FRAME_BUF_COUNT 2

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

ImageReader::ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format)
    : reader_(NULL),
      image_height_(res->height),
      image_width_(res->width) {

    media_status_t status = AImageReader_new(res->width, res->height, format,
                                             1, &reader_);
    assert(reader_ && status == AMEDIA_OK); // failed to create AImageReader

    AImageReader_ImageListener listener{
        .context = this,
        .onImageAvailable = OnImageCallback,
    };
    AImageReader_setImageListener(reader_, &listener);

    // assuming 4 bit per pixel max
    LOGE("Image Buffer Size: %d", res->width * res->height * 4);
    image_buffer_ = (uint8_t *)malloc(res->width * res->height * 4);
    assert(image_buffer_); // failed to allocate image buffer
}

ImageReader::~ImageReader() {
    assert(reader_);
    AImageReader_delete(reader_);

    if (image_buffer_) {
        free(image_buffer_);
    }
}

void ImageReader::ImageCallback(AImageReader *reader) {
    int32_t format;
    media_status_t status = AImageReader_getFormat(reader, &format);
    assert(status == AMEDIA_OK); // failed to get format

    if (format == AIMAGE_FORMAT_JPEG) {
        // Create a thread and write out the jpeg files
        AImage *image = NULL;
        status = AImageReader_acquireNextImage(reader, &image);
        assert(status == AMEDIA_OK); // image is not available

        int plane_count;
        status = AImage_getNumberOfPlanes(image, &plane_count);
        assert(status == AMEDIA_OK && plane_count == 1); // unsupported plane count

        uint8_t *data = NULL;
        int len;
        status = AImage_getPlaneData(image, 0, &data, &len);
        assert(status == AMEDIA_OK); // failed to get plane data

        AImage_delete(image);
    }
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
