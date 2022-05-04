#pragma once

/**
 * A helper class to assist image size comparison, by comparing the absolute
 * size
 * regardless of the portrait or landscape mode.
 */
class DisplayDimension {
 public:
  DisplayDimension(int32_t w, int32_t h) : w_(w), h_(h), portrait_(false) {
    if (h > w) {
      // make it landscape
      w_ = h;
      h_ = w;
      portrait_ = true;
    }
  }

  DisplayDimension(const DisplayDimension& other) {
    w_ = other.w_;
    h_ = other.h_;
    portrait_ = other.portrait_;
  }

  DisplayDimension() {
    w_ = 0;
    h_ = 0;
    portrait_ = false;
  }

  DisplayDimension& operator=(const DisplayDimension& other) {
    w_ = other.w_;
    h_ = other.h_;
    portrait_ = other.portrait_;

    return (*this);
  }

  bool IsSameRatio(DisplayDimension& other) {
    return (w_ * other.h_ == h_ * other.w_);
  }
  bool operator>(DisplayDimension& other) {
    return (w_ >= other.w_) || (h_ >= other.h_);
  }
  bool operator==(DisplayDimension& other) {
    return w_ == other.w_ && h_ == other.h_ && portrait_ == other.portrait_;
  }
  DisplayDimension operator-(DisplayDimension& other) {
    return DisplayDimension(w_ - other.w_, h_ - other.h_);
  }

  void Flip() { portrait_ = !portrait_; }
  bool IsPortrait() { return portrait_; }
  int32_t width() { return w_; }
  int32_t height() { return h_; }
  int32_t org_width() { return (portrait_ ? h_ : w_); }
  int32_t org_height() { return (portrait_ ? w_ : h_); }

private:
  int32_t w_, h_;
  bool portrait_;
};
