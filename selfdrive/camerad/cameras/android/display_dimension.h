#pragma once

/**
 * A helper class to assist image size comparison, by comparing the absolute
 * size regardless of the portrait or landscape mode.
 */
class DisplayDimension {
public:
  DisplayDimension(uint32_t w, uint32_t h) : w_(w), h_(h), portrait_(false) {
    if (h > w) {
      // make it landscape
      w_ = h;
      h_ = w;
      portrait_ = true;
    }
  }

  DisplayDimension(const DisplayDimension& other) : w_(other.w_), h_(other.h_), portrait_(other.portrait_) {}

  DisplayDimension() : w_(0), h_(0), portrait_(false) {}

  DisplayDimension& operator=(const DisplayDimension& other) {
    w_ = other.w_;
    h_ = other.h_;
    portrait_ = other.portrait_;
    return *this;
  }

  bool operator>(const DisplayDimension& other) const {
    return (w_ > other.w_) || (h_ > other.h_);
  }
  bool operator==(const DisplayDimension& other) const {
    return (w_ == other.w_) && (h_ == other.h_) && (portrait_ == other.portrait_);
  }
  DisplayDimension operator-(const DisplayDimension& other) const {
    return DisplayDimension(w_ - other.w_, h_ - other.h_);
  }

  bool IsSameRatio(const DisplayDimension& other) const {
    return (w_ * other.h_) == (h_ * other.w_);
  }

  void Flip() {
    portrait_ = !portrait_;
  }
  bool IsPortrait() const {
    return portrait_;
  }
  uint32_t width() const {
    return w_;
  }
  uint32_t height() const {
    return h_;
  }
  uint32_t org_width() const {
    return portrait_ ? h_ : w_;
  }
  uint32_t org_height() const {
    return portrait_ ? w_ : h_;
  }

private:
  int32_t w_, h_;
  bool portrait_;
};
