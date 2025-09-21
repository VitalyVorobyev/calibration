#pragma once

// std
#include <ostream>
#include <sstream>
#include <string>

namespace calib {

class StreamCapture {
  public:
    explicit StreamCapture(std::ostream& stream)
        : stream_(stream), old_buf_(stream.rdbuf(buffer_.rdbuf())) {}
    StreamCapture(const StreamCapture&) = delete;
    StreamCapture& operator=(const StreamCapture&) = delete;
    ~StreamCapture() { stream_.rdbuf(old_buf_); }

    [[nodiscard]] auto str() const -> std::string { return buffer_.str(); }

  private:
    std::ostream& stream_;
    std::ostringstream buffer_;
    std::streambuf* old_buf_;
};

}  // namespace calib
