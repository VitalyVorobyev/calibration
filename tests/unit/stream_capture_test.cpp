#include "calib/io/stream_capture.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

TEST(StreamCapture, CapturesAndRestoresStreamBuffer) {
    std::ostringstream stream;

    {
        calib::StreamCapture capture(stream);
        stream << "hello" << ' ' << "world";
        EXPECT_EQ(capture.str(), "hello world");
    }

    stream << "done";
    EXPECT_EQ(stream.str(), "done");
}
