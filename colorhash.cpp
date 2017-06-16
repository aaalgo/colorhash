#include <array>
#include "colorhash.h"

namespace colorhash {

static unsigned constexpr WEB_COLOR = 216;

class WebColorHistogram: Histogram {
    cv::Mat lab;
public:
    WebColorHistogram () {
        cv::Mat rgb(1, WebRGB.size(), CV_8UC3);
        uint8_t *p = palette.ptr<uint8_t>(0);
        for (uint32_t v: WebRGB) {
            *p = v % 0xFF;  // B
            ++p;
            v /= 0x100;
            *p = v % 0xFF;  // G
            ++p;
            v /= 0x100;
            *p = v % 0xFF;  // R
            ++p;
        }
        cv::cvtColor(rgb, lab, CV_BGR2Lab);
        {
            rgb.reshape(6*3);
            cv::Mat big;
            cv::resize(rgb, big, Size(), 10, 10, cv::INTER_NEAREST);
            cv::imwrite("palette.jpg", big);
        }
    }

    virtual unsigned size () const {
        return WEB_COLOR;
    }

    virtual void apply (cv::Mat &image, float *hist) const {
        int count[WEB_COLOR];
    }



};


}
