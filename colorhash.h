#ifndef AAALGO_COLORHASH
#define AAALGO_COLORHASH

#include <opencv2/opencv.hpp>

namespace colorhash {

    class Histogram {
        unsigned c1, c2, c3, code;
    public:
        Histogram (unsigned c1_ = 8, unsigned c2_ = 4, unsigned c3_ = 4, unsigned code_ = CV_BGR2Lab)
            : c1(c1_), c2(c2_), c3(c3_), code(code_) {
        }
        // returns histogram dimensions
        unsigned size () const {
            return c1 * c2 * c3;
        }
        // extract histogram, output address must be pre-allocated to hold >= size() floats
        void apply (cv::Mat &image, float *hist) const {
            if (image.type() != CV_8UC3) throw 0;
            cv::Mat im;
            if (code == 0) {
                im = image;
            }
            else {
                cv::cvtColor(image, im, code);
            }
            unsigned sz = size();
            std::fill(hist, hist + sz, 0);
            for (int y = 0; y < im.rows; ++y) {
                uint8_t const *p = im.ptr<uint8_t const>(y);
                for (int x = 0; x < im.cols; ++x) {
                    unsigned v1 = p[0];
                    unsigned v2 = p[1];
                    unsigned v3 = p[2];

                    unsigned off = (v1 * c1 / 256) * c2 * c3
                                 + (v2 * c2 / 256) * c3
                                 + (v3 * c3 / 256);
                    hist[off] += 1.0;
                    p += 3;
                }
            }
            unsigned total = im.rows * im.cols;
            for (unsigned i = 0; i < sz; ++i) {
                hist[i] /= total;
            }
        }
    };

    class Hash {
    public:
        // return hash size in bytes
        unsigned size () const;
        // output address must be pre-allocated to hold >= bytes() bytes
        virtual void apply (float const *hist, uint8_t *hash) const;
    };

}

#endif
