#ifndef AAALGO_COLORHASH
#define AAALGO_COLORHASH

#include <opencv2/opencv.hpp>

namespace colorhash {

    class Histogram {
        unsigned c1, c2, c3, code;
        float smooth;
    public:
        Histogram (unsigned c1_ = 7, unsigned c2_ = 7, unsigned c3_ = 7, unsigned code_ = 0, float sm = 10)
            : c1(c1_), c2(c2_), c3(c3_), code(code_), smooth(sm) {
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
                    float v1 = (p[0] + 0.5) * (c1-1) / 256;
                    float v2 = (p[1] + 0.5) * (c2-1) / 256;
                    float v3 = (p[2] + 0.5) * (c3-1) / 256;
                    int q1 = int(floor(v1));
                    int q2 = int(floor(v2));
                    int q3 = int(floor(v3));
                    if (smooth == 0) {
                        unsigned off = q1 * c2 * c3
                                     + q2 * c3
                                     + q3;
                        hist[off] += 1.0;
                    }
                    else {
                        // do smoothing
                        float ws[8];
                        int offs[8];
                        unsigned o = 0;
                        for (int d1 = 0; d1 < 2; ++d1) { int b1 = q1 + d1;
                        for (int d2 = 0; d2 < 2; ++d2) { int b2 = q2 + d2;
                        for (int d3 = 0; d3 < 2; ++d3) { int b3 = q3 + d3;
                            offs[o] = b1 * c2 * c3
                                    + b2 * c3
                                    + b3;
                            if (b1 >= c1 || b2 >= c2 || b3 >= c3) throw 1;

                            ws[o] = (v1 - d1) * (v1 - d1)
                                  + (v2 - d2) * (v2 - d2)
                                  + (v3 - d3) * (v3 - d3);
                            ++o;
                        }}}
                        float sum = 0;
                        for (auto &w: ws) {
                            w = exp(-w/smooth);
                            sum += w;
                        }
                        for (auto &w: ws) {
                            w /= sum;
                        }
                        for (unsigned i = 0; i < 8; ++i) {
                            if (offs[i] >= 0) {
                                hist[offs[i]] += ws[i];
                            }
                        }
                    }
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
