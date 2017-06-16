#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <boost/progress.hpp>
#include <boost/program_options.hpp>
#include <kgraph.h>
#include <kgraph-data.h>
#include <fmt/format.h>
#include "colorhash.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;

float LimitSize (cv::Mat input, int max_size, cv::Mat *output) {
    if (input.rows == 0) {
        *output = cv::Mat();
        return 0;
    }
    float scale = 1.0;
    cv::Size sz(input.cols, input.rows);
    int maxs = std::min(sz.width, sz.height);

    if ((max_size > 0) && (maxs > max_size)) {
        scale = 1.0 * max_size / maxs;
        sz = cv::Size(sz.width * max_size / maxs, sz.height * max_size / maxs);
    }
    if ((sz.width != input.cols) || (sz.height != input.rows)) {
        cv::Mat tmp;
        cv::resize(input, tmp, sz);
        input = tmp;
    }
    *output = input;
    return scale;
}

float l2 (vector<float> const &o1, vector<float> const &o2) {
    float v = 0;
    for (unsigned i = 0; i < o1.size(); ++i) {
        float d = o1[i] - o2[i];
        v += d * d;
    }
    return v;
}

int main (int argc, char *argv[]) {
    int c1, c2, c3;
    string list;
    string cs;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("c1", po::value(&c1)->default_value(6), "")
        ("c2", po::value(&c2)->default_value(6), "")
        ("c3", po::value(&c3)->default_value(6), "")
        //("code", po::value(&code)->default_value(CV_BGR2Lab), "")
        ("cs", po::value(&cs)->default_value("rgb"), "rgb, hsv, lab")
        ("list", po::value(&list)->default_value("list"), "")
        ;

    po::positional_options_description p;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << "Usage:" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }

    int code = 0;
    if (cs == "rgb") {
        code = 0;
    }
    else if (cs == "lab") {
        code = CV_BGR2Lab;
    }
    else if (cs == "hsv") {
        code = CV_BGR2HSV;
    }

    colorhash::Histogram hist(c1, c2, c3, code);
    vector<string> paths;

    {
        ifstream is(list.c_str());
        string path;
        while (is >> path) {
            paths.push_back(path);
        }
    }
    boost::progress_display progress(paths.size());
    kgraph::Matrix<float> data(paths.size(), hist.size());
    vector<unsigned> bad;
#pragma omp parallel for
    for (unsigned i = 0; i < paths.size(); ++i) {
        cv::Mat image = cv::imread(paths[i], CV_LOAD_IMAGE_COLOR);
        if (image.total() == 0) {
#pragma omp critical
            bad.push_back(i);
        }
        else {
            cv::Mat thumb;
            LimitSize(image, 256, &thumb);
            hist.apply(thumb, data[i]);
        }
#pragma omp critical
        ++progress;
    }
    data.save_lshkit(fmt::format("{}{}{}{}", cs, c1, c2, c3));
    for (auto v: bad) {
        cerr << paths[v] << endl;
    }
    return 0;

    /*
    kgraph::VectorOracle<vector<vector<float>>, vector<float>> oracle(ft, l2);
    kgraph::KGraph *index = KGraph::create();
    kgraph::KGraph::IndexParams params;
    index->build(oracle, params, NULL);
    delete index;
    */
}

