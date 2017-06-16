#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <boost/progress.hpp>
#include <boost/program_options.hpp>
#include <kgraph.h>
#include <kgraph-data.h>
#include <fmt/format.h>

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;

namespace kgraph {
    namespace metric {
        /// L2 square distance.
        struct chi2 {
            template <typename T>
            /// L2 square distance.
            static float apply (T const *t1, T const *t2, unsigned dim) {
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    float v = float(t1[i]) - float(t2[i]);
                    float b = float(t1[i]) + float(t2[i]);
                    if (b > 0) {
                        r += v * v / b;
                    }
                }
                return r;
            }
        };

        struct dot {
            template <typename T>
            /// L2 square distance.
            static float apply (T const *t1, T const *t2, unsigned dim) {
                float r = 0;
                float l1 = 0, l2 = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    r += t1[i] * t2[i];
                    l1 += t1[i] * t1[i];
                    l2 += t2[i] * t2[i];
                }
                return -r/sqrt(l1 * l2);
            }
        };
    }
}

int main (int argc, char *argv[]) {
    string list;
    string matrix;
    string metric;
    unsigned N, K;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("list", po::value(&list)->default_value("list"), "")
        ("matrix", po::value(&matrix), "")
        (",N", po::value(&N)->default_value(100), "")
        (",K", po::value(&K)->default_value(10), "")
        ("metric,m", po::value(&metric)->default_value("chi2"), "")
        ;

    po::positional_options_description p;
    p.add("matrix", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || (vm.count("matrix") == 0)) {
        cout << "Usage:" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }

    vector<string> paths;
    {
        ifstream is(list.c_str());
        string path;
        while (is >> path) {
            paths.push_back(path);
        }
    }
    vector<unsigned> idx(paths.size());
    for (unsigned i = 0; i < idx.size(); ++i) idx[i] = i;
    std::random_shuffle(idx.begin(), idx.end());
    idx.resize(N);

    kgraph::Matrix<float> data;
    data.load_lshkit(matrix);
    if (data.size() != paths.size()) {
        cerr << "size not match" << endl;
        throw 0;
    }
    kgraph::MatrixOracle<float, kgraph::metric::l2sqr> l2oracle(data);
    kgraph::MatrixOracle<float, kgraph::metric::chi2> chi2oracle(data);
    kgraph::MatrixOracle<float, kgraph::metric::dot> dotoracle(data);
    kgraph::KGraph *index = kgraph::KGraph::create();
    kgraph::KGraph::IndexParams params;
    if (metric == "l2") {
        index->build(l2oracle, params, NULL);
    }
    else if (metric == "chi2") {
        index->build(chi2oracle, params, NULL);
    }
    else if (metric == "dot") {
        index->build(dotoracle, params, NULL);
    }
    else {
        throw 0;
    }
    cout << "<html><body><table border='1'>";
    vector<unsigned> nns(params.L);
    vector<float> dists(params.L);
    for (unsigned i: idx) {
        unsigned M, L;
        index->get_nn(i, &nns[0], &dists[0], &M, &L);
        cout << "<tr><td><img src='" << paths[i] << "'/><td>";
        for (unsigned j = 0; j < K; ++j) {
            cout << "<td><img src='" << paths[nns[j]] << "'/><br/>" << dists[j] << "<td>";
        }
        cout << "</tr>" << endl;
    }
    cout << "</table></body></html>" << endl;
    delete index;
}

