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

float l2 (vector<float> const &o1, vector<float> const &o2) {
    float v = 0;
    for (unsigned i = 0; i < o1.size(); ++i) {
        float d = o1[i] - o2[i];
        v += d * d;
    }
    return v;
}

int main (int argc, char *argv[]) {
    string list;
    string matrix;
    unsigned N, K;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("list", po::value(&list)->default_value("list"), "")
        ("matrix", po::value(&matrix), "")
        (",N", po::value(&N)->default_value(100), "")
        (",K", po::value(&K)->default_value(10), "")
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

    kgraph::Matrix<float> data;
    data.load_lshkit(matrix);
    if (data.size() != paths.size()) {
        cerr << "size not match" << endl;
        throw 0;
    }
    kgraph::MatrixOracle<float, kgraph::metric::l2sqr> oracle(data);
    kgraph::KGraph *index = kgraph::KGraph::create();
    kgraph::KGraph::IndexParams params;
    index->build(oracle, params, NULL);
    cout << "<html><body><table border='1'>";
    vector<unsigned> idx(paths.size());
    for (unsigned i = 0; i < idx.size(); ++i) idx[i] = i;
    std::random_shuffle(idx.begin(), idx.end());
    idx.resize(N);
    vector<unsigned> nns(params.L);
    vector<float> dists(params.L);
    for (unsigned i: idx) {
        unsigned M, L;
        index->get_nn(i, &nns[0], &dists[0], &M, &L);
        cout << "<tr><td><img src='" << paths[i] << "'/><td>";
        for (unsigned j = 0; j < K; ++j) {
            cout << "<tr><td><img src='" << paths[nns[j]] << "'/><br/>" << dists[j] << "<td>";
        }
        cout << "</tr>" << endl;
    }
    cout << "</table></body></html>" << endl;
    delete index;
}

