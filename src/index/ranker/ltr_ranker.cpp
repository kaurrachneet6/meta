/**
 * @file ltr_ranker.cpp
 * @author Anthony Huang
 */

#include <cmath>
#include <fstream>
#include <cassert>
#include "meta/index/inverted_index.h"
#include "meta/index/ranker/ltr_ranker.h"
#include "meta/index/score_data.h"
#include "meta/math/fastapprox.h"

namespace meta
{
namespace index
{

const util::string_view ltr_ranker::id = "ltr_ranker";


ltr_ranker::ltr_ranker(std::string& weights_path, std::string& briefs_path)
{
    auto num_lines = filesystem::num_lines(weights_path);
    std::ifstream weights_in{weights_path};
    std::string line;
    for (std::size_t i = 0; i < num_lines; ++i)
    {
        std::getline(weights_in, line);
        weights.push_back(std::stod(line));
    }

    num_lines = filesystem::num_lines(briefs_path);
    std::ifstream briefs_in{briefs_path};
    for (std::size_t i = 0; i < num_lines; ++i)
    {
        std::getline(briefs_in, line);
        briefs.push_back(line);
    }

    assert(weights.size() == briefs.size());
    for (std::size_t i = 0; i < briefs.size(); ++i) {
        weights_map[briefs[i]] = weights[i];
    }
    assert(weights_map.size() == briefs.size());
}

ltr_ranker::ltr_ranker(std::istream& in)
{
    auto size = io::packed::read<std::size_t>(in);
    weights.resize(size);
    for (std::size_t i = 0; i < size; ++i)
        io::packed::read(in, weights[i]);

    size = io::packed::read<std::size_t>(in);
    briefs.resize(size);
    for (std::size_t i = 0; i < size; ++i)
        io::packed::read(in, briefs[i]);

    assert(weights.size() == briefs.size());
    for (std::size_t i = 0; i < briefs.size(); ++i) {
        weights_map[briefs[i]] = weights[i];
    }
    assert(weights_map.size() == briefs.size());
}

void ltr_ranker::save(std::ostream& out) const
{
    io::packed::write(out, id);

    io::packed::write(out, weights.size());
    for (const auto& weight : weights)
        io::packed::write(out, weight);

    io::packed::write(out, briefs.size());
    for (const auto& brief : briefs)
        io::packed::write(out, brief);
}

float ltr_ranker::score_one(const score_data& sd)
{
    float BM25_doc = bm25_ranker.score_one(sd);
    float ABS_doc = abs_ranker.score_one(sd);
    float DIR_doc = dir_ranker.score_one(sd);
    float JM_doc = jm_ranker.score_one(sd);

    float score = 0.0;
    score += weights_map["bm25_doc"] * BM25_doc;
    score += weights_map["abs_doc"] * ABS_doc;
    score += weights_map["dir_doc"] * DIR_doc;
    score += weights_map["jm_doc"] * JM_doc;

    return score;
}

template <>
std::unique_ptr<ranker> make_ranker<ltr_ranker>(const cpptoml::table& config)
{
    auto weights_path = config.get_as<std::string>("weights").value_or("");
    auto briefs_path = config.get_as<std::string>("briefs").value_or("");

    if (weights_path.length() == 0 || !filesystem::file_exists(weights_path))
        throw ranker_exception{"Invalid weights_path for ltr_ranker!"};

    if (briefs_path.length() == 0 || !filesystem::file_exists(briefs_path))
        throw ranker_exception{"Invalid briefs_path for ltr_ranker!"};

    return make_unique<ltr_ranker>(weights_path, briefs_path);
}
}
}
