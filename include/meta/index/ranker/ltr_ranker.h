/**
 * @file ltr_ranker.h
 * @author Anthony Huang
 *
 * All files in META are released under the MIT license. For more details,
 * consult the file LICENSE in the root of the project.
 */

#ifndef META_LTR_RANKER_H_
#define META_LTR_RANKER_H_

#include <unordered_map>

#include "meta/index/ranker/ranker.h"
#include "meta/index/ranker/ranker_factory.h"
#include "meta/index/ranker/absulute_discount.h"
#include "meta/index/ranker/dirichlet_prior.h"
#include "meta/index/ranker/jelinek_mercer.h"
#include "meta/index/ranker/okapi_bm25.h"

namespace meta
{
namespace index
{

/**
 * The ranker with learn to rank.
 *
 * Required config parameters:
 * ~~~toml
 * [ranker]
 * method = "ltr_ranker"
 * weights = "path to file containing weights of trained model"
 * briefs = "path to file containing brief description of each weight in weights"
 * ~~~
 */
class ltr_ranker : public ranking_function
{
  public:
    /// The identifier for this ranker.
    const static util::string_view id;

    /**
     * @param weights_path Path to file containing value of weights
     * @param briefs_path Path to file containing brief of weights
     */
    ltr_ranker(std::string& weights_path, std::string& briefs_path);

    /**
     * Loads an ltr_ranker from a stream.
     * @param in The stream to read from
     */
    ltr_ranker(std::istream& in);

    void save(std::ostream& out) const override;

    /**
     * @param sd score_data for the current query
     */
    float score_one(const score_data& sd) override;

  private:
    okapi_bm25 bm25_ranker;
    absolute_discount abs_ranker;
    dirichlet_prior dir_ranker;
    jelinek_mercer jm_ranker;
    /// weights of trained learn to rank model
    std::vector<double> weights;
    /// brief of weights
    std::vector<std::string> briefs;
    /// mapping from brief to weight value
    std::unordered_map<std::string, double> weights_map;
};

/**
 * Specialization of the factory method used to create ltr_ranker
 */
template <>
std::unique_ptr<ranker> make_ranker<ltr_ranker>(const cpptoml::table&);
}
}
#endif
