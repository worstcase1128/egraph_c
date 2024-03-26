/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_EGRAPH_EXTRACTOR_H_
#define MINDSPORE_EGRAPH_EXTRACTOR_H_

#include <vector>
#include <optional>
#include <functional>
#include <algorithm>
#include <utility>
#include <unordered_map>

#include "egraph/egraph.h"

namespace mindspore::egraph {

using Cost = uint64_t;

class CostModel {
 public:
  using ClassCostFunc = std::function<Cost(EClassId)>;

  // Calculates the cost of an enode whose children costs can be retrieved by cost_func.
  // For this to work properly, your cost function should be monotonic, i.e. it should
  // return a `Cost` greater than any of the child costs of the given enode.
  virtual Cost GetCost(const ENodePtr &enode, ClassCostFunc cost_func) = 0;

  // Calculates the total cost of a RecExpr.
  // As provided, this just recursively calls GetCost all the way down the RecExpr.
  Cost GetExprCost(const RecExpr &expr) {
    std::unordered_map<EClassId, Cost> costs;
    for (size_t i = 0; i < expr.size(); ++i) {
      auto &enode = expr[i];
      auto cost = GetCost(enode, [&costs](EClassId id) { return costs[id]; });
      costs.emplace(static_cast<EClassId>(i), cost);
    }
    return costs[expr.size() - 1];
  }
};

class Extractor {
 public:
  Extractor(const EGraph &egraph, std::unique_ptr<CostModel> cost_model)
      : egraph_(egraph), cost_model_(std::move(cost_model)) {
    FindCosts();
  }
  ~Extractor() = default;

  std::pair<RecExpr, Cost> FindBest(EClassId eclass) {
    std::vector<EClassId> ids;
    return FindBestWithIds(eclass, &ids);
  }

  // Find the cheapest e-node in the given e-class.
  ENodePtr FindBestNode(EClassId id) {
    auto iter = costs_.find(egraph_.Find(id));
    if (iter == costs_.end()) {
      return nullptr;
    }
    return iter->second.second;
  }

  /// Find the cost of the term that would be extracted from this e-class.
  std::optional<Cost> FindBestCost(EClassId id) {
    auto iter = costs_.find(egraph_.Find(id));
    if (iter == costs_.end()) {
      return std::nullopt;
    }
    return iter->second.first;
  }

 private:
  std::pair<RecExpr, Cost> FindBestWithIds(EClassId eclass, std::vector<EClassId> *ids) {
    RecExpr expr;
    std::unordered_map<EClassId, EClassId> added_memo;
    auto result = FindBestRecusively(eclass, ids, &added_memo, &expr);
    return {std::move(expr), result.second};
  }

  std::pair<EClassId, Cost> FindBestRecusively(EClassId eclass, std::vector<EClassId> *ids,
                                               std::unordered_map<EClassId, EClassId> *added_memo, RecExpr *expr) {
    auto id = egraph_.Find(eclass);
    auto [best_cost, best_node] = costs_[id];
    assert(best_node != nullptr);
    auto iter = added_memo->find(id);
    if (iter != added_memo->end()) {
      return {iter->second, best_cost};
    }
    auto node = best_node->Clone();
    node->ForEachChildren([this, ids, added_memo, expr](EClassId &child) {
      auto result = FindBestRecusively(child, ids, added_memo, expr);
      child = result.first;
    });
    auto id_expr = expr->Add(node);
    if (id_expr == static_cast<EClassId>(expr->size() - 1)) {
      ids->push_back(id);
    }
    added_memo->emplace(id, id_expr);
    return {id_expr, best_cost};
  }

  void FindCosts() {
    auto did_something = true;
    while (did_something) {
      did_something = false;
      for (auto &[id, eclass] : egraph_.eclasses()) {
        auto [cost, enode] = MakePass(eclass);
        if (enode == nullptr) {
          continue;
        }
        auto [iter, is_new] = costs_.emplace(id, std::make_pair(cost, enode));
        if (is_new) {
          did_something = true;
        } else if (cost < iter->second.first) {
          iter->second = {cost, enode};
          did_something = true;
        }
      }
    }
  }

  std::pair<Cost, ENodePtr> MakePass(const EClassPtr &eclass) {
    Cost min_cost = std::numeric_limits<Cost>::max();
    ENodePtr min_cost_node = nullptr;
    for (auto &enode : eclass->enodes()) {
      auto cost = GetTotalCost(enode);
      if (cost.has_value() && cost.value() < min_cost) {
        min_cost = cost.value();
        min_cost_node = enode;
      }
    }
    return {min_cost, min_cost_node};
  }

  std::optional<Cost> GetTotalCost(const ENodePtr &enode) {
    bool cost_not_found = std::any_of(enode->children().begin(), enode->children().end(),
                                      [this](EClassId id) { return costs_.find(egraph_.Find(id)) == costs_.end(); });
    if (cost_not_found) {
      return std::nullopt;
    }
    return cost_model_->GetCost(enode, [this](EClassId id) { return costs_[egraph_.Find(id)].first; });
  }

  const EGraph &egraph_;
  std::unique_ptr<CostModel> cost_model_;
  std::unordered_map<EClassId, std::pair<Cost, ENodePtr>> costs_;
};

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_EXTRACTOR_H_