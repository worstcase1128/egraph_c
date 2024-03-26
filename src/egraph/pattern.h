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
#ifndef MINDSPORE_EGRAPH_PATTERN_H_
#define MINDSPORE_EGRAPH_PATTERN_H_

#include <vector>

#include "egraph/enode.h"
#include "egraph/recexpr.h"
#include "egraph/matcher.h"
#include "egraph/egraph.h"

namespace mindspore::egraph {

struct SearchMatches {
  /// The eclass id that these matches were found in.
  EClassId id;
  /// The matches themselves.
  std::vector<Substitution> substs;

  void DebugPrint() const {
    std::cout << "eclass: " << id << '\n';
    for (auto &subst : substs) {
      std::cout << "subst: " << subst.ToString() << '\n';
    }
  }
};

class Pattern {
 public:
  explicit Pattern(const RecExpr &expr) : expr_(expr) {
    MatcherCompiler compiler(expr_);
    matcher_ = compiler.compile();
  }

  Pattern(Pattern &&other) = default;
  Pattern(const Pattern &other) = default;
  ~Pattern() = default;

  std::vector<SearchMatches> Search(const EGraph &egraph) const {
    std::vector<SearchMatches> matches;
    auto &root = expr_.enodes().back();
    if (root->IsA<Var>()) {
      // Pattern root is Var, search all classes.
      for (auto &entry : egraph.eclasses()) {
        auto search_result = SearchEClass(egraph, entry.first);
        if (search_result.has_value()) {
          matches.emplace_back(std::move(search_result.value()));
        }
      }
    } else {
      // Pattern root is not Var, search classes according the op name.
      auto &classes = egraph.FindClassesByOp(root->GetOpName());
      if (!classes.empty()) {
        for (auto id : classes) {
          auto search_result = SearchEClass(egraph, id);
          if (search_result.has_value()) {
            matches.emplace_back(std::move(search_result.value()));
          }
        }
      }
    }
    return matches;
  }

  std::unordered_set<ENodePtr> vars() const {
    std::unordered_set<ENodePtr> var_set;
    for (auto &enode : expr_.enodes()) {
      if (enode->IsA<Var>()) {
        var_set.emplace(enode);
      }
    }
    return var_set;
  }

  std::string ToString() const { return expr_.ToString(); }

  void DebugPrint() const {
    std::cout << "Pattern: " << expr_.ToString() << '\n';
    expr_.DebugPrint();
    std::cout << "Matcher:\n";
    matcher_.DebugPrint();
  }

 protected:
  std::optional<SearchMatches> SearchEClass(const EGraph &egraph, EClassId id) const {
    auto substs = matcher_.Run(egraph, id);
    if (substs.empty()) {
      return std::nullopt;
    }
    return SearchMatches{.id = id, .substs = std::move(substs)};
  }

 private:
  RecExpr expr_;
  Matcher matcher_;
};

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_PATTERN_H_