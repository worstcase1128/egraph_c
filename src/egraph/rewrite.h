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
#ifndef MINDSPORE_EGRAPH_REWRITE_H_
#define MINDSPORE_EGRAPH_REWRITE_H_

#include <atomic>
#include <string>
#include <vector>
#include <functional>
#include <memory>

#include "egraph/recexpr.h"
#include "egraph/pattern.h"

namespace mindspore::egraph {

class Applier {
 public:
  Applier() = default;
  virtual ~Applier() = default;

  // Apply many substititions.
  // This method should call `ApplyOne` for each match and then unify the results with the matched eclass.
  // This should return a list of Ids where the merge actually did something.
  // The default implementation does this and should suffice for most use cases.
  virtual std::vector<EClassId> Apply(const std::vector<SearchMatches> &matches, EGraph *egraph) const {
    std::vector<EClassId> applied;
    for (auto &m : matches) {
      for (auto &subst : m.substs) {
        auto ids = ApplyOne(subst, egraph, m.id);
        for (auto id : ids) {
          auto [merged_id, merged] = egraph->Merge(id, m.id);
          if (merged) {
            applied.emplace_back(merged_id);
          }
        }
      }
    }
    return applied;
  }

  // Apply a single substitition.
  // An `Applier` should only add things to the egraph here, _not_ merge them with the id `eclass`.
  // That is the responsibility of the `Apply` method. The `eclass` parameter allows the implementer
  // to inspect the eclass where the match was found if they need to.
  // This should return a list of Ids of things you'd like to be merged with `eclass`.
  // There can be zero, one, or many.
  virtual std::vector<EClassId> ApplyOne(const Substitution &subst, EGraph *egraph, EClassId eclass) const = 0;

  // Returns a list of variables that this Applier assumes are bound.
  // We will check that the corresponding `Searcher` binds those variables.
  // By default this return an empty `Vec`, which basically turns off the checking.
  virtual std::vector<ENodePtr> vars() const { return {}; }

  // ToString for debug purpose.
  virtual std::string ToString() const = 0;
};

class ExprApplier : public Applier {
 public:
  explicit ExprApplier(const RecExpr &expr) : expr_(expr) {}
  explicit ExprApplier(RecExpr &&expr) : expr_(std::move(expr)) {}
  ~ExprApplier() override = default;

  std::vector<EClassId> ApplyOne(const Substitution &subst, EGraph *egraph, EClassId) const override {
    std::vector<EClassId> ids;
    ids.reserve(expr_.size());
    for (size_t i = 0; i < expr_.size(); ++i) {
      auto &enode = expr_[i];
      if (enode->IsA<Var>()) {
        auto var_id = subst.Get(std::static_pointer_cast<Var>(enode));
        assert(var_id.has_value());
        ids.emplace_back(var_id.value());
      } else {
        auto new_node = enode->Clone();
        new_node->ForEachChildren([&ids](EClassId &id) { id = ids[id]; });
        ids.emplace_back(egraph->Add(new_node));
      }
    }
    return {ids.back()};
  }

  std::vector<ENodePtr> vars() const override {
    std::vector<ENodePtr> var_list;
    for (auto &enode : expr_.enodes()) {
      if (enode->IsA<Var>()) {
        var_list.emplace_back(enode);
      }
    }
    return var_list;
  }

  std::string ToString() const override { return expr_.ToString(); }

 private:
  RecExpr expr_;
};

using ConditionFunc = std::function<bool(const Substitution &, const EGraph &, EClassId)>;

class ConditionalApplier : public Applier {
 public:
  ConditionalApplier(std::unique_ptr<Applier> applier, ConditionFunc condition_func)
      : applier_(std::move(applier)), condition_func_(condition_func) {}

  ConditionalApplier(const RecExpr &expr, ConditionFunc condition_func)
      : ConditionalApplier(std::make_unique<ExprApplier>(expr), condition_func) {}

  ~ConditionalApplier() override = default;

  std::vector<EClassId> ApplyOne(const Substitution &subst, EGraph *egraph, EClassId eclass) const override {
    if (condition_func_(subst, *egraph, eclass)) {
      return applier_->ApplyOne(subst, egraph, eclass);
    }
    return {};
  }

  std::vector<ENodePtr> vars() const override { return applier_->vars(); }

  // ToString for debug purpose.
  std::string ToString() const override { return applier_->ToString() + " with condition"; }

 private:
  std::unique_ptr<Applier> applier_;
  ConditionFunc condition_func_;
};

using ApplyFunc = std::function<std::vector<EClassId>(const Substitution &, EGraph *, EClassId)>;

class DynamicApplier : public Applier {
 public:
  explicit DynamicApplier(ApplyFunc apply_func) : apply_func_(apply_func) {}

  ~DynamicApplier() override = default;

  std::vector<EClassId> ApplyOne(const Substitution &subst, EGraph *egraph, EClassId eclass) const override {
    return apply_func_(subst, egraph, eclass);
  }

  std::string ToString() const override { return "DynamicApplier"; }

 private:
  ApplyFunc apply_func_;
};

class Rewrite {
 public:
  using Id = uint32_t;

  // Basic Rewrite.
  Rewrite(const std::string &name, const RecExpr &lhs, const RecExpr &rhs)
      : id_(NewId()), name_(name), lhs_(lhs), rhs_(std::make_unique<ExprApplier>(rhs)) {}

  // Conditional Rewrite.
  Rewrite(const std::string &name, const RecExpr &lhs, const RecExpr &rhs, ConditionFunc cond)
      : id_(NewId()), name_(name), lhs_(lhs), rhs_(std::make_unique<ConditionalApplier>(rhs, cond)) {}

  // Dynamic Rewrite.
  Rewrite(const std::string &name, const RecExpr &lhs, ApplyFunc apply_func)
      : id_(NewId()), name_(name), lhs_(lhs), rhs_(std::make_unique<DynamicApplier>(apply_func)) {}

  Rewrite(Rewrite &&other) = default;

  ~Rewrite() = default;

  Rewrite &operator=(Rewrite &&other) = default;

  bool CheckVariables() const {
    auto var_set = lhs_.vars();
    auto vars = rhs_->vars();
    for (auto &var : vars) {
      if (var_set.find(var) == var_set.end()) {
        std::cout << "var " << var->ToString() << " not found in pattern!\n";
        return false;
      }
    }
    return true;
  }

  const std::string &name() const { return name_; }

  const Id id() const { return id_; }

  std::vector<SearchMatches> Search(const EGraph &egraph) const { return lhs_.Search(egraph); }

  std::vector<EClassId> Apply(const std::vector<SearchMatches> &matches, EGraph *egraph) const {
    return rhs_->Apply(matches, egraph);
  }

  std::string ToString() const {
    return "[" + std::to_string(id_) + "] " + name_ + ": " + lhs_.ToString() + " => " + rhs_->ToString();
  }

 private:
  static Id NewId() {
    static std::atomic<Id> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
  }

  Id id_;
  std::string name_;
  Pattern lhs_;
  std::unique_ptr<Applier> rhs_;
};

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_REWRITE_H_