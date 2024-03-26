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
#ifndef MINDSPORE_EGRAPH_EGRAPH_H_
#define MINDSPORE_EGRAPH_EGRAPH_H_

#include <any>
#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

#include "egraph/indexset.h"
#include "egraph/recexpr.h"
#include "egraph/eclass.h"
#include "egraph/unionfind.h"

namespace mindspore::egraph {

class EGraph;

class Analysis {
 public:
  enum class Ordering { kEqual, kLess, kGreater, kUnknown };

  Analysis() = default;

  virtual ~Analysis() = default;

  // Makes a new Analysis data for a given enode.
  virtual std::any MakeData(const EGraph &egraph, const ENodePtr &enode) { return {}; };

  // An optional hook that allows inspection before a `Merge` occurs.
  // `PreMerge` is called a lot, so doing anything significant
  // (like printing) will cause things to slow down.
  virtual void PreMerge(const EGraph &egraph, EClassId id1, EClassId id2) {}

  // Defines how to merge two `Data`s when their containing EClasses merge.
  // Since `merge` can modify `a`, let `a0`/`a1` be the value of
  // `a` before/after the call to `merge`, respectively.
  // The return value of `merge` should be the partial ordering of `a0` and `b`.
  // After `merge` returns, `a1` must be the least upper bound of `a0` and `b`.
  // In other words, `merge` must respect the following:
  // - if `a0 < b`, then `a1 = b`,
  // - if `a0 > b`, then `a0 = a1`,
  // - if `a0 == b`, then `a0 = a1`,
  // - if they cannot be compared, then `a1 >= a0` and `a1 >= b`.
  virtual Ordering MergeData(std::any *a, const std::any &b) { return Ordering::kEqual; };

  // A hook that allows the modification of the `EGraph`.
  // by default this does nothing.
  virtual void Modify(EClassId id, EGraph *egraph) {}
};

class EGraph {
 public:
  EGraph() : analysis_(std::make_unique<Analysis>()) {}

  EGraph(std::unique_ptr<Analysis> &&analysis) : analysis_(std::move(analysis)) {}

  virtual ~EGraph() = default;

  EClassId Find(EClassId id) const { return union_find_.Find(id); }

  std::optional<EClassId> Lookup(const ENodePtr &enode) const {
    // Canonicalize enode children.
    Canonicalize(enode);
    // Find eclass-id according the enode.
    auto iter = enodes_.find(enode);
    if (iter == enodes_.end()) {
      return std::nullopt;
    }
    // Canonicalize eclass.
    return Find(iter->second);
  }

  std::optional<EClassId> LookupExpr(const RecExpr &expr) const {
    auto &enodes = expr.enodes();
    std::vector<EClassId> new_ids;
    new_ids.reserve(enodes.size());
    for (auto &enode : enodes) {
      auto clone_node = enode->Clone();
      clone_node->ForEachChildren([&new_ids](EClassId &id) { id = new_ids[static_cast<size_t>(id)]; });
      auto eclass_id = Lookup(clone_node);
      if (!eclass_id.has_value()) {
        return std::nullopt;
      }
      new_ids.emplace_back(eclass_id.value());
    }
    return new_ids.back();
  }

  EClassId Add(const ENodePtr &enode) {
    auto result = Lookup(enode);
    if (result.has_value()) {
      return result.value();
    }
    auto id = union_find_.MakeSet();
    auto eclass = std::make_shared<EClass>(id, enode);
    eclass->data() = analysis_->MakeData(*this, enode);

    // Add this enode to the parent lists of its children.
    for (auto &child : enode->children()) {
      auto child_eclass = GetEClass(child);
      child_eclass->AddParent(enode, id);
    }

    // Map id to eclass.
    eclasses_.emplace(id, std::move(eclass));
    // Map enode to eclass.
    enodes_.emplace(enode, id);

    // Tell analysis that new eclass added.
    analysis_->Modify(id, this);
    return id;
  }

  EClassId AddExpr(const RecExpr &expr) {
    auto &enodes = expr.enodes();
    std::vector<EClassId> new_ids;
    new_ids.reserve(enodes.size());
    for (auto &enode : enodes) {
      auto new_node = enode->Clone();
      new_node->ForEachChildren([&new_ids](EClassId &id) { id = new_ids[static_cast<size_t>(id)]; });
      new_ids.emplace_back(Add(new_node));
    }
    return new_ids.back();
  }

  std::pair<EClassId, bool> Merge(EClassId x, EClassId y) {
    auto id1 = Find(x);
    auto id2 = Find(y);

    if (id1 == id2) {
      // Already merged.
      return {id1, false};
    }

    // Make sure class2 has fewer parents.
    auto class1 = GetEClass(id1);
    auto class2 = GetEClass(id2);
    auto parents1 = class1->parents().size();
    auto parents2 = class2->parents().size();
    if (parents1 < parents2) {
      std::swap(id1, id2);
      std::swap(class1, class2);
    }

    // Call analysis before merge.
    analysis_->PreMerge(*this, id1, id2);

    // Make id1 the new root.
    union_find_.UnionRoot(id1, id2);

    // Remove id2 from eclasses.
    eclasses_.erase(id2);

    // Add parents to pending.
    auto &class2_parents = class2->parents();
    pending_.insert(pending_.end(), class2_parents.begin(), class2_parents.end());

    // Merge analysis data.
    auto ordering = analysis_->MergeData(&class1->data(), class2->data());
    switch (ordering) {
      case Analysis::Ordering::kGreater:
        // class1->data() > class2->data()
        analysis_pending_.append(class2->parents());
        break;
      case Analysis::Ordering::kLess:
        // class1->data() < class2->data()
        analysis_pending_.append(class1->parents());
        break;
      case Analysis::Ordering::kUnknown:
        analysis_pending_.append(class1->parents());
        analysis_pending_.append(class2->parents());
        break;
      default:
        break;  // Do nothing if Equal.
    }

    // Merge nodes from class2 to class1.
    class1->MergeNodes(*class2);

    analysis_->Modify(id1, this);
    return {id1, true};
  }

  size_t Rebuild() {
    auto n_merges = MergePending();
    RebuildClasses();
    return n_merges;
  }

  size_t MergePending() {
    size_t n_merges = 0;
    while (!pending_.empty()) {
      while (!pending_.empty()) {
        auto [enode, class_id] = pending_.back();
        pending_.pop_back();
        Canonicalize(enode);
        auto [iter, new_class] = enodes_.emplace(enode, class_id);
        if (!new_class) {
          auto old_class_id = iter->second;
          iter->second = class_id;
          auto merged = Merge(old_class_id, class_id);
          if (std::get<bool>(merged)) {
            ++n_merges;
          }
        }
      }
      while (!analysis_pending_.empty()) {
        auto [enode, id] = analysis_pending_.pop();
        auto class_id = Find(id);
        auto node_data = analysis_->MakeData(*this, enode);
        auto eclass = GetEClass(class_id);
        assert(eclass != nullptr);
        auto ordering = analysis_->MergeData(&eclass->data(), node_data);
        if (ordering == Analysis::Ordering::kLess || ordering == Analysis::Ordering::kUnknown) {
          analysis_pending_.append(eclass->parents());
          analysis_->Modify(class_id, this);
        }
      }
    }
    return n_merges;
  }

  size_t RebuildClasses() {
    classes_by_op_.clear();
    size_t trimmed = 0;
    for (auto &entry : eclasses_) {
      auto &eclass = entry.second;
      auto old_size = eclass->enodes().size();
      for (auto &enode : eclass->enodes()) {
        Canonicalize(enode);
      }
      eclass->DeduplicateENodes();
      trimmed += old_size - eclass->enodes().size();

      for (auto &enode : eclass->enodes()) {
        classes_by_op_[enode->GetOpName()].insert(eclass->id());
      }
    }
    return trimmed;
  }

  EClassPtr GetEClass(EClassId id) const {
    auto iter = eclasses_.find(id);
    if (iter == eclasses_.end()) {
      return nullptr;
    }
    return iter->second;
  }

  const EClass &operator[](EClassId id) const {
    auto iter = eclasses_.find(id);
    return *iter->second;
  }

  const std::unordered_set<EClassId> &FindClassesByOp(const std::string &opname) const {
    static const std::unordered_set<EClassId> empty_set;
    auto iter = classes_by_op_.find(opname);
    if (iter == classes_by_op_.end()) {
      return empty_set;
    }
    return iter->second;
  }

  const std::unordered_map<EClassId, EClassPtr> &eclasses() const { return eclasses_; }

  const size_t size() const { return enodes_.size(); };

  void DebugPrint() const {
    std::cout << "E-Graph {\n";
    for (auto &e : eclasses_) {
      auto id = e.second->id();
      auto root_id = Find(id);
      std::cout << "  E-class: " << root_id << "/" << id << " {\n";
      for (auto &enode : e.second->enodes()) {
        std::cout << "    enode: " << enode->ToString() << " { ";
        for (auto &child : enode->children()) {
          std::cout << " " << child;
        }
        std::cout << " }\n";
      }
      std::cout << "  }\n";
    }
    std::cout << "}" << std::endl;
  }

  // Write e-graph to graphviz dot.
  void WriteDot(std::ostream &out) const {
    out << "digraph egraph {\n"
        << "compound=true\n"
        << "graph [style=\"filled,rounded\",color=lightgray]\n"
        << "node [shape=record,style=filled,fillcolor=white,width=0.1,height=0.3]\n"
        << "edge [arrowsize=0.5]\n";
    for (auto &e : eclasses_) {
      auto id = e.second->id();
      out << "subgraph cluster" << id << " {\n";
      for (auto &enode : e.second->enodes()) {
        out << NodeId(enode) << " [label=\"" << ToDotLabel(enode) << "\"]\n";
      }
      out << "}\n";
    }
    for (auto &e : eclasses_) {
      for (auto &enode : e.second->enodes()) {
        for (size_t i = 0; i < enode->children().size(); i++) {
          auto &child = enode->children()[i];
          auto eclass = GetEClass(child);
          out << NodeId(enode) << ":" << i << " -> " << NodeId(eclass->enodes()[0]) << " [lhead=cluster" << eclass->id()
              << "]\n";
        }
      }
    }
    out << "}\n";
  }

 protected:
  static inline std::string NodeId(const ENodePtr &enode) {
    return std::to_string(reinterpret_cast<uintptr_t>(enode.get()));
  }

  static inline std::string ToDotEscapedString(const std::string &input) {
    std::string output;
    for (char c : input) {
      if (c == '<' || c == '>' || c == '|' || c == '\"' || c == '\\') {
        output += '\\';
      }
      output += c;
    }
    return output;
  }

  static inline std::string ToDotLabel(const ENodePtr &enode) {
    std::string label = ToDotEscapedString(enode->ToString());
    for (size_t i = 0; i < enode->children().size(); i++) {
      label += "|<";
      label += std::to_string(i);
      label += ">";
    }
    return label;
  }

  void Canonicalize(const ENodePtr &enode) const {
    enode->ForEachChildren([this](EClassId &id) { id = this->Find(id); });
  }

 private:
  std::unique_ptr<Analysis> analysis_;
  UnionFind<EClassId> union_find_;
  std::unordered_map<EClassId, EClassPtr> eclasses_;
  std::unordered_map<ENodePtr, EClassId> enodes_;
  std::vector<NodeClassPair> pending_;
  IndexSet<NodeClassPair> analysis_pending_;
  std::unordered_map<std::string, std::unordered_set<EClassId>> classes_by_op_;
};

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_EGRAPH_H_