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
#ifndef MINDSPORE_EGRAPH_ECLASS_H_
#define MINDSPORE_EGRAPH_ECLASS_H_

#include <any>
#include <memory>
#include <utility>
#include <unordered_set>

#include "egraph/enode.h"

namespace mindspore::egraph {

class EClass {
 public:
  EClass(EClassId id, const ENodePtr &enode) : id_(id), enodes_({enode}) {}
  ~EClass() = default;

  void AddParent(const ENodePtr &enode, EClassId id) { parents_.emplace_back(enode, id); }

  void MergeNodes(const EClass &other) {
    enodes_.insert(enodes_.end(), other.enodes_.begin(), other.enodes_.end());
    parents_.insert(parents_.end(), other.parents_.begin(), other.parents_.end());
  }

  void DeduplicateENodes() {
    std::unordered_set<ENodePtr> node_set;
    node_set.insert(enodes_.begin(), enodes_.end());
    enodes_.clear();
    enodes_.insert(enodes_.end(), node_set.begin(), node_set.end());
  }

  const std::vector<ENodePtr> &enodes() const { return enodes_; }

  const std::vector<NodeClassPair> &parents() const { return parents_; }

  EClassId id() const { return id_; }

  const std::any &data() const { return data_; }

  std::any &data() { return data_; }

 private:
  EClassId id_;
  std::vector<ENodePtr> enodes_;
  std::vector<NodeClassPair> parents_;
  std::any data_;
};

using EClassPtr = std::shared_ptr<EClass>;

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_ECLASS_H_