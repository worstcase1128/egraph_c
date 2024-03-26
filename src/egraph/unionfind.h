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
#ifndef MINDSPORE_EGRAPH_UNIONFIND_H_
#define MINDSPORE_EGRAPH_UNIONFIND_H_

#include "egraph/eclass.h"
#include "egraph/unionfind.h"

namespace mindspore::egraph {

template <typename Id>
class UnionFind {
 public:
  Id MakeSet() {
    auto id = static_cast<Id>(parents_.size());
    parents_.push_back(id);
    return id;
  }

  Id Find(Id id) const {
    auto current = id;
    while (current != GetParent(current)) {
      auto grandparent = GetParent(GetParent(current));
      SetParent(current, grandparent);
      current = grandparent;
    }
    return current;
  }

  Id Union(Id x, Id y) {
    auto x_root = Find(x);
    auto y_root = Find(y);
    if (x_root != y_root) {
      SetParent(y_root, x_root);
    }
    return x_root;
  }

  Id UnionRoot(Id root1, Id root2) {
    if (root1 != root2) {
      SetParent(root2, root1);
    }
    return root1;
  }

 protected:
  Id GetParent(Id id) const { return parents_[static_cast<size_t>(id)]; }

  void SetParent(Id id, Id parent) const { parents_[static_cast<size_t>(id)] = parent; }

 private:
  mutable std::vector<Id> parents_;
};

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_UNIONFIND_H_