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
#ifndef MINDSPORE_EGRAPH_RECEXPR_H_
#define MINDSPORE_EGRAPH_RECEXPR_H_

#include <vector>
#include <memory>

#include "egraph/enode.h"

namespace mindspore::egraph {

class RecExpr {
 public:
  using Id = EClassId;

  const std::vector<ENodePtr> &enodes() const { return enodes_; };

  Id Add(const ENodePtr &enode) {
    enodes_.emplace_back(enode);
    return static_cast<Id>(enodes_.size() - 1);
  }

  std::string ToString() const {
    if (enodes_.empty()) {
      return "()";
    }
    return ToString(enodes_.back());
  }

  const ENodePtr &operator[](size_t index) const { return enodes_[index]; }

  const size_t size() const { return enodes_.size(); }

  void DebugPrint() const {
    for (size_t i = 0; i < enodes_.size(); ++i) {
      auto &enode = enodes_[i];
      std::cout << i << ": " << enode->ToString();
      if (!enode->children().empty()) {
        std::cout << " {";
        for (auto c : enode->children()) {
          std::cout << " " << c;
        }
        std::cout << " }";
      }
      std::cout << '\n';
    }
  }

 protected:
  std::string ToString(const ENodePtr &enode) const {
    if (enode->IsLeaf()) {
      return enode->ToString();
    }
    auto str = "(" + enode->ToString();
    for (auto id : enode->children()) {
      auto child = enodes_[static_cast<size_t>(id)];
      str += " " + ToString(child);
    }
    str += ")";
    return str;
  }

 private:
  std::vector<ENodePtr> enodes_;
};

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_RECEXPR_H_