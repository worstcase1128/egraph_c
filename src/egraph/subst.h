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
#ifndef MINDSPORE_EGRAPH_SUBST_H_
#define MINDSPORE_EGRAPH_SUBST_H_

#include <vector>
#include <optional>
#include <utility>

#include "egraph/enode.h"

namespace mindspore::egraph {

class Substitution {
 public:
  std::optional<EClassId> Insert(const VarPtr &var, EClassId eclass_id) {
    for (auto &[v, id] : var2id_) {
      if (v == var) {
        auto old_id = id;
        id = eclass_id;
        return old_id;
      }
    }
    var2id_.emplace_back(var, eclass_id);
    return std::nullopt;
  }

  std::optional<EClassId> Get(const VarPtr &var) const {
    for (auto &[v, id] : var2id_) {
      if (v == var || v->Matches(*var)) {
        return id;
      }
    }
    return std::nullopt;
  }

  std::optional<EClassId> var(const std::string &var_name) const {
    for (auto &[v, id] : var2id_) {
      if (v->name() == var_name) {
        return id;
      }
    }
    return std::nullopt;
  }

  const std::vector<std::pair<VarPtr, EClassId>> &items() const { return var2id_; }

  std::string ToString() const {
    std::string s = "{";
    for (auto &entry : var2id_) {
      s += "(";
      s += entry.first->ToString();
      s += " : ";
      s += std::to_string(entry.second);
      s += ") ";
    }
    s += "}";
    return s;
  }

 private:
  std::vector<std::pair<VarPtr, EClassId>> var2id_;
};

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_SUBST_H_