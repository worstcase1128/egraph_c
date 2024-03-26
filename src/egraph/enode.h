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
#ifndef MINDSPORE_EGRAPH_ENODE_H_
#define MINDSPORE_EGRAPH_ENODE_H_

#include <algorithm>
#include <cinttypes>
#include <sstream>
#include <vector>
#include <memory>

namespace mindspore::egraph {

inline std::size_t HashCombine(std::size_t seed, std::size_t value) {
  return ((seed << 6) + (seed >> 2) + 0x9e3779b9 + value) ^ seed;
}

using EClassId = uint32_t;
using IdVec = std::vector<EClassId>;

class ENode;
using ENodePtr = std::shared_ptr<ENode>;

class ENode {
 public:
  const std::vector<EClassId> &children() const { return children_; }

  bool IsLeaf() const { return children_.empty(); }

  template <typename T>
  bool IsA() const {
    return dynamic_cast<const T *>(this) != nullptr;
  }

  template <typename F>
  void ForEachChildren(F &&func) {
    std::for_each(children_.begin(), children_.end(), func);
  }

  bool operator==(const ENode &other) const { return Matches(other) && (children_ == other.children_); }

  virtual size_t HashCode() const {
    size_t seed = 0;
    for (auto &id : children_) {
      seed = HashCombine(seed, static_cast<size_t>(id));
    }
    return seed;
  }

  virtual ENodePtr Clone() const = 0;

  virtual std::string ToString() const = 0;

  virtual bool Matches(const ENode &other) const = 0;

  virtual std::string GetOpName() const = 0;

 protected:
  std::vector<EClassId> children_;
};

class Var : public ENode {
 public:
  Var(const std::string &name) : name_(name) {}

  size_t HashCode() const override { return std::hash<std::string>{}(name_); }

  ENodePtr Clone() const override { return std::make_shared<Var>(name_); }

  std::string ToString() const override { return "$" + name_; }

  std::string GetOpName() const override { return "$" + name_; }

  bool Matches(const ENode &other) const override {
    auto node = dynamic_cast<const Var *>(&other);
    return node != nullptr && node->name_ == name_;
  }

  const std::string &name() const { return name_; }

 private:
  std::string name_;
};

using VarPtr = std::shared_ptr<Var>;

using NodeClassPair = std::pair<ENodePtr, EClassId>;

}  // namespace mindspore::egraph

namespace std {
// std::hash for ENodePtr.
template <>
struct hash<mindspore::egraph::ENodePtr> {
  std::size_t operator()(const mindspore::egraph::ENodePtr &enode) const noexcept { return enode->HashCode(); }
};

// std::equal_to for ENodePtr.
template <>
struct equal_to<mindspore::egraph::ENodePtr> {
  bool operator()(const mindspore::egraph::ENodePtr &a, const mindspore::egraph::ENodePtr &b) const {
    if (a == b) {
      return true;
    }
    if (a == nullptr || b == nullptr) {
      return false;
    }
    return (*a) == (*b);
  }
};

// std::hash for NodeClassPair.
template <>
struct hash<mindspore::egraph::NodeClassPair> {
  std::size_t operator()(const mindspore::egraph::NodeClassPair &pair) const noexcept { return pair.first->HashCode(); }
};
}  // namespace std

#endif  // MINDSPORE_EGRAPH_ENODE_H_