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

#ifndef MINDSPORE_EGRAPH_INDEXSET_H_
#define MINDSPORE_EGRAPH_INDEXSET_H_

#include <algorithm>
#include <unordered_set>
#include <vector>
#include <utility>
#include <memory>

namespace mindspore {
// IndexSet is a C++ version of Rust IndexSet.
template <class T, class Hash = std::hash<T>, class KeyEqual = std::equal_to<T>>
class IndexSet {
 public:
  using element_type = T;
  using hasher = Hash;
  using equal = KeyEqual;
  using indexed_type = std::vector<element_type>;
  using iterator = typename indexed_type::iterator;
  using const_iterator = typename indexed_type::const_iterator;
  using set_type = std::unordered_set<element_type, hasher, equal>;
  using index_set_type = IndexSet<element_type, hasher, equal>;

  IndexSet() = default;
  ~IndexSet() = default;

  IndexSet(const IndexSet &other) = default;
  IndexSet(IndexSet &&other) = default;
  IndexSet &operator=(const IndexSet &other) = default;
  IndexSet &operator=(IndexSet &&other) = default;

  bool insert(const element_type &e) {
    auto result = set_data_.emplace(e);
    if (result.second) {
      indexed_data_.emplace_back(e);
    }
    return result.second;
  }

  void append(const indexed_type &vec) {
    for (auto &item : vec) {
      (void)insert(item);
    }
  }

  element_type pop() {
    element_type item = indexed_data_.back();
    indexed_data_.pop_back();
    set_data_.erase(item);
    return item;
  }

  std::size_t size() const { return indexed_data_.size(); }

  bool empty() const { return indexed_data_.empty(); }

  void clear() {
    indexed_data_.clear();
    set_data_.clear();
  }

  bool contains(const element_type &e) const { return (set_data_.find(e) != set_data_.end()); }

  iterator begin() { return indexed_data_.begin(); }
  iterator end() { return indexed_data_.end(); }

  const_iterator begin() const { return indexed_data_.cbegin(); }
  const_iterator end() const { return indexed_data_.cend(); }

  const_iterator cbegin() const { return indexed_data_.cbegin(); }
  const_iterator cend() const { return indexed_data_.cend(); }

 private:
  set_type set_data_;
  indexed_type indexed_data_;
};
}  // namespace mindspore

#endif  // MINDSPORE_EGRAPH_INDEXSET_H_
