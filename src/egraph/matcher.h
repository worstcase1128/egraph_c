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
#ifndef MINDSPORE_EGRAPH_MATCHER_H_
#define MINDSPORE_EGRAPH_MATCHER_H_

#include <map>
#include <queue>
#include <cassert>
#include <variant>

#include "egraph/recexpr.h"
#include "egraph/subst.h"
#include "egraph/egraph.h"

namespace mindspore::egraph {

using Reg = EClassId;

using VarToReg = std::unordered_map<ENodePtr, Reg>;

struct CompareInstruction {
  Reg i;
  Reg j;
  CompareInstruction(Reg i, Reg j) : i(i), j(j) {}
  ~CompareInstruction() = default;
};

struct BindInstruction {
  Reg i;
  ENodePtr node;
  Reg out;
  BindInstruction(Reg i, const ENodePtr &node, Reg out) : i(i), node(node), out(out) {}
  ~BindInstruction() = default;
};

using Instruction = std::variant<CompareInstruction, BindInstruction>;

class Matcher {
 public:
  Matcher() = default;
  ~Matcher() = default;
  Matcher(Matcher &&other) = default;
  Matcher &operator=(Matcher &&other) = default;

  Matcher(std::vector<Instruction> &&insts, std::map<EClassId, RecExpr> &&ground_terms, Substitution &&subst)
      : instructions_(std::move(insts)), ground_terms_(std::move(ground_terms)), subst_(std::move(subst)) {}

  std::vector<Substitution> Run(const EGraph &egraph, EClassId eclass) const {
    std::vector<Reg> regs;
    // First, we search ground terms.
    for (auto &gterm : ground_terms_) {
      auto result = egraph.LookupExpr(gterm.second);
      if (!result.has_value()) {
        // No matches if any of ground terms not found in the graph.
        return {};
      }
      regs.emplace_back(result.value());
    }
    // And then execute instructions.
    regs.emplace_back(eclass);
    std::vector<Substitution> substs;
    auto pc = instructions_.begin();
    Execute(egraph, pc, &substs, &regs);
    return substs;
  }

  void DebugPrint() const {
    std::cout << "ground_terms " << ground_terms_.size() << '\n';
    for (auto &t : ground_terms_) {
      std::cout << "  " << t.first << ": " << t.second.ToString() << '\n';
    }
    std::cout << "instructions " << instructions_.size() << '\n';
    for (auto &inst : instructions_) {
      if (std::holds_alternative<CompareInstruction>(inst)) {
        auto &compare = std::get<CompareInstruction>(inst);
        std::cout << "  Compare: i=" << compare.i << " j=" << compare.j << '\n';
      } else {
        auto &bind = std::get<BindInstruction>(inst);
        std::cout << "     Bind: i=" << bind.i << " node=" << bind.node->ToString() << " out=" << bind.out << '\n';
      }
    }
    std::cout << "substituions " << subst_.ToString() << std::endl;
  }

 protected:
  void Execute(const EGraph &egraph, std::vector<Instruction>::const_iterator pc, std::vector<Substitution> *substs,
               std::vector<Reg> *regs) const {
    for (; pc != instructions_.end(); ++pc) {
      auto &inst = *pc;
      if (std::holds_alternative<BindInstruction>(inst)) {
        // Bind.
        auto &bind = std::get<BindInstruction>(inst);
        auto eclass = egraph.GetEClass(regs->at(bind.i));
        assert(eclass != nullptr);
        for (auto &enode : eclass->enodes()) {
          if (enode->Matches(*bind.node)) {
            regs->resize(static_cast<size_t>(bind.out));
            regs->insert(regs->end(), enode->children().begin(), enode->children().end());
            Execute(egraph, pc + 1, substs, regs);
          }
        }
        return;
      }
      // Compare.
      assert(std::holds_alternative<CompareInstruction>(inst));
      auto &compare = std::get<CompareInstruction>(inst);
      if (egraph.Find(regs->at(compare.i)) != egraph.Find(regs->at(compare.j))) {
        return;
      }
    }
    // Find a match, add a new substitution.
    Substitution subst;
    for (auto &v2r : subst_.items()) {
      subst.Insert(v2r.first, regs->at(v2r.second));
    }
    substs->emplace_back(std::move(subst));
  }

 private:
  std::vector<Instruction> instructions_;
  std::map<EClassId, RecExpr> ground_terms_;
  Substitution subst_;
};

class MatcherCompiler {
 public:
  struct Todo {
    Reg reg;
    bool is_ground;
    size_t location;
    ENodePtr pattern;
  };

  struct TodoCompare {
    bool operator()(const Todo &lhs, const Todo &rhs) const {
      if (lhs.is_ground != rhs.is_ground) {
        // Ground term has higher priority.
        return rhs.is_ground;
      }
      const bool lhs_is_var = lhs.pattern->IsA<Var>();
      const bool rhs_is_var = rhs.pattern->IsA<Var>();
      if (lhs_is_var != rhs_is_var) {
        // Var is higher priority than ENode.
        return rhs_is_var;
      }
      // Fewer children means higher priority.
      return lhs.pattern->children().size() > rhs.pattern->children().size();
    }
  };

  using TodoList = std::priority_queue<Todo, std::vector<Todo>, TodoCompare>;

  explicit MatcherCompiler(const RecExpr &pattern_expr) : pattern_expr_(pattern_expr) {}
  ~MatcherCompiler() = default;

  Matcher compile() {
    // Mark ground term nodes.
    std::vector<bool> is_ground(pattern_expr_.size(), false);
    for (size_t i = 0; i < pattern_expr_.size(); ++i) {
      auto &enode = pattern_expr_[i];
      if (!enode->IsA<Var>()) {
        // ENode is a ground term if all of its children are ground terms.
        auto &children = enode->children();
        is_ground[i] = std::all_of(children.begin(), children.end(),
                                   [&is_ground](EClassId id) { return is_ground[static_cast<size_t>(id)]; });
      }
    }
    // Find locations of the ground terms.
    auto ground_locs = GetGroundLocs(is_ground);
    // Build ground terms.
    std::map<EClassId, RecExpr> ground_terms;
    for (size_t i = 0; i < ground_locs.size(); i++) {
      auto loc = ground_locs[i];
      if (loc.has_value()) {
        auto r = loc.value();
        RecExpr &expr = ground_terms[r];
        BuildGroundTerms(i, &expr);
      }
    }
    // Put root node (the last node) to todo list.
    todo_list_.emplace(Todo{.reg = IncOut(),
                            .is_ground = is_ground.back(),
                            .location = pattern_expr_.size() - 1,
                            .pattern = pattern_expr_.enodes().back()->Clone()});
    // Generate instructions.
    std::vector<Instruction> instructions;
    while (!todo_list_.empty()) {
      auto todo = todo_list_.top();
      todo_list_.pop();
      if (todo.pattern->IsA<Var>()) {
        // Var.
        auto iter = var2reg_.find(todo.pattern);
        if (iter != var2reg_.end()) {
          instructions.emplace_back(CompareInstruction(todo.reg, iter->second));
        } else {
          var2reg_.emplace(todo.pattern, todo.reg);
        }
      } else {
        // ENode.
        auto &loc = ground_locs[todo.location];
        if (loc.has_value()) {
          // Add Compare for ground term node.
          instructions.emplace_back(CompareInstruction(todo.reg, loc.value()));
          continue;
        }
        // Add children to todo list.
        auto out = out_;
        for (auto &child : todo.pattern->children()) {
          size_t child_index = static_cast<size_t>(child);
          todo_list_.emplace(Todo{.reg = IncOut(),
                                  .is_ground = is_ground[child_index],
                                  .location = child_index,
                                  .pattern = pattern_expr_[child_index]->Clone()});
        }
        // Zero out the children so Bind can use it to sort.
        todo.pattern->ForEachChildren([](EClassId &id) { id = 0; });
        instructions.emplace_back(BindInstruction{todo.reg, todo.pattern, out});
      }
    }
    // Generate substituion from var2reg_.
    Substitution subst;
    for (auto &[v, r] : var2reg_) {
      subst.Insert(std::static_pointer_cast<Var>(v), r);
    }
    // Construct the matcher.
    return {std::move(instructions), std::move(ground_terms), std::move(subst)};
  }

 protected:
  std::vector<std::optional<Reg>> GetGroundLocs(const std::vector<bool> &is_ground) {
    std::vector<std::optional<Reg>> ground_locs(pattern_expr_.size(), std::nullopt);
    for (size_t i = 0; i < pattern_expr_.size(); ++i) {
      auto &enode = pattern_expr_[i];
      if (enode->IsA<Var>() || is_ground[i]) {
        continue;
      }
      for (auto c : enode->children()) {
        // If a ground pattern is a child of a non-ground pattern,
        // we load the ground pattern.
        size_t index = static_cast<size_t>(c);
        if (is_ground[index] && !ground_locs[index].has_value()) {
          auto &child_node = pattern_expr_[index];
          assert(!child_node->IsA<Var>());
          ground_locs[index] = IncOut();
        }
      }
    }
    // Check if root is ground term.
    if (is_ground.back() && !pattern_expr_.enodes().back()->IsA<Var>()) {
      ground_locs.back() = IncOut();
    }
    return ground_locs;
  }

  void BuildGroundTerms(size_t loc, RecExpr *expr) {
    auto &node = pattern_expr_[loc];
    assert(!node->IsA<Var>());
    auto enode = node->Clone();
    enode->ForEachChildren([this, expr](EClassId &id) {
      BuildGroundTerms(static_cast<size_t>(id), expr);
      id = static_cast<EClassId>(expr->size() - 1);
    });
    expr->Add(enode);
  }

  Reg IncOut() {
    auto old_out = out_;
    ++out_;
    return old_out;
  }

 private:
  const RecExpr &pattern_expr_;
  VarToReg var2reg_;
  TodoList todo_list_;
  Reg out_ = 0;
};

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_MATCHER_H_