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
#ifndef MINDSPORE_EGRAPH_RUNNER_H_
#define MINDSPORE_EGRAPH_RUNNER_H_

#include <chrono>
#include <vector>
#include <limits>
#include <optional>
#include <utility>
#include <numeric>
#include <algorithm>
#include <unordered_map>

#include "egraph/egraph.h"
#include "egraph/rewrite.h"

namespace mindspore::egraph {

enum class StopReason {
  // The e-graph is saturated.
  kSaturated,
  // The iteration limit was hit.
  kIterationLimit,
  // The enode limit was hit.
  kNodeLimit,
  // The time limit was hit.
  kTimeLimit,
  // Some other reason to stop.
  kOther,
};

inline std::string ToString(StopReason reason) {
  switch (reason) {
    case StopReason::kSaturated:
      return "Saturated";
    case StopReason::kIterationLimit:
      return "IterationLimit";
    case StopReason::kNodeLimit:
      return "NodeLimit";
    case StopReason::kTimeLimit:
      return "TimeLimit";
    case StopReason::kOther:
      return "Other";
    default:
      return "Unknown!";
  }
}

struct Iteration {
  // The number of enodes in the egraph at the start of this iteration.
  size_t nodes = 0;
  // The number of eclasses in the egraph at the start of this iteration.
  size_t classes = 0;
  // The number of rebuild iterations done after this iteration completed.
  size_t rebuilds = 0;
  // A map from rule name to number of times it was newly applied in this iteration.
  std::unordered_map<std::string, size_t> applied;
  // Total time spent in this iteration, including data generation time.
  std::chrono::milliseconds total_time;
  // If the runner stopped on this iterations, this is the reason.
  std::optional<StopReason> stop_reason;
};

class RewriteScheduler {
 public:
  RewriteScheduler() = default;
  virtual ~RewriteScheduler() = default;

  // Whether or not the `Runner` is allowed to say it has saturated.
  // This is only called when the runner is otherwise saturated.
  virtual bool CanStop(size_t iteration) { return true; }

  // A hook allowing you to customize rewrite searching behavior.
  // Useful to implement rule management.
  virtual std::vector<SearchMatches> SearchRewrite(size_t iter_num, const Rewrite &rewrite, const EGraph &egraph) = 0;

  // A hook allowing you to customize rewrite application behavior.
  // Useful to implement rule management.
  virtual size_t ApplyRewrite(size_t iter_num, const Rewrite &rewrite, const std::vector<SearchMatches> &matches,
                              EGraph *egraph) = 0;
};

class SimpleScheduler : public RewriteScheduler {
 public:
  SimpleScheduler() = default;
  ~SimpleScheduler() override = default;

  std::vector<SearchMatches> SearchRewrite(size_t, const Rewrite &rewrite, const EGraph &egraph) override {
    return rewrite.Search(egraph);
  }

  size_t ApplyRewrite(size_t, const Rewrite &rewrite, const std::vector<SearchMatches> &matches,
                      EGraph *egraph) override {
    return rewrite.Apply(matches, egraph).size();
  }
};

// A `RewriteScheduler` that implements exponentional rule backoff.
// For each rewrite, there exists a configurable initial match limit.
// If a rewrite search yield more than this limit, then we ban this
// rule for number of iterations, double its limit, and double the time
// it will be banned next time.
// This seems effective at preventing explosive rules like
// associativity from taking an unfair amount of resources.
class BackoffScheduler : public RewriteScheduler {
 public:
  struct RuleState {
    size_t times_applied = 0;
    size_t banned_until = 0;
    size_t times_banned = 0;
    size_t match_limit = 0;
    size_t ban_length = 0;
    RuleState(size_t match_limit, size_t ban_length) : match_limit(match_limit), ban_length(ban_length) {}
    ~RuleState() = default;
  };

  BackoffScheduler() = default;
  ~BackoffScheduler() override = default;

  BackoffScheduler &SetInitMatchLimit(size_t limit) {
    init_match_limit_ = limit;
    return *this;
  }

  BackoffScheduler &SetInitBanLength(size_t length) {
    init_ban_length_ = length;
    return *this;
  }

  // Never ban a particular rule.
  BackoffScheduler &DoNotBan(const Rewrite &rule) {
    GetState(rule).match_limit = std::numeric_limits<size_t>::max();
    return *this;
  }

  /// Set the initial match limit for a rule.
  BackoffScheduler &SetMatchLimit(const Rewrite &rule, size_t limit) {
    GetState(rule).match_limit = limit;
    return *this;
  }

  // Set the initial ban length for a rule.
  BackoffScheduler &SetBanLength(const Rewrite &rule, size_t length) {
    GetState(rule).ban_length = length;
    return *this;
  }

  bool CanStop(size_t iteration) override {
    size_t min_ban = std::numeric_limits<size_t>::max();
    std::vector<RuleState *> banned;
    for (auto &s : states_) {
      if (s.second.banned_until > iteration) {
        if (s.second.banned_until < min_ban) {
          min_ban = s.second.banned_until;
        }
        banned.push_back(&s.second);
      }
    }
    if (banned.empty()) {
      return true;
    }
    auto delta = min_ban - iteration;
    for (auto banned_state : banned) {
      banned_state->banned_until -= delta;
    }
    return false;
  }

  std::vector<SearchMatches> SearchRewrite(size_t iteration, const Rewrite &rewrite, const EGraph &egraph) override {
    auto &state = GetState(rewrite);
    if (state.banned_until > iteration) {
      // Rewrite rule is banned.
      return {};
    }
    auto matches = rewrite.Search(egraph);
    auto total_len = std::accumulate(matches.begin(), matches.end(), size_t(0),
                                     [](size_t acc, const SearchMatches &m) { return acc + m.substs.size(); });
    auto threshold = (state.match_limit << state.times_banned);
    if (total_len > threshold) {
      // Ban rewrite if the match threshold is reached.
      auto ban_length = (state.ban_length << state.times_banned);
      ++state.times_banned;
      state.banned_until = iteration + ban_length;
      return {};
    }
    ++state.times_applied;
    return matches;
  }

  size_t ApplyRewrite(size_t, const Rewrite &rewrite, const std::vector<SearchMatches> &matches,
                      EGraph *egraph) override {
    return rewrite.Apply(matches, egraph).size();
  }

 private:
  RuleState &GetState(const Rewrite &rw) {
    auto found = states_.find(rw.id());
    if (found != states_.end()) {
      return found->second;
    }
    auto [iter, ok] = states_.emplace(rw.id(), RuleState(init_match_limit_, init_ban_length_));
    assert(ok);
    return iter->second;
  }

  size_t init_match_limit_ = 1000;
  size_t init_ban_length_ = 5;
  std::unordered_map<Rewrite::Id, RuleState> states_;
};

// Runner is the equality saturation engine that has reasonable defaults
// and implements many useful things like saturation checking, egraph
// size limits, and customizable rule scheduling (RewriteScheduler).
// Consider using Runner before rolling your own outer loop.
class Runner {
 public:
  using HookFunc = std::function<void(EGraph *, const std::vector<Iteration> &)>;

  Runner(EGraph *egraph) : Runner(egraph, std::make_unique<BackoffScheduler>()) {}
  Runner(EGraph *egraph, std::unique_ptr<RewriteScheduler> scheduler)
      : egraph_(egraph), scheduler_(std::move(scheduler)) {}
  ~Runner() = default;

  void SetIterLimit(size_t limit) { iter_limit_ = limit; }

  void SetNodeLimit(size_t limit) { node_limit_ = limit; }

  template <class Rep, class Period>
  void SetTimeLimit(const std::chrono::duration<Rep, Period> &limit) {
    time_limit_ = std::chrono::duration_cast<std::chrono::milliseconds>(limit);
  }

  void AddHook(HookFunc func) { hooks_.push_back(func); }

  void Run(const std::vector<Rewrite> &rules) {
    start_time_ = std::chrono::steady_clock::now();
    egraph_->Rebuild();
    while (!stop_reason_.has_value()) {
      auto iter = RunIteration(rules);
      stop_reason_ = iter.stop_reason;
      iterations_.emplace_back(std::move(iter));
    }
  }

  Iteration RunIteration(const std::vector<Rewrite> &rules) {
    const size_t iter_num = iterations_.size();
    Iteration iteration;
    // EGraph size before iteration.
    iteration.nodes = egraph_->size();
    iteration.classes = egraph_->eclasses().size();
    // Call hooks.
    for (auto &hook_func : hooks_) {
      hook_func(egraph_, iterations_);
    }
    // Search.
    std::vector<std::vector<SearchMatches>> matches;
    if (!iteration.stop_reason.has_value()) {
      for (auto &rule : rules) {
        auto ms = scheduler_->SearchRewrite(iter_num, rule, *egraph_);
        matches.emplace_back(std::move(ms));
        iteration.stop_reason = CheckLimits();
        if (iteration.stop_reason.has_value()) {
          break;
        }
      }
    }
    // Apply.
    if (!iteration.stop_reason.has_value()) {
      for (size_t i = 0; i < rules.size(); ++i) {
        auto &rule = rules.at(i);
        auto &ms = matches.at(i);
        auto applied = scheduler_->ApplyRewrite(iter_num, rule, ms, egraph_);
        if (applied > 0) {
          iteration.applied[rule.name()] += applied;
        }
        iteration.stop_reason = CheckLimits();
        if (iteration.stop_reason.has_value()) {
          break;
        }
      }
    }
    // Rebuild.
    iteration.rebuilds += egraph_->Rebuild();
    // Check saturation.
    if (!iteration.stop_reason.has_value()) {
      bool saturated = iteration.applied.empty() &&                      // no rewrite applies
                       scheduler_->CanStop(iter_num) &&                  // scheduler allows stop
                       egraph_->size() == iteration.nodes &&             // number of nodes not changed
                       egraph_->eclasses().size() == iteration.classes;  // number of classes not changed
      if (saturated) {
        iteration.stop_reason = StopReason::kSaturated;
      }
    }
    return iteration;
  }

  std::optional<StopReason> CheckLimits() {
    if (time_limit_.has_value()) {
      auto elapsed = std::chrono::steady_clock::now() - start_time_;
      if (elapsed > time_limit_.value()) {
        return StopReason::kTimeLimit;
      }
    }
    if (node_limit_.has_value() && egraph_->size() > node_limit_.value()) {
      return StopReason::kNodeLimit;
    }
    if (iter_limit_.has_value() && iterations_.size() > iter_limit_.value()) {
      return StopReason::kIterationLimit;
    }
    return std::nullopt;
  }

  void DebugPrint() const {
    std::cout << "=== EGraph Runner ===\n";
    if (!stop_reason_.has_value()) {
      std::cout << "Runner not finished.";
      return;
    }
    std::cout << "Iterations: " << iterations_.size() << " stop_reason: " << ToString(stop_reason_.value()) << '\n';
    for (size_t i = 0; i < iterations_.size(); ++i) {
      auto &iter = iterations_.at(i);
      std::cout << "[" << i << "] enodes:" << iter.nodes << " eclasses:" << iter.classes
                << " rebuilds: " << iter.rebuilds << '\n';
      for (auto &apply : iter.applied) {
        std::cout << "  " << apply.first << ": " << apply.second << '\n';
      }
    }
  }

 protected:
  // The `EGraph` used.
  EGraph *egraph_;

  // Data accumulated over each `Iteration`.
  std::vector<Iteration> iterations_;

  // The roots of expressions added, in insertion order.
  std::vector<EClassId> roots;

  // Why the `Runner` stopped. This will not be set if it hasn't stopped yet.
  std::optional<StopReason> stop_reason_;

  // The hook functions added, in insertion order.
  std::vector<HookFunc> hooks_;

  // limits
  std::optional<size_t> iter_limit_;
  std::optional<size_t> node_limit_;
  std::optional<std::chrono::milliseconds> time_limit_;

  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  std::unique_ptr<RewriteScheduler> scheduler_;
};

}  // namespace mindspore::egraph

#endif  // MINDSPORE_EGRAPH_RUNNER_H_