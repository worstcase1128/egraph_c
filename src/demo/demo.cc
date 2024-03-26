#include <cctype>
#include <cassert>
#include <iostream>
#include <fstream>
#include <thread>

#include "egraph/egraph.h"
#include "egraph/recexpr.h"
#include "egraph/pattern.h"
#include "egraph/matcher.h"
#include "egraph/rewrite.h"
#include "egraph/runner.h"
#include "egraph/extractor.h"

namespace eg = mindspore::egraph;

class MyNode : public eg::ENode {
 public:
  MyNode(const std::string &op) : op(op) {}

  MyNode(const std::string &op, const std::vector<eg::EClassId> &children) : op(op) {
    children_.insert(children_.end(), children.begin(), children.end());
  }

  size_t HashCode() const override {
    auto hashcode = std::hash<std::string>{}(op);
    return eg::HashCombine(hashcode, eg::ENode::HashCode());
  }

  eg::ENodePtr Clone() const override { return std::make_shared<MyNode>(op, children_); }

  std::string ToString() const override { return op; }

  std::string GetOpName() const override { return op; }

  bool Matches(const ENode &other) const override {
    auto node = dynamic_cast<const MyNode *>(&other);
    return node != nullptr && node->op == op;
  }

  std::string op;
};

class ConstantFolding : public eg::Analysis {
 public:
  ConstantFolding() = default;
  ~ConstantFolding() override = default;

  std::any MakeData(const eg::EGraph &egraph, const eg::ENodePtr &enode) override {
    auto mynode = std::dynamic_pointer_cast<MyNode>(enode);
    const bool is_const = std::isdigit(mynode->op[0]);
    if (is_const) {
      return std::stoi(mynode->op);
    }
    if (mynode->op == "*") {
      int value = 1;
      for (auto &c : mynode->children()) {
        auto &data = egraph.GetEClass(c)->data();
        if (!data.has_value()) {
          return {};
        }
        value *= std::any_cast<int>(data);
      }
      return value;
    }
    if (mynode->op == "+") {
      int value = 0;
      for (auto &c : mynode->children()) {
        auto &data = egraph.GetEClass(c)->data();
        if (!data.has_value()) {
          return {};
        }
        value += std::any_cast<int>(data);
      }
      return value;
    }
    return {};
  };

  eg::Analysis::Ordering MergeData(std::any *a, const std::any &b) override {
    if (!a->has_value() && !b.has_value()) {
      return eg::Analysis::Ordering::kEqual;
    }
    if (a->has_value() && !b.has_value()) {
      return eg::Analysis::Ordering::kGreater;
    }
    if (!a->has_value() && b.has_value()) {
      *a = b;
      return eg::Analysis::Ordering::kLess;
    }
    assert(std::any_cast<int>(*a) == std::any_cast<int>(b));
    return eg::Analysis::Ordering::kEqual;
  }

  void Modify(eg::EClassId id, eg::EGraph *egraph) override {
    auto eclass = egraph->GetEClass(id);
    assert(eclass != nullptr);
    if (eclass->data().has_value()) {
      int i = std::any_cast<int>(eclass->data());
      auto added = egraph->Add(std::make_shared<MyNode>(std::to_string(i)));
      egraph->Merge(id, added);
    }
  }
};

class MyCostModel : public eg::CostModel {
 public:
  eg::Cost GetCost(const eg::ENodePtr &enode, ClassCostFunc cost_func) override {
    eg::Cost cost = 1;
    auto op = enode->GetOpName();
    if (op == "*") {
      cost = 100;
    } else if (op == "<<") {
      cost = 50;
    }
    for (auto &child : enode->children()) {
      cost += cost_func(child);
    }
    return cost;
  }
};

std::vector<std::string> GetTokens(const std::string &s) {
  std::vector<std::string> tokens;
  std::stringstream input(s);
  std::string token;
  while (std::getline(input, token, ' ')) {
    if (!token.empty()) {
      tokens.emplace_back(std::move(token));
    }
  }
  return tokens;
}

eg::RecExpr::Id Parse(std::vector<std::string>::iterator *iter_ptr, eg::RecExpr *expr) {
  auto &iter = *iter_ptr;
  if (*iter == "(") {
    iter++;
    auto op = *iter++;
    std::vector<eg::RecExpr::Id> children;
    while (*iter != ")") {
      children.emplace_back(Parse(&iter, expr));
    }
    iter++;
    return expr->Add(std::make_shared<MyNode>(op, children));
  } else {
    auto name = *iter++;
    if (name[0] == '$') {
      return expr->Add(std::make_shared<eg::Var>(name.substr(1)));
    }
    return expr->Add(std::make_shared<MyNode>(name));
  }
}

eg::RecExpr Parse(const std::string &str) {
  eg::RecExpr expr;
  auto tokens = GetTokens(str);
  auto iter = tokens.begin();
  Parse(&iter, &expr);
  return expr;
}

void WriteDot(const eg::EGraph &g, const std::string &filename) {
  std::fstream out;
  out.open(filename, std::ios::out);
  if (!out.is_open()) {
    std::cout << "Open file " << filename << " failed!\n";
    return;
  }
  g.WriteDot(out);
  out.close();
}

void TestUnionFind() {
  eg::UnionFind<eg::EClassId> uf;
  auto a = uf.MakeSet();
  auto b = uf.MakeSet();
  auto c = uf.MakeSet();
  auto e = uf.MakeSet();
  auto f = uf.MakeSet();
  assert(uf.Find(a) == a);
  uf.Union(a, b);
  assert(uf.Find(b) == a);
  uf.Union(e, f);
  uf.Union(f, c);
  assert(uf.Find(c) == e);
  std::cout << "TestUnionFind done.\n";
}

eg::RecExpr TestRecExpr() {
  eg::RecExpr e;
  auto a = e.Add(std::make_shared<MyNode>("a"));
  auto two = e.Add(std::make_shared<MyNode>("2"));
  auto mul = e.Add(std::make_shared<MyNode>("*", std::vector{a, two}));
  auto two2 = e.Add(std::make_shared<MyNode>("2"));
  e.Add(std::make_shared<MyNode>("/", std::vector{mul, two2}));
  std::cout << "RecExpr: " << e.ToString() << std::endl;
  return e;
}

void TestEGraph() {
  eg::EGraph g;
  auto a = g.Add(std::make_shared<MyNode>("a"));
  auto two = g.Add(std::make_shared<MyNode>("2"));
  auto mul = g.Add(std::make_shared<MyNode>("*", std::vector{a, two}));
  g.Add(std::make_shared<MyNode>("/", std::vector{mul, two}));
  WriteDot(g, "demo_init.dot");

  auto one = g.Add(std::make_shared<MyNode>("1"));
  auto lshift = g.Add(std::make_shared<MyNode>("<<", std::vector{a, one}));
  WriteDot(g, "demo_add.dot");
  g.Merge(mul, lshift);
  WriteDot(g, "demo_merge.dot");
  g.Rebuild();
  WriteDot(g, "demo_rebuild.dot");
  g.DebugPrint();
}

void TestAddExpr() {
  auto e = TestRecExpr();
  eg::EGraph g;
  g.AddExpr(e);
  WriteDot(g, "demo_addexpr.dot");
  g.DebugPrint();
}

void TestPatternVar() {
  eg::RecExpr e;
  eg::ENodePtr var = std::make_shared<eg::Var>("a");
  assert(var->IsA<eg::Var>());
  auto a = e.Add(var);
  auto two = e.Add(std::make_shared<MyNode>("2"));
  auto mul = e.Add(std::make_shared<MyNode>("*", std::vector{a, two}));
  auto two2 = e.Add(std::make_shared<MyNode>("2"));
  e.Add(std::make_shared<MyNode>("/", std::vector{mul, two2}));
  std::cout << "Pattern: " << e.ToString() << std::endl;
}

void TestLookupExpr() {
  eg::RecExpr e;
  auto a = e.Add(std::make_shared<MyNode>("a"));
  auto two = e.Add(std::make_shared<MyNode>("2"));
  e.Add(std::make_shared<MyNode>("*", std::vector{a, two}));
  eg::EGraph g;
  a = g.Add(std::make_shared<MyNode>("a"));
  two = g.Add(std::make_shared<MyNode>("2"));
  auto mul = g.Add(std::make_shared<MyNode>("*", std::vector{a, two}));
  g.Add(std::make_shared<MyNode>("/", std::vector{mul, two}));
  auto id = g.LookupExpr(e);
  assert(id.has_value() && id.value() == g.Find(mul));
  std::cout << "TestLookupExpr: " << e.ToString() << " ok" << std::endl;
}

void TestPatternCompile() {
  eg::RecExpr e;
  eg::ENodePtr var = std::make_shared<eg::Var>("a");
  assert(var->IsA<eg::Var>());
  auto a = e.Add(var);
  auto two = e.Add(std::make_shared<MyNode>("2"));
  auto mul = e.Add(std::make_shared<MyNode>("*", std::vector{a, two}));
  auto three = e.Add(std::make_shared<MyNode>("3"));
  auto add = e.Add(std::make_shared<MyNode>("+", std::vector{three, two}));
  auto root = e.Add(std::make_shared<MyNode>("/", std::vector{mul, add}));
  std::cout << "Pattern: " << e.ToString() << " last id: " << root << std::endl;
  e.DebugPrint();
  eg::MatcherCompiler compiler(e);
  eg::Matcher matcher = compiler.compile();
  matcher.DebugPrint();
}

void TestParseExpr() {
  auto e = Parse("( * a ( + 1 2 ) )");
  std::cout << e.ToString() << std::endl;
  e = Parse("( / ( * a 2 ) 2 )");
  std::cout << e.ToString() << std::endl;
  e = Parse("( / ( * a 2 ) $x )");
  std::cout << e.ToString() << std::endl;
  e.DebugPrint();
}

void TestPatternSearch() {
  eg::Pattern pattern(Parse("( / $x $x )"));
  eg::EGraph g;
  g.AddExpr(Parse("( * ( / a a ) 1 )"));
  g.Rebuild();
  g.DebugPrint();

  auto matches = pattern.Search(g);
  assert(matches.size() == 1);
  std::cout << "TestPatternSearch matches: " << matches.size() << std::endl;
  for (auto &m : matches) {
    m.DebugPrint();
  }
}

void TestRewriteSearchApply() {
  eg::Rewrite rw("mul2shift", Parse("( * $x 2 )"), Parse("( << $x 1 )"));
  std::cout << rw.ToString() << std::endl;

  eg::EGraph g;
  g.AddExpr(Parse("( / ( * a 2 ) 2 )"));
  g.Rebuild();
  g.DebugPrint();

  auto matches = rw.Search(g);
  std::cout << "matches: " << matches.size() << std::endl;
  for (auto &m : matches) {
    m.DebugPrint();
  }

  rw.Apply(matches, &g);
  g.Rebuild();
  WriteDot(g, "pattern_apply.dot");
}

void TestApplyRewrites() {
  std::vector<eg::Rewrite> rewrites;
  rewrites.emplace_back("mul2shift", Parse("( * $x 2 )"), Parse("( << $x 1 )"));
  rewrites.emplace_back("mul_comm", Parse("( / ( * $x $y ) $z )"), Parse("( * $x ( / $y $z ) )"));
  rewrites.emplace_back("div2one", Parse("( / $x $x )"), Parse("1"));
  rewrites.emplace_back("mul_by_one", Parse("( * $x 1 )"), Parse("$x"));

  for (auto &rw : rewrites) {
    std::cout << rw.ToString() << std::endl;
  }

  eg::EGraph g;
  g.AddExpr(Parse("( / ( * a 2 ) 2 )"));
  g.Rebuild();
  g.DebugPrint();

  for (int i = 0; i < 4; ++i) {
    std::vector<std::pair<eg::Rewrite &, std::vector<eg::SearchMatches>>> todo;
    for (auto &rw : rewrites) {
      auto matches = rw.Search(g);
      if (!matches.empty()) {
        todo.emplace_back(rw, std::move(matches));
      }
    }
    for (auto &t : todo) {
      auto &rw = t.first;
      auto &matches = t.second;
      auto applied = rw.Apply(matches, &g);
      std::cout << i << ": " << rw.name() << " applied: " << applied.size() << std::endl;
    }
    g.Rebuild();
  }
  WriteDot(g, "apply_rewrites.dot");
}

void TestSimpleAlgebra() {
  eg::EGraph g(std::make_unique<ConstantFolding>());
  g.AddExpr(Parse("( + ( pow2 ( + a b ) ) c )"));  // pow2(a + b) + c
  g.Rebuild();

  eg::Rewrite rw("Algebra", Parse("( pow2 ( + $x $y ) )"), Parse("( + ( pow2 $x ) ( * 2 $x $y ) ( pow2 $y ) )"));
  auto matches = rw.Search(g);
  rw.Apply(matches, &g);
  g.Rebuild();
  WriteDot(g, "algebra.dot");
}

void TestConstFolding() {
  eg::EGraph g(std::make_unique<ConstantFolding>());
  g.AddExpr(Parse("( + ( * a 0 ) ( + 1 2 ) )"));
  g.Rebuild();
  WriteDot(g, "const_folding1.dot");

  eg::Rewrite rw("mul0", Parse("( * $x 0 )"), Parse("0"));
  auto matches = rw.Search(g);
  rw.Apply(matches, &g);
  g.Rebuild();
  WriteDot(g, "const_folding2.dot");
}

void TestConditonRewrite() {
  eg::EGraph g(std::make_unique<ConstantFolding>());
  g.AddExpr(Parse("( + ( + a 1 ) ( + 1 2 ) )"));
  g.Rebuild();

  eg::Rewrite rw("add-commu", Parse("( + $x $y )"), Parse("( + $y $x )"),
                 [](const eg::Substitution &subst, const eg::EGraph &g, eg::EClassId) {
                   // Not to apply if both 'x' and 'y' are constants.
                   auto x = subst.var("x").value();
                   auto y = subst.var("y").value();
                   return !(g[x].data().has_value() && g[y].data().has_value());
                 });
  auto matches = rw.Search(g);
  rw.Apply(matches, &g);
  g.Rebuild();
  WriteDot(g, "cond_rewrite.dot");
}

void TestDynamicRewrite() {
  eg::EGraph g(std::make_unique<ConstantFolding>());
  g.AddExpr(Parse("( + ( + a 1 ) ( + 1 2 ) )"));
  g.Rebuild();

  eg::Rewrite rw("add-commu", Parse("( + $x $y )"), [](const eg::Substitution &subst, eg::EGraph *g, eg::EClassId) {
    auto x = subst.var("x").value();
    auto y = subst.var("y").value();
    if ((*g)[x].data().has_value() && (*g)[y].data().has_value()) {
      // Not to apply if both 'x' and 'y' are constants.
      return eg::IdVec{};
    }
    auto id = g->Add(std::make_shared<MyNode>("+", eg::IdVec{y, x}));
    return eg::IdVec{id};
  });
  auto matches = rw.Search(g);
  rw.Apply(matches, &g);
  g.Rebuild();
  WriteDot(g, "dynamic_rewrite.dot");
}

void TestRunner() {
  std::vector<eg::Rewrite> rewrites;
  rewrites.emplace_back("mul2shift", Parse("( * $x 2 )"), Parse("( << $x 1 )"));
  rewrites.emplace_back("mul_comm", Parse("( / ( * $x $y ) $z )"), Parse("( * $x ( / $y $z ) )"));
  rewrites.emplace_back("div2one", Parse("( / $x $x )"), Parse("1"));
  rewrites.emplace_back("mul_by_one", Parse("( * $x 1 )"), Parse("$x"));

  eg::EGraph g;
  g.AddExpr(Parse("( / ( * a 2 ) 2 )"));

  eg::Runner runner(&g);
  runner.AddHook([](eg::EGraph *egraph, const std::vector<eg::Iteration> &iterations) {
    std::string filename = "iter" + std::to_string(iterations.size()) + ".dot";
    WriteDot(*egraph, filename);
  });
  runner.Run(rewrites);
  runner.DebugPrint();
  WriteDot(g, "runner.dot");
}

void TestRunnerTimeout() {
  std::vector<eg::Rewrite> rewrites;
  rewrites.emplace_back("mul2shift", Parse("( * $x 2 )"), Parse("( << $x 1 )"));
  rewrites.emplace_back("mul_comm", Parse("( / ( * $x $y ) $z )"), Parse("( * $x ( / $y $z ) )"));
  rewrites.emplace_back("div2one", Parse("( / $x $x )"), Parse("1"));
  rewrites.emplace_back("mul_by_one", Parse("( * $x 1 )"), Parse("$x"));

  eg::EGraph g;
  g.AddExpr(Parse("( / ( * a 2 ) 2 )"));

  eg::Runner runner(&g);
  runner.AddHook([](eg::EGraph *, const std::vector<eg::Iteration> &iters) {
    std::cout << "sleep in iteration " << iters.size() << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  });
  runner.SetTimeLimit(std::chrono::milliseconds(25));
  runner.Run(rewrites);
  runner.DebugPrint();
  WriteDot(g, "runner_timeout.dot");
}

void TestBackoffScheduler() {
  std::vector<eg::Rewrite> rewrites;
  rewrites.emplace_back("add_comm", Parse("( + $x $y )"), Parse("( + $y $x )"));
  rewrites.emplace_back("add_zero", Parse("( + $x 0 )"), Parse("$x"));

  eg::EGraph g;
  g.AddExpr(Parse("( + ( + ( + a 0 ) ( + b c ) ) ( + ( + d 0 ) ( + e f ) ) )"));

  auto scheduler = std::make_unique<eg::BackoffScheduler>();
  scheduler->SetInitMatchLimit(4).DoNotBan(rewrites[1]).SetBanLength(rewrites[0], 2);
  eg::Runner runner(&g, std::move(scheduler));
  runner.Run(rewrites);
  runner.DebugPrint();
  WriteDot(g, "backkoff_scheduler.dot");
}

void TestRewriteCheckVar() {
  eg::Rewrite rw1("test1", Parse("( + $x $y )"), Parse("( + $x $y )"));
  eg::Rewrite rw2("test1", Parse("( + $x $y )"), Parse("( + $x $z )"));
  assert(rw1.CheckVariables() == true);
  assert(rw2.CheckVariables() == false);
}

void TestExtractor() {
  std::vector<eg::Rewrite> rewrites;
  rewrites.emplace_back("mul2shift", Parse("( * $x 2 )"), Parse("( << $x 1 )"));
  rewrites.emplace_back("mul_comm", Parse("( / ( * $x $y ) $z )"), Parse("( * $x ( / $y $z ) )"));
  rewrites.emplace_back("div2one", Parse("( / $x $x )"), Parse("1"));
  rewrites.emplace_back("mul_by_one", Parse("( * $x 1 )"), Parse("$x"));

  eg::EGraph g(std::make_unique<ConstantFolding>());
  auto id = g.AddExpr(Parse("( * ( * a 2 ) ( + 1 1 ) )"));

  eg::Runner runner(&g);
  runner.Run(rewrites);
  WriteDot(g, "extractor.dot");

  eg::Extractor extractor(g, std::make_unique<MyCostModel>());
  auto [best_expr, cost] = extractor.FindBest(id);
  std::cout << "Best expr: " << best_expr.ToString() << " cost:" << cost << std::endl;
}

int main() {
  TestUnionFind();
  TestEGraph();
  TestAddExpr();
  TestPatternVar();
  TestLookupExpr();
  TestPatternCompile();
  TestParseExpr();
  TestPatternSearch();
  TestRewriteSearchApply();
  TestApplyRewrites();
  TestSimpleAlgebra();
  TestConstFolding();
  TestConditonRewrite();
  TestDynamicRewrite();
  TestRunner();
  TestRunnerTimeout();
  TestBackoffScheduler();
  TestRewriteCheckVar();
  TestExtractor();
}
