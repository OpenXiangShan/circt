//===- HWRepCut.cpp - RepCut algorithm implementation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass implements the RepCut algorithm for partitioning hardware designs
// with replication-aided graph partitioning. The algorithm creates a hypergraph
// from the design, partitions it using KaHyPar, and assigns partition IDs to
// each operation as attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <vector>

#define DEBUG_TYPE "hw-repcut"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWREPCUT
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;
using namespace mlir;

namespace {

// Node ID type
using NodeID = size_t;
#define UNVISITED_NODE (SIZE_MAX)
#define INVALID_NODE (SIZE_MAX - 1)

// Weighted hypergraph representation
struct HyperGraph {
  std::vector<std::vector<NodeID>>
      nodes; // nodes[i] contains edge IDs connected to node i
  std::vector<std::vector<NodeID>>
      edges; // edges[i] contains node IDs in edge i
  std::vector<size_t> nodeWeight;
  std::vector<size_t> edgeWeight;

  void addNode(NodeID id, size_t weight) {
    if (static_cast<size_t>(id) >= nodes.size()) {
      nodes.resize(id + 1);
      nodeWeight.resize(id + 1, SIZE_MAX);
    }
    nodeWeight[id] = weight;
  }

  void addEdge(const std::vector<NodeID> &edge, size_t weight) {
    if (edge.empty())
      return;

    NodeID edgeId = edges.size();
    edges.push_back(edge);
    edgeWeight.push_back(weight);

    for (NodeID nodeId : edge) {
      if (nodeId >= nodes.size()) {
        // should not reach here
        nodes.resize(nodeId + 1);
        nodeWeight.resize(nodeId + 1, SIZE_MAX);
      }
      nodes[nodeId].push_back(edgeId);
    }
  }
};

// Statement graph for dependency tracking
class StatementGraph {
public:
  std::vector<std::vector<NodeID>> outNeigh; // outgoing neighbors
  std::vector<std::vector<NodeID>> inNeigh;  // incoming neighbors
  std::map<Operation *, NodeID> opToID;
  std::vector<Operation *> idToOp;
  std::set<NodeID> validNodes;
  std::set<NodeID> sinkNodes; // nodes with no outgoing edges

  // Build dependency graph from MLIR operations
  void buildStatementGraph(hw::HWModuleOp moduleOp) {
    // Create edges based on operand dependencies
    moduleOp.walk([&](Operation *op) {
      if (isIgnoreOp(op))
        return;

      addNode(op);

      for (Value operand : op->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (defOp && !isSinkOp(defOp)) {
          if (isa<seq::FirMemOp>(defOp) &&
              isa<seq::FirMemReadOp, seq::FirMemWriteOp,
                  seq::FirMemReadWriteOp>(op)) {
            addEdge(op, defOp); // reverse connect
          } else {
            addEdge(defOp, op); // normal connect
          }
        }
      }
    });
  }

  NodeID addNode(Operation *op) {
    if (opToID.find(op) != opToID.end())
      return opToID[op];

    NodeID id = idToOp.size();
    opToID[op] = id;
    idToOp.push_back(op);

    // Resize neighbor lists if needed
    if (id >= outNeigh.size()) {
      outNeigh.resize(id + 1);
      inNeigh.resize(id + 1);
    }

    validNodes.insert(id);
    llvm::dbgs() << "Op: " << id << " : " << op->getName().getStringRef()
                 << "\n";
    return id;
  }

  void addEdge(Operation *from, Operation *to) {
    addNode(from);
    addNode(to);

    NodeID fromId = opToID[from];
    NodeID toId = opToID[to];

    llvm::dbgs() << "Edge: " << fromId << " -> " << toId << "\n";

    outNeigh[fromId].push_back(toId);
    inNeigh[toId].push_back(fromId);
  }

  // TODO: memory, output, etc.
  bool isSinkOp(Operation *op) const { return llvm::isa<seq::FirRegOp>(op); }

  bool isIgnoreOp(Operation *op) const {
    return llvm::isa<hw::OutputOp>(op) || llvm::isa<hw::HWModuleOp>(op);
  }

  bool isSinkID(NodeID id) const {
    Operation *op = idToOp[id];
    return isSinkOp(op);
  }

  bool isSinkNode(NodeID id) const {
    return outNeigh[id].empty() || isSinkID(id);
  }

  void identifySinkNodes() {
    sinkNodes.clear();
    for (NodeID id : validNodes) {
      if (isSinkNode(id)) {
        sinkNodes.insert(id);
      }
    }
  }

  size_t numNodes() const { return idToOp.size(); }
};

// Partition graph extending statement graph with RepCut-specific data
class PartGraph : public StatementGraph {
public:
  std::vector<std::set<NodeID>> idToTreeID; // which trees each node belongs to
  std::vector<NodeID> idToPieceID;          // which piece each node belongs to
  std::vector<size_t> idToNodeWeight;       // simulation weight of each node
  std::vector<std::set<NodeID>> trees;      // tree sets (cones)
  std::vector<std::set<NodeID>> pieces;     // piece sets (clusters)
  HyperGraph hg;

  void initializeFromGraph(StatementGraph &sg) {
    // Copy data from statement graph
    outNeigh = sg.outNeigh;
    inNeigh = sg.inNeigh;
    opToID = sg.opToID;
    idToOp = sg.idToOp;
    validNodes = sg.validNodes;
    sinkNodes = sg.sinkNodes;

    // Initialize RepCut-specific data
    size_t numNodes = this->numNodes();
    idToTreeID.resize(numNodes);
    idToPieceID.resize(numNodes, UNVISITED_NODE);
    idToNodeWeight.resize(numNodes, SIZE_MAX);

    // Mark invalid nodes
    for (size_t i = 0; i < numNodes; ++i) {
      if (validNodes.find(i) == validNodes.end()) {
        idToPieceID[i] = INVALID_NODE; // invalid
      }
    }
  }

  // Build trees (cones) starting from sink nodes
  void initTrees() {
    std::map<NodeID, std::set<NodeID>> treeCache;

    std::function<std::set<NodeID>(NodeID)> collectTree =
        [&](NodeID seed) -> std::set<NodeID> {
      if (treeCache.find(seed) != treeCache.end()) {
        return treeCache[seed];
      }

      std::set<NodeID> depNodes;

      if (validNodes.find(seed) == validNodes.end()) {
        // Invalid node
        treeCache[seed] = depNodes;
        return depNodes;
      }

      // Add current node and recursively collect dependencies
      depNodes.insert(seed);
      for (NodeID pred : inNeigh[seed]) {
        if (!isSinkID(pred)) { // Stop at sink operations
          std::set<NodeID> predDeps = collectTree(pred);
          depNodes.insert(predDeps.begin(), predDeps.end());
        }
      }

      treeCache[seed] = depNodes;
      return depNodes;
    };

    // Build trees for each sink node
    trees.clear();
    for (NodeID sinkId : sinkNodes) {
      std::set<NodeID> tree = collectTree(sinkId);
      trees.push_back(tree);

      // Mark which tree each node belongs to
      for (NodeID nodeId : tree) {
        idToTreeID[nodeId].insert(trees.size() - 1);
      }
    }
  }

  // Group nodes into pieces based on same tree membership
  void initPieces() {
    pieces.clear();
    idToPieceID.clear();
    idToPieceID.resize(numNodes(), static_cast<NodeID>(UNVISITED_NODE));

    std::function<void(NodeID, NodeID)> assignPiece = [&](NodeID nodeId,
                                                          NodeID pieceId) {
      if (idToPieceID[nodeId] != UNVISITED_NODE)
        return; // already assigned

      idToPieceID[nodeId] = pieceId;
      pieces[pieceId].insert(nodeId);

      // Recursively assign neighbors in the same tree signature
      auto neighbors = inNeigh[nodeId];
      neighbors.insert(neighbors.end(), outNeigh[nodeId].begin(),
                       outNeigh[nodeId].end());
      for (NodeID neighbor : neighbors) {
        if (idToPieceID[neighbor] == UNVISITED_NODE &&
            idToTreeID[neighbor] == idToTreeID[nodeId]) {
          assignPiece(neighbor, pieceId);
        }
      }
    };

    // First, collect all pieces for all sink nodes
    for (NodeID sinkId : sinkNodes) {
      pieces.emplace_back();
      NodeID pieceId = pieces.size() - 1;
      assignPiece(sinkId, pieceId);
    }

    // Collect pieces for all other nodes
    for (size_t i = 0; i < numNodes(); ++i) {
      if (idToPieceID[i] == UNVISITED_NODE) {
        pieces.emplace_back();
        NodeID pieceId = pieces.size() - 1;
        assignPiece(i, pieceId);
      }
    }
  }

  // Helper function to get bit width of a type
  size_t getTypeWidth(Type type) {
    if (auto intType = llvm::dyn_cast<IntegerType>(type))
      return intType.getWidth();
    return 1; // Default for other types
  }

  // Helper function to get max operand width
  size_t getMaxOperandWidth(Operation *op) {
    size_t maxWidth = 1;
    for (Value operand : op->getOperands()) {
      size_t width = getTypeWidth(operand.getType());
      if (width > maxWidth)
        maxWidth = width;
    }
    return maxWidth;
  }

  // Helper function to check if a value is a constant
  bool isConstant(Value value) {
    return isa_and_nonnull<hw::ConstantOp>(value.getDefiningOp());
  }

  // Helper function to check if all operands are constants
  bool hasAllConstantOperands(Operation *op) {
    return std::all_of(op->getOperands().begin(), op->getOperands().end(),
                       [this](Value operand) { return isConstant(operand); });
  }

  // Helper function to check if any operand is a constant
  bool hasAnyConstantOperand(Operation *op) {
    return std::any_of(op->getOperands().begin(), op->getOperands().end(),
                       [this](Value operand) { return isConstant(operand); });
  }

  // Calculate simulation weight for a node based on operation type and width
  size_t calculateNodeWeight(NodeID id) {
    if (idToNodeWeight[id] != SIZE_MAX)
      return idToNodeWeight[id];

    Operation *op = idToOp[id];
    size_t weight = 0;

    // Constant operations - weight is the bit width (like
    // UIntLiteral/SIntLiteral in Scala)
    if (isa<hw::ConstantOp>(op)) {
      Type resultType = op->getResult(0).getType();
      weight = getTypeWidth(resultType);
      idToNodeWeight[id] = weight;
      return weight;
    }

    // Register operations - no simulation cost for register definition
    if (isa<seq::FirRegOp>(op)) {
      // In MLIR, FirRegOp includes the next value
      // Cost is for assignment operation
      auto regOp = cast<seq::FirRegOp>(op);
      Type resultType = regOp.getResult().getType();
      size_t width = getTypeWidth(resultType);
      weight = (width <= 64) ? 2 : ((width + 63) / 64 + 1);
    }
    // Memory operations
    else if (isa<seq::FirMemOp>(op)) {
      weight = 0; // Memory definition has no cost
    } else if (isa<seq::FirMemReadOp, seq::FirMemWriteOp,
                   seq::FirMemReadWriteOp>(op)) {
      // Memory access cost: 1 base + operand costs
      weight = 1 + op->getNumOperands();
    }
    // Arithmetic operations
    else if (isa<comb::AddOp, comb::SubOp>(op)) {
      size_t maxWidth = getMaxOperandWidth(op);
      size_t outputWidth = maxWidth + 1;
      if (outputWidth <= 64)
        weight = 2;
      else if (outputWidth <= 128)
        weight = 8;
      else if (outputWidth <= 256)
        weight = 16;
      else
        weight = 30;
    } else if (auto mulOp = dyn_cast<comb::MulOp>(op)) {
      size_t maxWidth = getMaxOperandWidth(op);
      if (maxWidth <= 64) {
        // Find minimum operand width
        size_t minWidth = maxWidth;
        for (Value operand : mulOp->getOperands()) {
          size_t w = getTypeWidth(operand.getType());
          if (w < minWidth)
            minWidth = w;
        }
        if (minWidth <= 8)
          weight = 1;
        else if (minWidth <= 16)
          weight = 9;
        else
          weight = 25;
      } else {
        weight = 25;
      }
    } else if (isa<comb::DivUOp, comb::DivSOp, comb::ModUOp, comb::ModSOp>(
                   op)) {
      weight = 6; // Division/Remainder only supports < 64 bits
    }
    // Comparison operations
    else if (auto icmpOp = dyn_cast<comb::ICmpOp>(op)) {
      size_t maxWidth = getMaxOperandWidth(op);
      if (maxWidth <= 64)
        weight = 1;
      else if (maxWidth <= 128)
        weight = 3;
      else
        weight = 5;
    }
    // Shift operations
    else if (isa<comb::ShlOp, comb::ShrUOp, comb::ShrSOp>(op)) {
      size_t width = getTypeWidth(op->getResult(0).getType());
      // Check if shift amount (second operand) is constant
      bool isStaticShift =
          op->getNumOperands() >= 2 && isConstant(op->getOperand(1));

      if (isStaticShift) {
        // Static shift (Shl/Shr in FIRRTL)
        // However, this does not have effect because operation with constant
        // shift amount is weighted as 0 later
        // TODO: Should we fix this? Or just leave it as is?
        weight = 2;
      } else {
        // Dynamic shift - similar to Dshlw/Dshr in FIRRTL
        // Note: MLIR comb doesn't distinguish between Dshl/Dshlw/Dshr,
        // we treat all dynamic shifts uniformly
        if (width <= 64)
          weight = 5; // Average of Dshlw(6) and Dshr(5)
        else {
          size_t nWords = (width + 63) / 64;
          weight = nWords * 20;
        }
      }
    }
    // Bitwise operations
    else if (isa<comb::AndOp, comb::OrOp, comb::XorOp>(op)) {
      size_t maxWidth = getMaxOperandWidth(op);
      if (maxWidth <= 64)
        weight = 2;
      else {
        size_t nWords = (maxWidth + 63) / 64;
        weight = nWords * 2;
      }
    } else if (isa<comb::ParityOp>(op)) {
      size_t width = getTypeWidth(op->getResult(0).getType());
      size_t nWords = (width + 63) / 64;
      if (width == 1)
        weight = 5;
      else if (width <= 64)
        weight = 20;
      else if (width <= 192)
        weight = nWords * 20;
      else
        weight = nWords * 10;
    }
    // Concat, Extract, etc.
    else if (auto concatOp = dyn_cast<comb::ConcatOp>(op)) {
      size_t totalWidth = 0;
      for (Value operand : concatOp->getOperands())
        totalWidth += getTypeWidth(operand.getType());
      if (totalWidth <= 64)
        weight = 2;
      else {
        size_t nWords = (totalWidth + 63) / 64;
        weight = nWords * 5;
      }
    } else if (isa<comb::ExtractOp>(op)) {
      weight = 2; // Extract (Bits equivalent)
    } else if (isa<comb::ReplicateOp>(op)) {
      weight = 1;
    }
    // Mux operation
    else if (auto muxOp = dyn_cast<comb::MuxOp>(op)) {
      Type resultType = muxOp.getResult().getType();
      size_t width = getTypeWidth(resultType);
      size_t nWords = (width + 63) / 64;
      weight = nWords * 6;
    }
    // Default for other operations
    else {
      // For operations like wire assignments, etc.
      Type resultType =
          op->getNumResults() > 0 ? op->getResult(0).getType() : Type();
      if (resultType) {
        size_t width = getTypeWidth(resultType);
        weight = (width <= 64) ? 2 : ((width + 63) / 64 + 1);
      } else {
        weight = 1;
      }
    }

    // If any operand is a constant (like argLiterals.nonEmpty in Scala),
    // reduce weight to 0 (the constant weight is already counted separately)
    // This matches Scala's: if (argLiterals.nonEmpty) 0 else opWeight + ...
    if (op->getNumOperands() > 0 && hasAnyConstantOperand(op))
      weight = 0;

    idToNodeWeight[id] = weight;
    return weight;
  }

  // Calculate total weight of a piece (sum of all nodes in piece)
  size_t calculatePieceWeight(const std::set<NodeID> &piece) {
    // Find sink nodes in this piece
    // A node is a sink in the piece if its outgoing edges don't stay in the
    // piece
    std::vector<NodeID> pieceSinkNodes;
    for (NodeID nodeId : piece) {
      bool isSink = true;
      for (NodeID succ : outNeigh[nodeId]) {
        if (piece.find(succ) != piece.end()) {
          isSink = false;
          break;
        }
      }
      if (isSink)
        pieceSinkNodes.push_back(nodeId);
    }

    std::set<NodeID> visitedNodes;
    size_t totalWeight = 0;

    // Recursive function to calculate weight of a statement and its
    // dependencies
    std::function<size_t(NodeID)> stmtWeight = [&](NodeID sinkId) -> size_t {
      if (visitedNodes.find(sinkId) != visitedNodes.end())
        return 0;

      visitedNodes.insert(sinkId);

      // Ensure weight is calculated
      if (idToNodeWeight[sinkId] == SIZE_MAX)
        idToNodeWeight[sinkId] = calculateNodeWeight(sinkId);

      size_t currentWeight = idToNodeWeight[sinkId];

      // Add weights of predecessors that are valid and in this piece
      for (NodeID pred : inNeigh[sinkId]) {
        if (validNodes.find(pred) != validNodes.end() &&
            piece.find(pred) != piece.end()) {
          currentWeight += stmtWeight(pred);
        }
      }

      return currentWeight;
    };

    // Calculate weight starting from each sink node
    for (NodeID sinkId : pieceSinkNodes) {
      totalWeight += stmtWeight(sinkId);
    }

    // Add small cost for print/stop operations (approximation)
    // In MLIR, these might be sv::FWriteOp, sv::FinishOp, etc.
    size_t specialOpCount = 0;
    for (NodeID nodeId : piece) {
      Operation *op = idToOp[nodeId];
      if (op->getName().getStringRef().contains("finish") ||
          op->getName().getStringRef().contains("stop") ||
          op->getName().getStringRef().contains("fwrite") ||
          op->getName().getStringRef().contains("print")) {
        specialOpCount++;
      }
    }

    // Weight should be at least 1 to make KaHyPar happy
    // Add fractional cost for special operations
    return totalWeight + 1 + specialOpCount / 7;
  }

  // Build hypergraph for partitioning
  void updateHyperGraph() {

    // Calculate piece weights
    std::vector<size_t> pieceWeights;
    pieceWeights.reserve(pieces.size());
    for (const auto &piece : pieces) {
      pieceWeights.push_back(calculatePieceWeight(piece));
    }

    // Calculate pin counts (number of trees each piece connects to)
    std::vector<size_t> hePinCount;
    for (const auto &piece : pieces) {
      if (!piece.empty()) {
        hePinCount.push_back(idToTreeID[*piece.begin()].size());
      } else {
        hePinCount.push_back(1);
      }
    }

    // Add nodes
    for (size_t elem = 0; elem < trees.size(); ++elem) {
      size_t weight = pieceWeights[elem];
      std::vector<NodeID> connectPieces;
      for (NodeID id : trees[elem]) {
        auto pieceID = idToPieceID[id];
        if (pieceID != elem)
          connectPieces.push_back(pieceID);
      }
      hg.addNode(elem, weight);
    }

    // Add edges
    for (size_t elem = trees.size(); elem < pieces.size(); ++elem) {
      auto edgeWeight = pieceWeights[elem];
      auto edgeNodesSet = idToTreeID[*pieces[elem].begin()];
      std::vector<NodeID> edgeNodes(edgeNodesSet.begin(), edgeNodesSet.end());
      hg.addEdge(edgeNodes, edgeWeight);
    }
  }
};

struct HWRepCutPass : public circt::hw::impl::HWRepCutBase<HWRepCutPass> {
  using Base::Base;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Build statement graph from the module
    StatementGraph sg;
    sg.buildStatementGraph(moduleOp);

    // Create partition graph and run RepCut algorithm
    PartGraph pg;
    pg.initializeFromGraph(sg);

    // Run RepCut steps
    pg.identifySinkNodes();
    for (NodeID id : pg.sinkNodes)
      llvm::dbgs() << id << "\n";
    pg.initTrees();
    for (const auto &tree : pg.trees) {
      llvm::dbgs() << "Tree: ";
      for (auto node : tree)
        llvm::dbgs() << node << " ";
      llvm::dbgs() << "\n";
    }
    pg.initPieces();
    for (const auto &piece : pg.pieces) {
      llvm::dbgs() << "Piece: ";
      for (auto node : piece)
        llvm::dbgs() << node << " ";
      llvm::dbgs() << "\n";
    }
    pg.updateHyperGraph();
    llvm::dbgs() << "Hypergraph: " << pg.hg.nodes.size() << " nodes, "
                 << pg.hg.edges.size() << " edges\n";

    // Write hypergraph to file and call KaHyPar
    std::string tempDir = "/tmp";
    std::string hmetisFile = tempDir + "/repcut.hmetis";
    writeHypergraphFile(pg.hg, hmetisFile);

    // Run KaHyPar partitioner
    std::vector<NodeID> partitionAssignment;
    if (runKaHyPar(hmetisFile, numPartitions, partitionAssignment)) {
      // Apply partition assignments to operations
      applyPartitionAssignments(pg, partitionAssignment, moduleOp);
    } else {
      // Fallback: simple round-robin assignment
      llvm::errs() << "KaHyPar failed, using simple round-robin assignment\n";
      partitionAssignment.resize(pg.trees.size());
      for (size_t i = 0; i < pg.trees.size(); ++i) {
        partitionAssignment[i] = i % numPartitions;
      }
      applyPartitionAssignments(pg, partitionAssignment, moduleOp);
    }
  }

private:
  // Write hypergraph in hMETIS format
  void writeHypergraphFile(const HyperGraph &hg, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      llvm::errs() << "Failed to open file: " << filename << "\n";
      return;
    }

    // Header: number of hyperedges, number of vertices, format
    file << hg.edges.size() << " " << hg.nodes.size() << " 11\n";

    // Write hyperedges with weights
    for (size_t i = 0; i < hg.edges.size(); ++i) {
      file << hg.edgeWeight[i];
      for (NodeID nodeId : hg.edges[i]) {
        file << " " << (nodeId + 1); // 1-indexed for hMETIS
      }
      file << "\n";
    }

    // Write node weights
    for (size_t weight : hg.nodeWeight) {
      file << weight << "\n";
    }

    file.close();
  }

  // Run KaHyPar external tool
  bool runKaHyPar(const std::string &inputFile, unsigned k,
                  std::vector<NodeID> &partitionAssignment) {
    // Create config file
    std::string configFile = "/tmp/kahypar.config";
    writeKaHyParConfig(configFile);

    // Build command
    std::vector<std::string> args = {
        "KaHyPar", "-h",       inputFile, "-k",          std::to_string(k),
        "-e",      "0.015", // imbalance factor
        "-p",      configFile, "--seed",  "-1",          "-w",
        "true",    "--mode",   "direct",  "--objective", "km1"};

    // Execute KaHyPar
    std::string outputFile = inputFile + ".part" + std::to_string(k) +
                             ".epsilon0.015.seed-1.KaHyPar";

    // For now, use system() call - in production should use proper process
    // execution
    std::string cmd = "KaHyPar -h " + inputFile + " -k " + std::to_string(k) +
                      " -e 0.015 -p " + configFile + " --seed -1 -w true " +
                      "--mode direct --objective km1";

    int result = system(cmd.c_str());
    if (result != 0) {
      llvm::errs() << "KaHyPar execution failed\n";
      return false;
    }

    // Read partition assignment
    return readPartitionFile(outputFile, partitionAssignment);
  }

  // Write KaHyPar configuration file
  void writeKaHyParConfig(const std::string &filename) {
    std::ofstream file(filename);
    file << "# KaHyPar configuration for RepCut\n";
    file << "# general\n";
    file << "mode=direct\n";
    file << "objective=km1\n";
    file << "seed=-1\n";
    file << "cmaxnet=1000\n";
    file << "vcycles=0\n";
    file << "# main -> preprocessing -> min hash sparsifier\n";
    file << "p-use-sparsifier=true\n";
    file << "p-sparsifier-min-median-he-size=28\n";
    file << "p-sparsifier-max-hyperedge-size=1200\n";
    file << "p-sparsifier-max-cluster-size=10\n";
    file << "p-sparsifier-min-cluster-size=2\n";
    file << "p-sparsifier-num-hash-func=5\n";
    file << "p-sparsifier-combined-num-hash-func=100\n";
    file << "# main -> preprocessing -> community detection\n";
    file << "p-detect-communities=true\n";
    file << "p-detect-communities-in-ip=true\n";
    file << "p-reuse-communities=false\n";
    file << "p-max-louvain-pass-iterations=100\n";
    file << "p-min-eps-improvement=0.0001\n";
    file << "p-louvain-edge-weight=hybrid\n";
    file << "# main -> coarsening\n";
    file << "c-type=ml_style\n";
    file << "c-s=1\n";
    file << "c-t=160\n";
    file << "# main -> coarsening -> rating\n";
    file << "c-rating-score=heavy_edge\n";
    file << "c-rating-use-communities=true\n";
    file << "c-rating-heavy_node_penalty=no_penalty\n";
    file << "c-rating-acceptance-criterion=best_prefer_unmatched\n";
    file << "c-fixed-vertex-acceptance-criterion=fixed_vertex_allowed\n";
    file << "# main -> initial partitioning\n";
    file << "i-mode=recursive\n";
    file << "i-technique=multi\n";
    file << "# initial partitioning -> coarsening\n";
    file << "i-c-type=ml_style\n";
    file << "i-c-s=1\n";
    file << "i-c-t=150\n";
    file << "# initial partitioning -> coarsening -> rating\n";
    file << "i-c-rating-score=heavy_edge\n";
    file << "i-c-rating-use-communities=true\n";
    file << "i-c-rating-heavy_node_penalty=no_penalty\n";
    file << "i-c-rating-acceptance-criterion=best_prefer_unmatched\n";
    file << "i-c-fixed-vertex-acceptance-criterion=fixed_vertex_allowed\n";
    file << "# initial partitioning -> initial partitioning\n";
    file << "i-algo=pool\n";
    file << "i-runs=20\n";
    file << "# initial partitioning -> bin packing\n";
    file << "i-bp-algorithm=worst_fit\n";
    file << "i-bp-heuristic-prepacking=false\n";
    file << "i-bp-early-restart=true\n";
    file << "i-bp-late-restart=true\n";
    file << "# initial partitioning -> local search\n";
    file << "i-r-type=twoway_fm\n";
    file << "i-r-runs=-1\n";
    file << "i-r-fm-stop=simple\n";
    file << "i-r-fm-stop-i=50\n";
    file << "# main -> local search\n";
    file << "r-type=kway_fm_hyperflow_cutter_km1\n";
    file << "r-runs=-1\n";
    file << "r-fm-stop=adaptive_opt\n";
    file << "r-fm-stop-alpha=1\n";
    file << "r-fm-stop-i=350\n";
    file << "# local_search -> flow scheduling and heuristics\n";
    file << "r-flow-execution-policy=exponential\n";
    file << "# local_search -> hyperflowcutter configuration\n";
    file << "r-hfc-size-constraint=mf-style\n";
    file << "r-hfc-scaling=16\n";
    file << "r-hfc-distance-based-piercing=true\n";
    file << "r-hfc-mbc=true\n";
    file.close();
  }

  // Read partition assignment from KaHyPar output
  bool readPartitionFile(const std::string &filename,
                         std::vector<NodeID> &partitionAssignment) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      llvm::errs() << "Failed to open partition file: " << filename << "\n";
      return false;
    }

    partitionAssignment.clear();
    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty()) {
        partitionAssignment.push_back(std::stoi(line));
      }
    }

    file.close();
    return true;
  }

  // Apply partition assignments to MLIR operations
  void applyPartitionAssignments(PartGraph &pg,
                                 const std::vector<NodeID> &partitionAssignment,
                                 hw::HWModuleOp moduleOp) {
    MLIRContext *context = &getContext();

    // Create mapping from tree/piece to partitions
    std::map<NodeID, std::set<NodeID>> nodeToPartitions;

    // Process partition assignment results
    for (size_t treeId = 0;
         treeId < partitionAssignment.size() && treeId < pg.trees.size();
         ++treeId) {
      NodeID partitionId = partitionAssignment[treeId];

      // All nodes in this tree belong to this partition
      for (NodeID nodeId : pg.trees[treeId]) {
        nodeToPartitions[nodeId].insert(partitionId);
      }
    }

    // Handle replication - nodes in multiple trees may belong to multiple
    // partitions
    for (size_t nodeId = 0; nodeId < pg.numNodes(); ++nodeId) {
      std::set<NodeID> allPartitions;
      for (NodeID treeId : pg.idToTreeID[nodeId]) {
        if (treeId < partitionAssignment.size()) {
          allPartitions.insert(partitionAssignment[treeId]);
        }
      }
      nodeToPartitions[nodeId] = allPartitions;
    }

    // Apply attributes to operations
    for (auto &entry : pg.opToID) {
      Operation *op = entry.first;
      NodeID nodeId = entry.second;

      const auto &partitions = nodeToPartitions[nodeId];
      if (!partitions.empty()) {
        // Create array attribute with partition IDs
        std::vector<Attribute> partitionAttrs;
        for (NodeID partId : partitions) {
          partitionAttrs.push_back(IntegerAttr::get(
              IntegerType::get(context, 64, IntegerType::Unsigned), partId));
        }

        ArrayAttr partitionArrayAttr = ArrayAttr::get(context, partitionAttrs);
        op->setAttr("repcut_partitions", partitionArrayAttr);
      }
    }
  }
};

} // namespace
