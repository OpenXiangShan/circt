//===-- Passes.td - HW pass definition file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_PASSES_TD
#define CIRCT_DIALECT_HW_PASSES_TD

include "mlir/Pass/PassBase.td"

def PrintInstanceGraph : Pass<"hw-print-instance-graph", "mlir::ModuleOp"> {
  let summary = "Print a DOT graph of the module hierarchy.";
}

def PrintHWModuleGraph : Pass<"hw-print-module-graph", "mlir::ModuleOp"> {
  let summary = "Print a DOT graph of the HWModule's within a top-level module.";
  let options = [
    Option<"verboseEdges", "verbose-edges", "bool", "false",
      "Print information on SSA edges (types, operand #, ...)">,
  ];
}

def FlattenIO : Pass<"hw-flatten-io", "mlir::ModuleOp"> {
  let summary = "Flattens hw::Structure typed in- and output ports.";
  let options = [
    Option<"recursive", "recursive", "bool", "true",
      "Recursively flatten nested structs.">,
    Option<"flattenExtern", "flatten-extern", "bool", "false",
      "Flatten the extern modules also.">,
    Option<"joinChar", "join-char", "char", "'.'",
      "Use a custom character to construct the flattened names.">,
  ];
}

def FlattenModules : Pass<"hw-flatten-modules", "mlir::ModuleOp"> {
  let summary = "Eagerly inline private modules";
  let description = [{
    This pass eagerly inlines private HW modules into their instantiation sites.
    This is necessary for verification purposes, as model checking backends do not
    require or support the use of module hierarchy. For simulation, module hierarchies
    degenerate into a purely cosmetic construct, at which point it is beneficial
    to fully flatten the module hierarchy to simplify further analysis and
    optimization of state transfer arcs.
  }];
}

def HWSpecialize : Pass<"hw-specialize", "mlir::ModuleOp"> {
  let summary = "Specializes instances of parametric hw.modules";
  let description = [{
    Any `hw.instance` operation instantiating a parametric `hw.module` will
    trigger a specialization procedure which resolves all parametric types and
    values within the module based on the set of provided parameters to the
    `hw.instance` operation. This specialized module is created as a new
    `hw.module` and the referring `hw.instance` operation is rewritten to
    instantiate the newly specialized module.
  }];
}

def VerifyInnerRefNamespace : Pass<"hw-verify-irn"> {
  let summary = "Verify InnerRefNamespaceLike operations, if not self-verifying.";
}

/**
 * Tutorial Pass, doesn't do anything interesting
 */
def FooWires : Pass<"hw-foo-wires", "hw::HWModuleOp"> {
  let summary = "Change all wires' name to foo_<n>.";
  let description = [{
    Very basic pass that numbers all of the wires in a given module.
    The wires' names are then all converte to foo_<that number>.
  }];
}

def HWAggregateToComb : Pass<"hw-aggregate-to-comb", "hw::HWModuleOp"> {
  let summary = "Lower aggregate operations to comb operations";
  let description = [{
    This pass lowers aggregate *operations* to comb operations within modules.
    Note that this pass does not lower ports. Ports lowering is handled
    by FlattenIO.

    This pass will change the behavior of out-of-bounds access of arrays,
    specifically the last element of the array is used as a value for
    out-of-bounds access.
  }];
  let dependentDialects = ["comb::CombDialect"];
}

def HWReduceMemReadLatency : Pass<"hw-reduce-mem-read-latency", "mlir::ModuleOp"> {
  let summary = "Reduce FirMem read latency with extra FirReg";
  let description = [{
    This pass reduce `seq.firmem` with readLatency > N to N with extra `seq.firreg`.
  }];
  let options = [
    Option<"latency", "latency", "uint32_t", "1",
      "Target read latency (default: 1)">
  ];
}

def HWAssignUniqueInnerSym : Pass<"hw-assign-unique-inner-sym", "mlir::ModuleOp"> {
  let summary = "Assign unique InnerSym for firreg and firmem ports";
  let description = [{
    This pass add unique InnerSym for all `seq::FirRegOp`, `seq::FirMemReadOp`,
    `seq::FirMemWriteOp` and `seq::FirMemReadWriteOp` in a module.
  }];
}

def HWAssignUniqueID : Pass<"hw-assign-unique-id", "mlir::ModuleOp"> {
  let summary = "Assign unique ID for firreg and firmem ports";
  let description = [{
    This pass add unique ID for all `seq::FirRegOp`, `seq::FirMemReadOp`,
    `seq::FirMemWriteOp` and `seq::FirMemReadWriteOp` in a module.
  }];
}

def HWInsertWires : Pass<"hw-insert-wires", "hw::HWModuleOp"> {
  let summary = "Insert wires for direct connections between module ports and seq operations";
  let description = [{
    This pass inserts hw.wire operations between direct connections involving:
    - hw.module IOs and seq register/memory operations
    - Direct connections between seq register/memory operations
    
    The target operations are seq::FirRegOp, seq::FirMemReadOp, 
    seq::FirMemWriteOp, and seq::FirMemReadWriteOp.
  }];
}

def HWSplitSeqComb : Pass<"hw-split-seq-comb", "mlir::ModuleOp"> {
  let summary = "Split flattened module into sequential S and combinational C modules";
  let description = [{
    Duplicate the flattened hw.module into S and C. Keep hw.constant in both.
    In S: keep only seq ops (FirRegOp/FirMem*), replace their non-constant operands
    with new inputs, and export their results as outputs. Remove non-seq/non-const ops.
    In C: add outputs for values that feed seq op operands (non-const), add inputs for
    seq op results and replace uses, then erase seq ops. The original module is erased.
  }];
}

def HWCombmemToReg : Pass<"hw-combmem-to-reg", "mlir::ModuleOp"> {
  let summary = "Convert combmem to register";
  let description = [{
    This pass convert firmem with readLatency=0 and related read/write operations to firreg
  }];
  let dependentDialects = ["comb::CombDialect"];
}

def HWRepCut : Pass<"hw-repcut", "hw::HWModuleOp"> {
  let summary = "RepCut algorithm for replication-aided graph partitioning";
  let description = [{
    This pass implements the RepCut algorithm for partitioning hardware designs
    with replication-aided graph partitioning. The algorithm:
    1. Builds a dependency graph from MLIR operations
    2. Creates cones (trees) from sink nodes backwards to register boundaries
    3. Groups nodes into pieces based on shared cone membership  
    4. Builds an intersection hypergraph with sink pieces as vertices and 
       non-sink pieces as hyperedges
    5. Uses KaHyPar to partition the hypergraph
    6. Assigns partition IDs to each operation as "repcut_partitions" attribute
    
    Operations may belong to multiple partitions due to replication.
    Requires KaHyPar external tool to be available in PATH.
  }];
  let options = [
    Option<"numPartitions", "num-partitions", "unsigned", "2",
      "Number of partitions to create (default: 2)">
  ];
  let dependentDialects = ["seq::SeqDialect", "comb::CombDialect"];
}

#endif // CIRCT_DIALECT_HW_PASSES_TD
