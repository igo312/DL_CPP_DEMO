#ifndef TEST_SAMPLE_CALLBACK_H
#define TEST_SAMPLE_CALLBACK_H

#include <dlnne_build_modulator.h>
#include <functional>
#include <map>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

using ScheduleDAGFunc = std::function<bool(dl::nne::IDAG *)>;
using SubGraphGenFunc =
    std::function<bool(const dl::nne::IDAG *, dl::nne::ISubgraphContainer *)>;
using ScheduleSubGraphFunc = std::function<bool(const dl::nne::IDAG *)>;
using AllocSubGraphFunc = std::function<void()>;

class CallbackImpl : public dl::nne::IBuildModulator {
 public:
  CallbackImpl() {
    schedule_dag_callback_ = [](dl::nne::IDAG *) { return false; };
    sub_graph_generate_callback_ = [](const dl::nne::IDAG *,
                                      dl::nne::ISubgraphContainer *container) {
      return false;
    };
    schedule_sub_graph_callback_ = [](const dl::nne::IDAG *) { return false; };
    alloc_sub_graph_callback_ = []() {};
  }

  virtual ~CallbackImpl() {}

  bool ScheduleDAG(dl::nne::IDAG *graph) override {
    return schedule_dag_callback_(graph);
  }

  bool GenerateSubgraph(const dl::nne::IDAG *graph,
                        dl::nne::ISubgraphContainer *container) override {
    return sub_graph_generate_callback_(graph, container);
  }

 public:
  void setScheduleDagCallback(const ScheduleDAGFunc &callback) {
    schedule_dag_callback_ = callback;
  }

  void setSubGraphGenerateCallback(const SubGraphGenFunc &callback) {
    sub_graph_generate_callback_ = callback;
  }

  void setScheduleSubGraphCallback(const ScheduleSubGraphFunc &callback) {
    schedule_sub_graph_callback_ = callback;
  }

  void setAllocSubGraphCallback(const AllocSubGraphFunc &callback) {
    alloc_sub_graph_callback_ = callback;
  }

 private:
  ScheduleDAGFunc schedule_dag_callback_;
  SubGraphGenFunc sub_graph_generate_callback_;
  ScheduleSubGraphFunc schedule_sub_graph_callback_;
  AllocSubGraphFunc alloc_sub_graph_callback_;
};

void MergeAllNodes(const dl::nne::IDAG *graph,
                   dl::nne::ISubgraphContainer *container);
void MergeNodes(const dl::nne::IDAG *graph,
                dl::nne::ISubgraphContainer *container,
                std::vector<std::string> split_nodes);
void MergeSegment(const dl::nne::IDAG *graph,
                dl::nne::ISubgraphContainer *container,
                std::vector<std::map<std::string, std::string>> split_nodes_map);
void MergeSegmentToSubgraph(const dl::nne::IDAG *graph,
                            dl::nne::ISubgraphContainer *container,
                            std::unordered_set<std::string> sub_graph_boundary,
                            std::unordered_set<std::string> segment_end,
                            std::vector<std::pair<int, std::pair<int, bool>>>
                                subgraph_merge_segment_vec);                                                   
#endif  // TEST_SAMPLE_CALLBACK_H
