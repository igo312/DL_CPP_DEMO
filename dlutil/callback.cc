#include "callback.h"

void MergeAllNodes(const dl::nne::IDAG *graph,
                   dl::nne::ISubgraphContainer *container) {
  std::map<std::string, dl::nne::IDAGNode *> name_node_map;
  dl::nne::IDAGNodeList sg_node_list;

  auto node_list = graph->GetNodes();
  for (auto node : node_list) {
    name_node_map[node->GetName()] = node;
    sg_node_list.push_back(node);
  }

  container->AddSubgraph(&sg_node_list);
}

void MergeNodes(const dl::nne::IDAG *graph,
                dl::nne::ISubgraphContainer *container,
                std::vector<std::string> split_nodes) {
  std::map<std::string, dl::nne::IDAGNode *> name_node_map;
  dl::nne::IDAGNodeList sg_node_list;
  std::vector<std::vector<std::string>> combine_config;
  combine_config.resize(split_nodes.size());
  uint32_t current_config_index = 0;

  auto node_list = graph->GetNodes();
  for (auto node : node_list) {
    name_node_map[node->GetName()] = node;
  }

  for (auto node : node_list) {
    combine_config[current_config_index].push_back(node->GetName());
    if (node->GetName() == split_nodes[current_config_index]) {
      current_config_index++;
    }
  }

  for (auto &config : combine_config) {
    sg_node_list.clear();
    for (auto &name : config) {
      sg_node_list.push_back(name_node_map[name]);
    }
    if (sg_node_list.size() > 0) container->AddSubgraph(&sg_node_list);
  }
}

void MergeSegment(const dl::nne::IDAG *graph,
                dl::nne::ISubgraphContainer *container,
                std::vector<std::map<std::string, std::string>> split_nodes_map)
{
  std::map<std::string, dl::nne::IDAGNode *> name_node_map;
  dl::nne::IDAGNodeList sg_node_list;
  std::vector<std::vector<std::string>> combine_config;
  combine_config.resize(split_nodes_map.size());

  auto node_list = graph->GetNodes();
  for (auto node : node_list) {
    name_node_map[node->GetName()] = node;
    std::cout << node->GetName() << std::endl;
  }

  auto node_iter = node_list.begin();
  while(node_iter != node_list.end()) {
      bool found = false;
      for(uint i = 0; i < split_nodes_map.size() && !found; i++) {
          auto map_iter = split_nodes_map[i].begin();
          while(map_iter != split_nodes_map[i].end() && !found) {
              if(0 == (*node_iter)->GetName().compare( (*map_iter).first )) {
                  do {
                      combine_config[i].push_back((*node_iter)->GetName());
                      node_iter++;
                  } while(0 != (*node_iter)->GetName().compare( (*map_iter).second ));
                  combine_config[i].push_back((*node_iter)->GetName());
                  node_iter++;
                  found = true;
              }
              map_iter++;
          }
      }
  }

  for (auto &config : combine_config) {
    sg_node_list.clear();
    for (auto &name : config) {
      sg_node_list.push_back(name_node_map[name]);
    }
    if (sg_node_list.size() > 0) container->AddSubgraph(&sg_node_list);
  }
}


void MergeSegmentToSubgraph(const dl::nne::IDAG *graph,
                            dl::nne::ISubgraphContainer *container,
                            std::unordered_set<std::string> sub_graph_boundary,
                            std::unordered_set<std::string> segment_end,
                            std::vector<std::pair<int, std::pair<int, bool>>>
                                subgraph_merge_segment_vec) {
  int sub_graph_index = 0, segment_index = 0;
  std::unordered_map<int, dl::nne::IDAGNodeList> sub_graph_node_list_map;
  std::unordered_map<int, dl::nne::IDAGNodeList> segment_node_list_map;

  dl::nne::IDAGNodeList dag_node_list;
  auto node_list = graph->GetNodes();

  for (auto node : node_list) {
    if (sub_graph_boundary.count(node->GetName()) != 0) {
      // found sub graph end node
      dag_node_list.push_back(node);
      sub_graph_node_list_map[sub_graph_index++] = dag_node_list;
      dag_node_list.clear();
      continue;
    } else if (segment_end.count(node->GetName()) != 0) {
      // found segment end node
      dag_node_list.push_back(node);
      segment_node_list_map[segment_index++] = dag_node_list;
      dag_node_list.clear();
      continue;
    } else {
      // internal node or begin node
      dag_node_list.push_back(node);
    }
  }
  // splice segment to subgraph
  for (auto sub_seg : subgraph_merge_segment_vec) {
    auto position = sub_seg.second.second
                        ? sub_graph_node_list_map[sub_seg.first].begin()
                        : sub_graph_node_list_map[sub_seg.first].end();
    sub_graph_node_list_map[sub_seg.first].splice(
        position, segment_node_list_map[sub_seg.second.first]);
  }
  // add subgraph
  for (auto i = 0; i < sub_graph_node_list_map.size(); i++) {
    if (sub_graph_node_list_map[i].size() > 0) {
      container->AddSubgraph(&sub_graph_node_list_map[i]);
    } else {
      std::fprintf(stderr, "Error:  Subgraph %d is empty.\n", i);
      std::fflush(stderr);
    }
  }
}
