import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

# Function to compute shortest paths between pairs of nodes
def compute_shortest_paths(G):
    return dict(nx.all_pairs_dijkstra_path_length(G))

# --- APLS: Average Path Length Similarity ---
def APLS(ground_truth_graph, predicted_graph):
    # Compute the shortest paths for both graphs
    gt_shortest_paths = compute_shortest_paths(ground_truth_graph)
    pred_shortest_paths = compute_shortest_paths(predicted_graph)
    
    # Calculate the relative length difference for each corresponding path
    total_difference = 0
    path_count = 0
    
    for node1 in gt_shortest_paths:
        for node2 in gt_shortest_paths[node1]:
            if node2 in pred_shortest_paths[node1]:
                gt_length = gt_shortest_paths[node1][node2]
                pred_length = pred_shortest_paths[node1][node2]
                total_difference += abs(gt_length - pred_length) / gt_length
                path_count += 1
    
    # Return the average relative length difference
    if path_count == 0:
        return 0  # Avoid division by zero
    return total_difference / path_count

# --- TLTS: Total Length of Shortest Paths ---
def TLTS(ground_truth_graph, predicted_graph, threshold=0.05):
    # Compute the shortest paths for both graphs
    gt_shortest_paths = compute_shortest_paths(ground_truth_graph)
    pred_shortest_paths = compute_shortest_paths(predicted_graph)
    
    # Calculate the fraction of paths where relative length difference is within 5%
    correct_paths = 0
    total_paths = 0
    
    for node1 in gt_shortest_paths:
        for node2 in gt_shortest_paths[node1]:
            if node2 in pred_shortest_paths[node1]:
                gt_length = gt_shortest_paths[node1][node2]
                pred_length = pred_shortest_paths[node1][node2]
                if abs(gt_length - pred_length) / gt_length <= threshold:
                    correct_paths += 1
                total_paths += 1
    
    # Return the fraction of paths with length difference within the threshold
    if total_paths == 0:
        return 0  # Avoid division by zero
    return correct_paths / total_paths

# --- JCT: Junction Connectivity ---
def JCT(ground_truth_graph, predicted_graph):
    # Junctions are nodes in the graph, and their degree corresponds to the number of roads intersecting
    # Compute the junction recall and precision
    gt_junctions = set(ground_truth_graph.nodes())
    pred_junctions = set(predicted_graph.nodes())
    
    # Compute the recall and precision at each junction
    recall = len(gt_junctions & pred_junctions) / len(gt_junctions) if len(gt_junctions) > 0 else 0
    precision = len(gt_junctions & pred_junctions) / len(pred_junctions) if len(pred_junctions) > 0 else 0
    
    # Compute the F1 score for junctions
    if recall + precision == 0:
        return 0  # Avoid division by zero
    return 2 * (precision * recall) / (precision + recall)

# --- HM: Hub Matching (Comparing reachable locations) ---
def HM(ground_truth_graph, predicted_graph):
    # For each pair of corresponding points, find the set of reachable nodes
    gt_reachable = {}
    pred_reachable = {}
    
    # Compute reachable nodes using breadth-first search or Dijkstra's algorithm
    for node in ground_truth_graph.nodes():
        gt_reachable[node] = set(nx.single_source_shortest_path_length(ground_truth_graph, node).keys())
    for node in predicted_graph.nodes():
        pred_reachable[node] = set(nx.single_source_shortest_path_length(predicted_graph, node).keys())
    
    # Compare the sets of reachable nodes and compute the F1 score
    recall = 0
    precision = 0
    total = 0
    
    for node in gt_reachable:
        if node in pred_reachable:
            recall += len(gt_reachable[node] & pred_reachable[node]) / len(gt_reachable[node])
            precision += len(gt_reachable[node] & pred_reachable[node]) / len(pred_reachable[node])
            total += 1
    
    # Compute average F1 score over all nodes
    if total == 0:
        return 0  # Avoid division by zero
    recall /= total
    precision /= total
    return 2 * (precision * recall) / (precision + recall)

# --- CCQ: Correctness, Completeness, and Quality ---
def CCQ(ground_truth_mask, predicted_mask, distance_threshold=5):
    """
    Correctness, Completeness, and Quality (CCQ)
    - Correctness: Precision
    - Completeness: Recall
    - Quality: Intersection over Union (IoU) with a relaxation of co-occurrence within 5 pixels.
    """
    # Generate binary masks (True = road, False = non-road)
    gt = ground_truth_mask > 0.5
    pred = predicted_mask > 0.5
    
    # Correctness (Precision)
    tp = torch.sum(gt & pred).item()
    fp = torch.sum(~gt & pred).item()
    fn = torch.sum(gt & ~pred).item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # Quality (IoU with 5 pixel relaxation)
    # Use a distance-based IoU by allowing a 5-pixel distance tolerance for co-occurrence
    quality = torch.sum((gt | pred) & (torch.abs(gt - pred) <= distance_threshold)).item() / torch.sum(gt | pred).item()
    
    return {"Precision": precision, "Recall": recall, "IoU": iou, "Quality": quality}

