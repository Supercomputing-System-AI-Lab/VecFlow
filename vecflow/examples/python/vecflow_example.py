import numpy as np
from vecflow import VecFlow
import time
import os
import argparse
import json
import cupy as cp

def load_fbin(fname):
  """Load float32 binary file."""
  with open(fname, "rb") as f:
    # Read dimensions
    vec_count = np.fromfile(f, dtype=np.int32, count=1)[0]
    dim = np.fromfile(f, dtype=np.int32, count=1)[0]
    # Read data
    data = np.fromfile(f, dtype=np.float32, count=vec_count * dim)
  return data.reshape(vec_count, dim)

def load_labels(fname):
  """Load labels directly as Python list of lists"""
  with open(fname, 'rb') as f:
    nrow, ncol, nnz = np.fromfile(f, dtype=np.int64, count=3)
    indptr = np.fromfile(f, dtype=np.int64, count=nrow+1)
    indices = np.fromfile(f, dtype=np.int32, count=nnz)

  # Convert to list of lists directly
  labels_list = [indices[indptr[i]:indptr[i+1]].tolist()
                  for i in range(nrow)]
  return labels_list, ncol

def load_txt_labels(fname):
  """Load labels from text file format"""
  with open(fname, 'r') as f:
    lines = f.readlines()
  
  labels_list = []
  max_label = 0
  for line in lines:
    line = line.strip()
    if line and line != "-1":
      # Split by comma and convert to int
      labels = [int(label) for label in line.split(',')]
      labels_list.append(labels)
      if labels:
        max_label = max(max_label, max(labels))
    else:
      # "-1" means no labels
      labels_list.append([])
  
  ncol = max_label + 1 if max_label > 0 else 0
  return labels_list, ncol

def load_labels_auto(fname):
  """Load labels automatically detecting file type by extension"""
  if fname.endswith('.txt'):
    return load_txt_labels(fname)
  elif fname.endswith('.spmat'):
    return load_labels(fname)

def compute_recall(neighbors, gt_indices):
  """
  Compute recall of search results against ground truth.
  
  Parameters:
  -----------
  neighbors : numpy.ndarray or list
      Search results (nearest neighbor indices)
  gt_indices : numpy.ndarray or list
      Ground truth indices
      
  Returns:
  --------
  float
      Average recall
  """
  if isinstance(neighbors, list):
    neighbors = np.array(neighbors)
  if isinstance(gt_indices, list):
    gt_indices = np.array(gt_indices)
  
  # Get dimensions
  n_queries = neighbors.shape[0]
  if n_queries == 0:
    return 0.0
  
  if len(neighbors.shape) == 2 and len(gt_indices.shape) == 2:
    # Compute recall for each query
    recalls = np.zeros(n_queries)
    for i in range(n_queries):
      # Convert to set for efficient intersection
      neighbors_set = set(neighbors[i])
      gt_set = set(gt_indices[i])
      
      if len(gt_set) == 0:
        recalls[i] = 1.0  # If no ground truth, count as perfect recall
      else:
        # Compute intersection size
        intersection_size = len(neighbors_set.intersection(gt_set))
        recalls[i] = intersection_size / len(gt_set)
    
    # Return average recall across all queries
    return np.mean(recalls)
  else:
    raise ValueError("Both neighbors and gt_indices must be 2D arrays or lists of lists")

def main():
  parser = argparse.ArgumentParser(description='Test VecFlow API')
  parser.add_argument('--config', type=str, default='config.json',
            help='Path to configuration JSON file')
  args = parser.parse_args()

  # Load configuration
  try:
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
      config = json.load(f)
    
    # Extract parameters from config
    data_dir = config.get('data_dir')
    data_fname = config.get('data_fname')
    query_fname = config.get('query_fname')
    data_label_fname = config.get('data_label_fname')
    query_label_fname = config.get('query_label_fname')
    itopk_size = config.get('itopk_size', 32)
    specificity_threshold = config.get('spec_threshold', 1000)
    graph_degree = config.get('graph_degree', 16)
    topk = config.get('topk', 10)
    num_runs = config.get('num_runs', 1000)
    warmup_runs = config.get('warmup_runs', 10)
    force_rebuild = config.get('force_rebuild', False)
    ivf_graph_fname = config.get('ivf_graph_fname')
    ivf_bfs_fname = config.get('ivf_bfs_fname')
    ground_truth_fname = config.get('ground_truth_fname')
    
  except Exception as e:
    print(f"Error loading config file: {e}")
    return

  # Construct full file paths
  full_data_fname = os.path.join(data_dir, data_fname)
  full_query_fname = os.path.join(data_dir, query_fname)
  full_data_label_fname = os.path.join(data_dir, data_label_fname)
  full_query_label_fname = os.path.join(data_dir, query_label_fname)
  full_ivf_graph_fname = os.path.join(data_dir, ivf_graph_fname)
  full_ivf_bfs_fname = os.path.join(data_dir, ivf_bfs_fname)
  full_ground_truth_fname = os.path.join(data_dir, ground_truth_fname)
  
  # Print configuration 
  print("\n=== Configuration ===")
  print(f"iTopK size: {itopk_size}")
  print(f"Specificity threshold: {specificity_threshold}")
  print(f"Graph degree: {graph_degree}")
  print(f"TopK: {topk}")
  print(f"Number of runs: {num_runs}")
  print(f"Warmup runs: {warmup_runs}")

  print("\n=== Loading Dataset ===")
  dataset = load_fbin(full_data_fname)
  queries = load_fbin(full_query_fname)
  data_labels, _ = load_labels_auto(full_data_label_fname)
  query_labels, _ = load_labels_auto(full_query_label_fname)

  print(f"\n=== Dataset Information ===")
  print(f"Base dataset size: N={dataset.shape[0]}, dim={dataset.shape[1]}")
  print(f"Query dataset size: N={queries.shape[0]}, dim={queries.shape[1]}")

  # Initialize VecFlow
  vf = VecFlow()

  # Build index
  print("\n=== Building Index ===")
  start_time = time.time()
  vf.build(dataset,
           data_labels,
           graph_degree,
           specificity_threshold,
           full_ivf_graph_fname,
           full_ivf_bfs_fname,
           force_rebuild)
  build_time = time.time() - start_time
  print(f"Index building time: {build_time:.2f} seconds")

  # Extract the first label for each query or use -1 if no labels
  query_label_arr = np.array([labels[0] if labels else -1 for labels in query_labels])
  
  print("Converting to GPU arrays...")
  queries_gpu = cp.asarray(queries, dtype=cp.float32)
  query_label_arr_gpu = cp.asarray(query_label_arr, dtype=cp.int32)
  
  # Create NumPy arrays that reference GPU memory
  import ctypes
  queries_np_gpu = np.ctypeslib.as_array(
      ctypes.cast(queries_gpu.data.ptr, ctypes.POINTER(ctypes.c_float)),
      shape=queries_gpu.shape
  )
  query_labels_np_gpu = np.ctypeslib.as_array(
      ctypes.cast(query_label_arr_gpu.data.ptr, ctypes.POINTER(ctypes.c_int32)),
      shape=query_label_arr_gpu.shape
  )
  
  # Search with warmup
  print("\n=== Performing Search ===")
  # Warm-up runs
  for i in range(warmup_runs):
    _, _ = vf.search(queries_np_gpu, query_labels_np_gpu, itopk_size)
  
  # Benchmark runs
  start_time = time.perf_counter()
  for _ in range(num_runs):
    neighbors, distances = vf.search(queries_np_gpu, query_labels_np_gpu, itopk_size)
  total_time = time.perf_counter() - start_time
  avg_ms = (total_time * 1000) / num_runs
  qps = num_runs * queries.shape[0] / total_time
  print(f"Search timing ({num_runs} runs):")
  print(f"- Total time: {total_time*1000:.2f} ms")
  print(f"- Average per search: {avg_ms:.2f} ms")
  print(f"- QPS: {qps:.2f}")

  # Generate or load ground truth
  gt_indices = vf.generate_ground_truth(dataset, 
                                        queries, 
                                        data_labels, 
                                        query_labels,
                                        topk,
                                        full_ground_truth_fname)
  # Compute recall
  recall = compute_recall(neighbors, gt_indices)
  print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
  main()