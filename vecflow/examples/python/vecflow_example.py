import numpy as np
from vecflow import VecFlow
import time
import os
import argparse

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

def spmat2txt(input_file, output_file):
  """
    Save labels to a text file where:
    - Each row represents a data point
    - Labels are comma-separated
    - A row with only -1 means no labels for that data point
  """
  labels_list, ncol = load_labels(input_file)
  with open(output_file, 'w') as f:
    for i, labels in enumerate(labels_list):
      if labels:
        line = ','.join(map(str, [label for label in labels]))
      else:
        # Output "-1" for rows with no labels
        line = '-1'
      
      if i < len(labels_list) - 1:
        f.write(line + '\n')
      else:
        f.write(line)

def txt2spmat(input_file, output_file):
    """
    Convert a text file with label lists to spmat format
    - Each input line contains comma-separated label indices
    - Lines with "-1" indicate no labels for that row
    - All labels are decremented by 1 to convert from 1-indexed to 0-indexed
    """
    with open(input_file, 'r') as f:
      lines = f.readlines()
    
    labels_list = []
    for line in lines:
      line = line.strip()
      if line and line != "-1":
        # Split by comma and convert to int
        labels = [int(label) for label in line.split(',')]
        labels_list.append(labels)
      else:
        # "-1" means no labels
        labels_list.append([])
    
    nrow = len(labels_list)
    if nrow == 0:
      ncol = 0
      nnz = 0
    else:
      max_label = max([max(labels) if labels else 0 for labels in labels_list])
      ncol = max_label + 1
      nnz = sum(len(labels) for labels in labels_list)
    
    indptr = np.zeros(nrow + 1, dtype=np.int64)
    indices = np.zeros(nnz, dtype=np.int32)
    idx = 0
    for i, labels in enumerate(labels_list):
      indptr[i] = idx
      for label in labels:
          indices[idx] = label
          idx += 1
    indptr[nrow] = nnz
    
    with open(output_file, 'wb') as f:
      np.array([nrow, ncol, nnz], dtype=np.int64).tofile(f)
      indptr.tofile(f)
      indices.tofile(f)

def load_groundtruth(fname):
  """Load groundtruth from binary file."""
  gt_indices = []
  with open(fname, "rb") as f:
    rows = np.fromfile(f, dtype=np.int64, count=1)[0]
    cols = np.fromfile(f, dtype=np.int64, count=1)[0]
    data = np.fromfile(f, dtype=np.uint32, count=rows * cols)
    data = data.reshape(rows, cols)
    for i in range(rows):
      gt_indices.append(data[i].tolist())
  return gt_indices

def main():
  parser = argparse.ArgumentParser(description='Test VecFlow API')
  parser.add_argument('--data_dir', type=str, default='../data/',
            help='Directory containing dataset files')
  parser.add_argument('--itopk_size', type=int, default=32,
            help='Internal topk size for search')
  parser.add_argument('--spec_threshold', type=int, default=1000,
            help='Specificity threshold')
  parser.add_argument('--graph_degree', type=int, default=16,
            help='Graph degree')
  args = parser.parse_args()

  # File paths
  base_path = os.path.join(args.data_dir, "sift1M")
  data_fname = os.path.join(base_path, "sift.base.fbin")
  data_label_fname = os.path.join(base_path, "sift.base.spmat")
  query_fname = os.path.join(base_path, "sift.query.fbin")
  query_label_fname = os.path.join(base_path, "sift.query.spmat")
  gt_fname = os.path.join(base_path, "sift.groundtruth.neighbors.ibin")
  
  # Check if the binary spmat files exist and convert from text format if needed
  if not os.path.exists(data_label_fname):
    txt_file = os.path.join(base_path, "sift.base.txt")
    txt2spmat(txt_file, data_label_fname)
  if not os.path.exists(query_label_fname):
    txt_file = os.path.join(base_path, "sift.query.txt")
    txt2spmat(txt_file, query_label_fname)

  # Graph file names
  graph_fname = os.path.join(base_path,
              f"graph_{args.graph_degree * 2}_{args.graph_degree}_spec_{args.spec_threshold}.bin")
  bfs_fname = os.path.join(base_path,
              f"spec_{args.spec_threshold}.bin")

  print("\n=== Loading Dataset ===")
  dataset = load_fbin(data_fname)
  queries = load_fbin(query_fname)
  query_labels, _ = load_labels(query_label_fname)
  gt_indices = load_groundtruth(gt_fname)

  print(f"Base dataset size: {dataset.shape}")
  print(f"Query dataset size: {queries.shape}")

  # Initialize VecFlow
  vf = VecFlow()

  # Build index
  print("\n=== Building Index ===")
  start_time = time.time()
  vf.build(dataset,
           data_label_fname,
           args.graph_degree,
           args.spec_threshold,
           graph_fname,
           bfs_fname)
  build_time = time.time() - start_time
  print(f"Index building time: {build_time:.2f} seconds")

  # Search
  print("\n=== Performing Search ===")
  query_labels = np.array([i[0] for i in query_labels])
  for i in range(10):
      _, _ = vf.search(queries, query_labels, args.itopk_size)
  num_searches = 1000
  start_time = time.perf_counter()
  for _ in range(num_searches):
      neighbors, distances = vf.search(queries, query_labels, args.itopk_size)
  total_time = time.perf_counter() - start_time
  avg_ms = (total_time * 1000) / num_searches
  qps = num_searches * queries.shape[0] / total_time
  print(f"Search timing ({num_searches} runs):")
  print(f"- Total time: {total_time*1000:.2f} ms")
  print(f"- Average per search: {avg_ms:.2f} ms")
  print(f"- QPS: {qps:.2f}")

  # Compute recall
  print("\n=== Computing Recall ===")
  recall = vf.compute_recall(neighbors, np.array(gt_indices))
  print(f"Overall recall: {recall:.6f}")

if __name__ == "__main__":
  main()