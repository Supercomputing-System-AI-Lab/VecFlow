# Benchmark Guide

## Dataset Information

The benchmark datasets are available at [Google Drive](https://drive.google.com/drive/folders/1TFHifsmXZlAkVR7MHrFXYuDUYfpWQMz2?usp=sharing). Each dataset includes:

- Dataset file
- Index file

## Benchmark Types

### Inline Filtering Benchmarks

Files with "_if_" in their names implement inline filtering using predicate functions on a single graph index:

- `search_cagra_if_sift.cu`
- `search_cagra_if_yfcc.cu`

### Standard Search Benchmarks

Files without "if" perform IVF-CAGRA + IVF-BFS Search:

- `search_cagra_sift.cu`
- `search_cagra_wiki.cu`
- `search_cagra_yfcc.cu`

## Dataset-Specific Search Strategies

### YFCC Dataset

The search strategy for YFCC uses a hybrid approach based on query types:

1. Double-label queries (all labels):
   - Uses IVF-CAGRA
   - Searches within smaller clusters only with predicate function

2. Single-label queries:
   - Routes to either IVF-CAGRA or IVF-BFS based on specificity threshold

### SIFT Dataset

- Uses IVF-CAGRA exclusively due to high label-specificity
- Direct search without routing

## Running Benchmarks

### File Configuration

For each file, modify:

1. Dataset path
2. Index file path

### Building Index Files (Index files are already included in Google Drive)

Two separate build scripts are provided:

- `build_sift.cu`: For building SIFT dataset index
- `build_wiki.cu`: For building Wiki dataset index