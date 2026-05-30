#include <vecflow_chamfer/serialize.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <raft/core/copy.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <cuvs/neighbors/cagra.hpp>

namespace vecflow_chamfer {

namespace {

// On-disk layout. Each index lands as two files sharing a common stem:
//   <stem>.maxivf  : the underlying CAGRA index over anchor centroids
//                    (cuVS native serialization format)
//   <stem>.meta    : the vecflow-chamfer sidecar described below
//
// .meta file format. The file starts with a 4-byte magic followed by a u32
// format version; the rest of the layout depends on the version.
//
// version 2 (current):
//   magic 'vfcm'                                  : 4 bytes
//   version                                       : u32 (= 2)
//   --- MaxIVF core ---
//   n_anchor_labels                               : u32
//   anchor_labels       [n_anchor_labels]         : u32 * n
//   n_anchor_to_doc_ids                           : u32
//   anchor_to_doc_ids   [...]                     : u32 * n
//   n_offsets                                     : u32
//   anchor_to_doc_ids_offsets [...]               : u32 * n
//   n_doc_offsets                                 : u32
//   doc_offsets         [...]                     : u32 * n
//   --- VPQ tail ---
//   has_vpq                                       : u8 (0 or 1)
//   if has_vpq:
//     pq_bits, pq_dim, pq_len, n_pq_centers       : u32 * 4
//     n_tokens, quantized_dim                     : u32 * 2
//     pq_codebook  [n_pq_centers * pq_len]        : float * n
//     codes        [n_tokens * quantized_dim]     : u8    * n

constexpr const char* kCagraSuffix = ".maxivf";
constexpr const char* kMetaSuffix  = ".meta";
constexpr uint32_t kMetaMagic   = 0x6D636676u;  // 'v','f','c','m' little-endian
constexpr uint32_t kMetaVersion = 2u;

template <class T>
void write_count_and_bytes(std::ofstream& f, const std::vector<T>& v) {
  static_assert(sizeof(T) == sizeof(uint32_t), "this helper assumes 4-byte payload elements");
  uint32_t n = static_cast<uint32_t>(v.size());
  f.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
  f.write(reinterpret_cast<const char*>(v.data()),
          static_cast<std::streamsize>(n) * sizeof(T));
}

template <class T>
void read_count_and_bytes(std::ifstream& f, std::vector<T>& v) {
  static_assert(sizeof(T) == sizeof(uint32_t), "this helper assumes 4-byte payload elements");
  uint32_t n = 0;
  if (!f.read(reinterpret_cast<char*>(&n), sizeof(uint32_t))) {
    throw std::runtime_error("vecflow-chamfer meta file truncated (count header)");
  }
  v.resize(n);
  if (n > 0 &&
      !f.read(reinterpret_cast<char*>(v.data()),
              static_cast<std::streamsize>(n) * sizeof(T))) {
    throw std::runtime_error("vecflow-chamfer meta file truncated (payload)");
  }
}

template <class T>
std::vector<T> d2h(raft::device_resources const& res,
                   raft::device_vector<T, int64_t> const& v) {
  std::vector<T> h(v.size());
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(h.data(), v.data_handle(), v.size(), stream);
  raft::resource::sync_stream(res);
  return h;
}

template <class T, class IdxT>
std::vector<T> d2h_matrix(raft::device_resources const& res,
                          raft::device_matrix<T, IdxT> const& m) {
  const size_t n = static_cast<size_t>(m.extent(0)) * static_cast<size_t>(m.extent(1));
  std::vector<T> h(n);
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(h.data(), m.data_handle(), n, stream);
  raft::resource::sync_stream(res);
  return h;
}

template <class T>
raft::device_vector<T, int64_t> h2d(raft::device_resources const& res,
                                    std::vector<T> const& h) {
  auto v = raft::make_device_vector<T, int64_t>(res, h.size());
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(v.data_handle(), h.data(), h.size(), stream);
  raft::resource::sync_stream(res);
  return v;
}

template <class T, class IdxT>
raft::device_matrix<T, IdxT> h2d_matrix(raft::device_resources const& res,
                                        std::vector<T> const& h,
                                        IdxT rows, IdxT cols) {
  auto m = raft::make_device_matrix<T, IdxT>(res, rows, cols);
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(m.data_handle(), h.data(),
             static_cast<size_t>(rows) * static_cast<size_t>(cols), stream);
  raft::resource::sync_stream(res);
  return m;
}

} // namespace

void serialize(raft::device_resources const& res,
               std::string const& stem,
               index const& idx) {
  cuvs::neighbors::cagra::serialize(res, stem + kCagraSuffix, idx.anchor_cagra_index);

  const std::string meta_path = stem + kMetaSuffix;
  std::ofstream f(meta_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Failed to open meta file for writing: " + meta_path);
  }

  f.write(reinterpret_cast<const char*>(&kMetaMagic),   sizeof(uint32_t));
  f.write(reinterpret_cast<const char*>(&kMetaVersion), sizeof(uint32_t));

  write_count_and_bytes(f, d2h(res, idx.anchor_labels));
  write_count_and_bytes(f, d2h(res, idx.anchor_to_doc_ids));
  write_count_and_bytes(f, d2h(res, idx.anchor_to_doc_ids_offsets));
  write_count_and_bytes(f, d2h(res, idx.doc_offsets));

  const uint8_t has_vpq = idx.vpq.has_value() ? 1 : 0;
  f.write(reinterpret_cast<const char*>(&has_vpq), sizeof(uint8_t));
  if (has_vpq) {
    const auto& v = *idx.vpq;
    const uint32_t pq_bits      = v.pq_bits;
    const uint32_t pq_dim       = v.pq_dim;
    const uint32_t pq_len       = v.pq_len;
    const uint32_t n_pq_centers = static_cast<uint32_t>(v.pq_codebook.extent(0));
    const uint32_t n_tokens     = static_cast<uint32_t>(v.codes.extent(0));
    const uint32_t qdim         = static_cast<uint32_t>(v.codes.extent(1));

    f.write(reinterpret_cast<const char*>(&pq_bits),      sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&pq_dim),       sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&pq_len),       sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&n_pq_centers), sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&n_tokens),     sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&qdim),         sizeof(uint32_t));

    auto h_pq_codebook = d2h_matrix(res, v.pq_codebook);
    auto h_codes       = d2h_matrix(res, v.codes);
    f.write(reinterpret_cast<const char*>(h_pq_codebook.data()),
            static_cast<std::streamsize>(h_pq_codebook.size()) * sizeof(float));
    f.write(reinterpret_cast<const char*>(h_codes.data()),
            static_cast<std::streamsize>(h_codes.size()) * sizeof(uint8_t));
  }
}

index deserialize(raft::device_resources const& res, std::string const& stem) {
  auto anchor_cagra_index = cuvs::neighbors::cagra::index<half, uint32_t>(res);
  cuvs::neighbors::cagra::deserialize(res, stem + kCagraSuffix, &anchor_cagra_index);

  const std::string meta_path = stem + kMetaSuffix;
  std::ifstream f(meta_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Failed to open meta file: " + meta_path);
  }

  uint32_t magic = 0, version = 0;
  if (!f.read(reinterpret_cast<char*>(&magic),   sizeof(uint32_t)) ||
      !f.read(reinterpret_cast<char*>(&version), sizeof(uint32_t)) ||
      magic != kMetaMagic) {
    throw std::runtime_error(
      "vecflow-chamfer meta file '" + meta_path + "' is missing the expected magic header "
      "(stale or legacy cache); rebuild the index to upgrade to v" +
      std::to_string(kMetaVersion));
  }
  if (version != kMetaVersion) {
    throw std::runtime_error(
      "vecflow-chamfer meta file version mismatch: got " + std::to_string(version) +
      ", expected " + std::to_string(kMetaVersion));
  }

  std::vector<uint32_t> h_anchor_labels;
  std::vector<uint32_t> h_anchor_to_doc_ids;
  std::vector<uint32_t> h_anchor_to_doc_ids_offsets;
  std::vector<uint32_t> h_doc_offsets;
  read_count_and_bytes(f, h_anchor_labels);
  read_count_and_bytes(f, h_anchor_to_doc_ids);
  read_count_and_bytes(f, h_anchor_to_doc_ids_offsets);
  read_count_and_bytes(f, h_doc_offsets);

  std::optional<vpq_data> vpq;
  uint8_t has_vpq = 0;
  if (!f.read(reinterpret_cast<char*>(&has_vpq), sizeof(uint8_t))) {
    throw std::runtime_error("vecflow-chamfer meta file truncated (missing VPQ flag)");
  }
  if (has_vpq) {
    uint32_t pq_bits = 0, pq_dim = 0, pq_len = 0, n_pq_centers = 0;
    uint32_t n_tokens = 0, qdim = 0;
    if (!f.read(reinterpret_cast<char*>(&pq_bits),      sizeof(uint32_t)) ||
        !f.read(reinterpret_cast<char*>(&pq_dim),       sizeof(uint32_t)) ||
        !f.read(reinterpret_cast<char*>(&pq_len),       sizeof(uint32_t)) ||
        !f.read(reinterpret_cast<char*>(&n_pq_centers), sizeof(uint32_t)) ||
        !f.read(reinterpret_cast<char*>(&n_tokens),     sizeof(uint32_t)) ||
        !f.read(reinterpret_cast<char*>(&qdim),         sizeof(uint32_t))) {
      throw std::runtime_error("vecflow-chamfer meta file truncated (VPQ header)");
    }
    std::vector<float>   h_pq_codebook(static_cast<size_t>(n_pq_centers) * pq_len);
    std::vector<uint8_t> h_codes(static_cast<size_t>(n_tokens) * qdim);
    if (!f.read(reinterpret_cast<char*>(h_pq_codebook.data()),
                static_cast<std::streamsize>(h_pq_codebook.size()) * sizeof(float)) ||
        !f.read(reinterpret_cast<char*>(h_codes.data()),
                static_cast<std::streamsize>(h_codes.size()) * sizeof(uint8_t))) {
      throw std::runtime_error("vecflow-chamfer meta file truncated (VPQ payload)");
    }

    vpq = vpq_data{
      h2d_matrix<float,   uint32_t>(res, h_pq_codebook, n_pq_centers, pq_len),
      h2d_matrix<uint8_t, int64_t>(res, h_codes, n_tokens, qdim),
      pq_bits, pq_dim, pq_len,
    };
  }

  return index{
    std::move(anchor_cagra_index),
    h2d(res, h_anchor_labels),
    h2d(res, h_anchor_to_doc_ids),
    h2d(res, h_anchor_to_doc_ids_offsets),
    h2d(res, h_doc_offsets),
    std::move(vpq),
  };
}

} // namespace vecflow_chamfer
