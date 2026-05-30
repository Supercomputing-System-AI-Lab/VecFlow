#pragma once

#include <string>

#include <raft/core/device_resources.hpp>

#include <vecflow_chamfer/types.hpp>

namespace vecflow_chamfer {

// Persist `idx` to disk. Writes two files alongside one another, sharing the
// same `stem`:
//   - `<stem>.maxivf`  — the underlying CAGRA index over anchor centroids
//                        (cuVS native format)
//   - `<stem>.meta`    — vecflow-chamfer sidecar (anchor labels, inverted
//                        lists, doc_offsets)
//
// Both files must be readable together for deserialize() to succeed.
void serialize(raft::device_resources const& res,
               std::string const& stem,
               index const& idx);

// Load an index previously written by serialize(). Reads both `<stem>.maxivf`
// and `<stem>.meta`. Throws std::runtime_error if either file is missing or
// the sidecar is malformed; lets cuVS exceptions propagate if the CAGRA file
// is incompatible.
index deserialize(raft::device_resources const& res,
                  std::string const& stem);

} // namespace vecflow_chamfer
