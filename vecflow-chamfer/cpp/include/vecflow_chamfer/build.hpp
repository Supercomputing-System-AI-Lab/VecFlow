#pragma once

#include <raft/core/device_resources.hpp>

#include <vecflow_chamfer/types.hpp>

namespace vecflow_chamfer {

// Build a MaxIVF index over `ds`. Pure construction — no on-disk caching. Use
// vecflow_chamfer::serialize / deserialize (serialize.hpp) to persist.
index build(raft::device_resources const& res,
            dataset const& ds,
            index_params const& params);

} // namespace vecflow_chamfer
