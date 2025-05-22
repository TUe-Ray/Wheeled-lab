from omni.isaac.core.utils.prims import get_all_matching_child_prims
prims = get_all_matching_child_prims("/World/envs/env_0/Robot")
print([p.GetPath() for p in prims])