obstacle1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[5.0,0.0,1.5], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(2.5,2.5,3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
        ),
    )
obstacle2 = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle2",
    init_state=AssetBaseCfg.InitialStateCfg(pos=[-5.0,-3.0,1.0], rot=[1,0,0,0]),
    spawn=sim_utils.MeshCuboidCfg(
        size=(1.5,4.0,2.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2,0.8,0.2)),
    ),
)
obstacle3 = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle3",
    init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0,5.0,2.0], rot=[1,0,0,0]),
    spawn=sim_utils.MeshCuboidCfg(
        size=(3.0,1.0,4.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2,0.2,0.8)),
    ),
)