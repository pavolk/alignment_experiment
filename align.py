#%%

import open3d as o3d
import numpy as np
import copy

#%%

model = o3d.io.read_triangle_model("legoplatte.obj")
meshes = model.meshes
mesh = meshes[0].mesh

center = mesh.get_center()
mesh.scale(0.0004, center) # making metric (units: meter)
mesh.translate([0,0,0], relative=False)

#%%

mesh_pcd = mesh.sample_points_uniformly(number_of_points = 10000)
mesh_aabox = mesh_pcd.get_axis_aligned_bounding_box()
mesh_aabox.color = (1, 0, 0)
#o3d.visualization.draw_geometries([mesh_pcd, mesh_aabox])

# %%

pcd = o3d.io.read_point_cloud('D415_roi.ply')
pcd.translate([0, 0, 0], relative = False)
voxel_size = 0.0025
downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

#%%

downpcd_obox = downpcd.get_oriented_bounding_box()
downpcd_aabox = downpcd.get_axis_aligned_bounding_box()

#%%

R = downpcd_obox.R
R_inv = downpcd_obox.R.T

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
rotated_origin = copy.deepcopy(origin)
rotated_origin.rotate(R_inv, center=downpcd_obox.get_center())
rotated_origin.paint_uniform_color([0, 1, 1])

o3d.visualization.draw_geometries([origin, rotated_origin],  mesh_show_wireframe=True)


#%%

pcd = copy.deepcopy(downpcd)
pcd = pcd.rotate(R, downpcd.get_center())
pcd.paint_uniform_color([1, 0, 0])

pcd_inv = copy.deepcopy(downpcd)
pcd_inv = pcd_inv.rotate(R_inv, downpcd.get_center())
pcd_inv.paint_uniform_color([0, 1, 0])

#downpcd_aabox.color = [0, 0, 1]
#o3d.visualization.draw_geometries([downpcd, downpcd_aabox, pcd_inv])

#%%

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([downpcd, pcd, pcd_inv, mesh, coord_frame],  mesh_show_wireframe=True)

# %%
