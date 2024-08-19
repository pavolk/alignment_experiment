#%%

import open3d as o3d
import numpy as np

#%%

mesh = o3d.io.read_triangle_model("legoplatte.obj")

#%%

meshes = mesh.meshes
materials = mesh.materials
triangle_mesh = [m.mesh for m in meshes]
mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 100, origin=[0, 0, 0])
triangle_mesh.append(mesh_coord_frame)

o3d.visualization.draw_geometries(triangle_mesh, mesh_show_wireframe=True)

#%%

o3d.visualization.draw_geometries([model, aabox])

mesh_pcd = triangle_mesh[0].sample_points_uniformly(number_of_points = 10000)
o3d.visualization.draw_geometries([mesh_pcd])

#%%

aabox = mesh_pcd.get_axis_aligned_bounding_box()
c

# %%

pcd = o3d.io.read_point_cloud('D415_roi.ply')
#o3d.visualization.draw_geometries([pcd])

#%%

voxel_size = 0.0025
downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
#o3d.visualization.draw_geometries([downpcd])

#%%

radius = voxel_size * 5
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

# %%

aabox = downpcd.get_axis_aligned_bounding_box()
aabox.color = (1, 0, 0)

obox = downpcd.get_oriented_bounding_box()
obox.color = (0, 1, 0)

o3d.visualization.draw_geometries([downpcd, aabox, obox])

#%%


