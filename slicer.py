import trimesh
import numpy as np
from PIL import Image, ImageDraw
import os
from tqdm import tqdm
import shutil
import plotly.graph_objects as go
import plotly.express as px


class VibeCodedSlicer:
    def __init__(self, file_path, output_dir='output'):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        self.mesh = trimesh.load(file_path)
        self.output_dir = output_dir
        self.volume_dims = [100, 100, 100]  # Default volume dimensions

    def place_and_scale_model_in_volume(self, scale_factor):
        """
        Scales the mesh, places its lowest point (min Z) at 0,
        and centers it in X and Y around the origin (0,0).
        Then, updates volume_dims based on the mesh's new bounds with padding.
        """
        # Apply the initial scale factor
        self.mesh.apply_scale(scale_factor)

        # Get the current bounds of the mesh after scaling
        bounds = self.mesh.bounds
        extents = bounds[1] - bounds[0]

        # Calculate translation to move the lowest point (min Z) to 0
        # and center in X and Y around the origin (0,0)
        translation = np.array([
            -bounds[0][0] - extents[0] / 2,  # Center X around 0
            -bounds[0][1] - extents[1] / 2,  # Center Y around 0
            -bounds[0][2]                   # Move lowest Z to 0
        ])
        self.mesh.apply_translation(translation)

        # After translation, recalculate bounds to get the new position
        new_bounds = self.mesh.bounds
        new_extents = new_bounds[1] - new_bounds[0]

        # Set volume_dims based on the mesh's new extents with some padding
        # This ensures the volume always encompasses the mesh
        padding = 10 # A small padding around the mesh
        self.volume_dims = [
            new_extents[0] + padding,
            new_extents[1] + padding,
            new_extents[2] + padding
        ]
        # Ensure volume_dims are at least a certain size if the mesh is tiny
        min_dim_size = 100
        self.volume_dims[0] = max(self.volume_dims[0], min_dim_size)
        self.volume_dims[1] = max(self.volume_dims[1], min_dim_size)
        self.volume_dims[2] = max(self.volume_dims[2], min_dim_size)

        print(f"Updated volume dimensions to: {self.volume_dims}")

    def set_volume_dimensions(self, dims):
        self.volume_dims = dims

    def get_rotated_normal(self, plane='xy', angle_deg=0.0, rotation_axis='X'):
        """
        Calculates the normal vector for a plane rotated by a specific angle.
        """
        # Define the base normal vectors for each plane
        base_normals = {
            'xy': [0, 0, 1],
            'yz': [1, 0, 0],
            'xz': [0, 1, 0]
        }
        
        # Define the rotation axes
        rotation_axes = {
            'X': [1, 0, 0],
            'Y': [0, 1, 0],
            'Z': [0, 0, 1]
        }

        if plane not in base_normals:
            raise ValueError("Invalid plane specified. Choose from 'xy', 'yz', 'xz'.")
        if rotation_axis not in rotation_axes:
            raise ValueError("Invalid rotation axis specified. Choose from 'X', 'Y', 'Z'.")

        # Get the base normal vector
        base_normal = base_normals[plane]
        
        # If angle is 0, no rotation is needed
        if angle_deg == 0.0:
            return np.array(base_normal)

        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)

        # Create the rotation matrix
        rotation_vector = rotation_axes[rotation_axis]
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, rotation_vector)

        # Apply the rotation to the base normal vector
        rotated_normal = (rotation_matrix @ np.append(base_normal, 1))[:3]
        
        return rotated_normal

    

    

    def plot_3d_plotly(self, slice_normal, all_slice_meshes, all_full_plane_meshes):
        """
        Visualizes the 3D model, volume, actual slicing cross-sections, and full slicing planes using Plotly.
        """
        data = []

        # Add mesh trace
        mesh_trace = go.Mesh3d(
            x=self.mesh.vertices[:, 0],
            y=self.mesh.vertices[:, 1],
            z=self.mesh.vertices[:, 2],
            i=self.mesh.faces[:, 0],
            j=self.mesh.faces[:, 1],
            k=self.mesh.faces[:, 2],
            color='cyan',
            opacity=0.5,
            name='Model'
        )
        data.append(mesh_trace)

        # Add volume bounding box trace
        volume_box = trimesh.creation.box(extents=self.volume_dims)
        volume_box_center_translation = np.array([0, 0, self.volume_dims[2] / 2])
        volume_box.apply_translation(volume_box_center_translation)

        box_vertices = volume_box.vertices
        box_faces = volume_box.faces

        box_trace = go.Mesh3d(
            x=box_vertices[:, 0],
            y=box_vertices[:, 1],
            z=box_vertices[:, 2],
            i=box_faces[:, 0],
            j=box_faces[:, 1],
            k=box_faces[:, 2],
            color='blue',
            opacity=0.1,
            name='Volume Box'
        )
        data.append(box_trace)

        # Add actual slicing cross-sections traces
        for i, slice_mesh in enumerate(all_slice_meshes):
            if slice_mesh is not None:
                slice_trace = go.Mesh3d(
                    x=slice_mesh.vertices[:, 0],
                    y=slice_mesh.vertices[:, 1],
                    z=slice_mesh.vertices[:, 2],
                    i=slice_mesh.faces[:, 0],
                    j=slice_mesh.faces[:, 1],
                    k=slice_mesh.faces[:, 2],
                    color='red',
                    opacity=0.8,
                    name=f'Slice {i}'
                )
                data.append(slice_trace)

        

        fig = go.Figure(data=data)
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data' # Keep aspect ratio for 3D plot
            ),
            title='VibeCodedSlicer 3D Visualization'
        )
        # Save the figure as an HTML file
        output_html_path = os.path.join(self.output_dir, 'slicer_3d_visualization.html')
        fig.write_html(output_html_path)
        print(f"3D visualization saved to {output_html_path}")

    def slice(self, slice_normal=[0, 0, 1], num_slices=100, debug=False, visualize=False):
        """
        Slices the mesh using a specified normal vector.
        """
        slice_normal = np.array(slice_normal) / np.linalg.norm(slice_normal)
        
        # Determine the slicing range by projecting vertices onto the normal
        # Use the mesh's bounding box to determine the min/max extent along the slice normal
        # This ensures the slicing planes cover the entire mesh.
        mesh_extents_along_normal = np.dot(self.mesh.bounds, slice_normal)
        min_proj = np.min(mesh_extents_along_normal)
        max_proj = np.max(mesh_extents_along_normal)
        slice_origins = np.linspace(min_proj, max_proj, num_slices)

        # Calculate z_heights for visualization based on mesh bounds
        min_z_mesh = self.mesh.bounds[0][2]
        max_z_mesh = self.mesh.bounds[1][2]
        z_heights_for_viz = np.linspace(min_z_mesh, max_z_mesh, num_slices)

        print(f"Slicing along normal: {slice_normal} with {num_slices} slices.")
        if debug:
            if not os.path.exists('debug_output'):
                os.makedirs('debug_output')
            print("Debug mode enabled. Saving slice coordinates to 'debug_output'")

        # Collect 3D representations of 2D slices for visualization
        all_slice_meshes = []
        all_full_plane_meshes = [] # To store meshes for full slicing planes

        for i, origin_dist in enumerate(tqdm(slice_origins)):
            # Add a small random perturbation to the origin to mitigate floating point issues
            perturbation = (np.random.rand(3) - 0.5) * 1e-6 # small random offset
            plane_origin = origin_dist * slice_normal + perturbation

            # Get the z_origin_height for this specific slice
            # This is the Z-coordinate of the plane_origin
            z_origin_height = plane_origin[2]
            
            # Create and store the full slicing plane mesh for visualization
            # Calculate plane_size based on the diagonal of the volume's XY dimensions
            plane_size = np.sqrt(self.volume_dims[0]**2 + self.volume_dims[1]**2) * 1.2 # Slightly larger
            full_plane_mesh = trimesh.creation.box(extents=[plane_size, plane_size, 0.001]) # Very thin

            # Create a transformation matrix that maps the XY plane to the slicing plane
            # The slicing plane is defined by plane_origin and slice_normal
            # Calculate the rotation matrix to align the Z-axis with slice_normal
            # The default normal for trimesh.creation.box is [0,0,1] (along Z)
            current_normal = np.array([0, 0, 1])
            axis = np.cross(current_normal, slice_normal)
            
            # Handle the case where current_normal and slice_normal are colinear
            if np.linalg.norm(axis) == 0:
                # If vectors are colinear, no rotation is needed if they are in the same direction
                # If they are in opposite directions, rotate 180 degrees around an arbitrary axis (e.g., X-axis)
                if np.dot(current_normal, slice_normal) < 0:
                    axis = np.array([1, 0, 0]) # Arbitrary axis for 180 degree rotation
                    angle = np.pi
                else:
                    angle = 0 # No rotation needed
            else:
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(current_normal, slice_normal))

            
            
            # Get the 2D cross-section
            section = self.mesh.section(plane_origin=plane_origin, plane_normal=slice_normal)
            
            if section is None or len(section.vertices) < 3:
                # If the section is empty or just a line, save a blank image
                img = Image.new('L', (int(self.volume_dims[0]), int(self.volume_dims[1])), 0)
                img.save(os.path.join(self.output_dir, f'slice_{i:04d}.png'))
                continue

            # Use to_2D to get a Path2D object, which is more robust.
            path_2D, to_3D_transform = section.to_2D()

            # Convert the 2D path to a 3D mesh for visualization
            # We need to extrude the 2D path slightly to make it visible in 3D
            # and then transform it back to its original 3D plane.
            try:
                # Extrude the 2D path into a thin 3D mesh
                extruded_result = path_2D.extrude(height=0.01)
                
                # Ensure extruded_result is always a list of meshes
                if isinstance(extruded_result, list):
                    extruded_meshes = extruded_result
                else:
                    extruded_meshes = [extruded_result]

                for extruded_mesh in extruded_meshes:
                    # Apply the original 3D transform to the extruded mesh
                    # This places the extruded slice in its correct 3D orientation and position
                    extruded_mesh.apply_transform(to_3D_transform)

                    all_slice_meshes.append(extruded_mesh)
            except Exception as e:
                print(f"Warning: Could not process 2D path for slice {i}: {e}")
                # If processing fails, we still want to save the 2D image
                img = Image.new('L', (int(self.volume_dims[0]), int(self.volume_dims[1])), 0)
                img.save(os.path.join(self.output_dir, f'slice_{i:04d}.png'))
                continue

            if debug:
                with open(f"debug_output/slice_{i:04d}.txt", "w") as f:
                    for polygon in path_2D.polygons_full:
                        f.write(f"Exterior:\n")
                        for v in polygon.exterior.coords:
                            f.write(f"{v[0]}, {v[1]}\n")
                        f.write(f"Holes:\n")
                        for hole in polygon.interiors:
                            for v in hole.coords:
                                f.write(f"{v[0]}, {v[1]}\n")


            # Create a black image with dimensions based on volume_dims
            # The image dimensions should correspond to the X and Y dimensions of the volume
            img_width = int(self.volume_dims[0])
            img_height = int(self.volume_dims[1])
            img = Image.new('L', (int(img_width), int(img_height)), 0)
            draw = ImageDraw.Draw(img)

            # Calculate offset to center the mesh's (0,0) in the image
            # The mesh is centered around (0,0) in XY, so the image center should be (0,0)
            offset_x = img_width / 2
            offset_y = img_height / 2

            # A Path2D object has `polygons_full` which gives the exterior and holes.
            for polygon in path_2D.polygons_full:
                # The first polygon is the exterior, subsequent ones are holes.
                exterior = polygon.exterior.coords
                holes = [hole.coords for hole in polygon.interiors]
                
                # Apply offset and draw the exterior polygon (filled)
                transformed_exterior = [(v[0] + offset_x, v[1] + offset_y) for v in exterior]
                draw.polygon([tuple(v) for v in transformed_exterior], fill=255, outline=255)
                
                # Apply offset and draw the holes (filled with black)
                for hole in holes:
                    transformed_hole = [(v[0] + offset_x, v[1] + offset_y) for v in hole]
                    draw.polygon([tuple(v) for v in transformed_hole], fill=0, outline=0)

            img.save(os.path.join(self.output_dir, f'slice_{i:04d}.png'))
        
        print(f"Finished slicing. Images saved in '{self.output_dir}'.")

        if visualize:
            self.plot_3d_plotly(slice_normal, all_slice_meshes, all_full_plane_meshes)


if __name__ == '__main__':    
    # 1. Initialize the slicer with the STL file
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, 'input_stl', 'Eiffel.stl')
    slicer = VibeCodedSlicer(model_path, output_dir='output_angled') 
    
    # 2. Rescale the model to fit and center it at the origin
    slicer.place_and_scale_model_in_volume(20) #variable scales by factor of x
    
    # 4. Calculate the slicing normal based on a plane and an angle
    # Here, we take the 'xy' plane and rotate it 30 degrees around the 'Y' axis.
    angled_normal = slicer.get_rotated_normal(plane='xy', angle_deg=30.0, rotation_axis='Y')
    
    # 5. Perform the slicing with the calculated normal
    slicer.slice(slice_normal=angled_normal, num_slices=40, debug=True, visualize=True)