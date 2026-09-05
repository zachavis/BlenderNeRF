import bpy


class CARR_UI(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_carr_ui'
    bl_label = 'Camera Array CArr'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderNeRF'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.use_property_split = True
        layout.prop(scene, 'carr_geometry')
        layout.prop(scene, 'carr_camera_count')
        layout.prop(scene, 'carr_location')
        layout.prop(scene, 'carr_look_at')
        layout.prop(scene, 'carr_rotation')
        layout.prop(scene, 'carr_radius')
        layout.prop(scene, 'carr_focal')
        layout.use_property_split = False
        layout.separator()
        layout.label(text='Splits')
        row = layout.row(align=True)
        row.prop(scene, 'train_data', toggle=True)
        row.prop(scene, 'test_data', toggle=True)
        layout.separator()
        layout.label(text='Preview')
        layout.prop(scene, 'carr_show_rig', toggle=True)
        layout.separator()
        layout.use_property_split = True
        layout.prop(scene, 'carr_dataset_name')
        layout.separator()
        layout.operator('object.camera_array', text='PLAY CArr')
