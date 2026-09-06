import json

import bpy


scene = bpy.context.scene
assert scene.render.engine == 'BLENDER_WORKBENCH', 'Fast cartoon renderer must be configured'
assert scene.frame_end == scene.render.fps * 16, 'Opening must last 16 seconds'
assert len(scene.timeline_markers) == 4, 'Opening must contain four camera cuts'
assert len(json.loads(scene['cast'])) == 4, 'All four characters must be present'
assert len([obj for obj in scene.objects if obj.name.startswith('Drifting snow')]) == 65
for index, marker in enumerate(scene.timeline_markers):
    scene.frame_set(marker.frame)
    assert scene.camera == marker.camera, f'Camera cut {index + 1} is not active'
    assert marker.frame == index * 4 * scene.render.fps + 1
scene.frame_set(1)
bus_start = bpy.data.objects['School bus'].location.x
cast_start = bpy.data.objects['Pip'].location.z
scene.frame_set(1 + scene.render.fps // 4)
assert bpy.data.objects['Pip'].location.z != cast_start, 'Characters must bounce to the beat'
scene.frame_set(8 * scene.render.fps)
assert bpy.data.objects['School bus'].location.x > bus_start + 20, 'Bus must cross the town'
scene.frame_set(12 * scene.render.fps)
assert bpy.data.objects['Main title'].scale.x == 0, 'Title must stay hidden before the last shot'
scene.frame_set(12 * scene.render.fps + 1)
assert bpy.data.objects['Main title'].scale.x == 1, 'Final shot must display the title'
assert all(image.packed_file for image in bpy.data.images if image.source == 'FILE'), 'Scene must not depend on external textures'
assert all(font.packed_file for font in bpy.data.fonts if font.filepath and font.filepath != '<builtin>'), 'External fonts must be packed'
print('PASS: duration, four cuts, cast, snow, bus travel, character motion and title timing')
