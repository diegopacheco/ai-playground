import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=24)
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


args = arguments()
random.seed(27)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.resolution_x = args.width
scene.render.resolution_y = args.height
scene.render.resolution_percentage = 100
scene.render.fps = args.fps
scene.frame_start = 1
scene.frame_end = 16 * args.fps
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.filepath = '//frames/'
scene.display.render_aa = '16'
shading = scene.display.shading
shading.light = 'STUDIO'
shading.studio_light = 'paint.sl'
shading.studiolight_rotate_z = 0.4
shading.color_type = 'MATERIAL'
shading.show_shadows = True
shading.show_cavity = True
shading.cavity_type = 'BOTH'
shading.curvature_ridge_factor = 0.65
shading.curvature_valley_factor = 0.65
shading.show_specular_highlight = False
shading.show_object_outline = True
shading.object_outline_color = (0.12, 0.17, 0.21)
shading.background_type = 'WORLD'
scene.world.color = (0.39, 0.68, 0.79)
scene.view_settings.view_transform = 'Standard'


def material(name, color):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    return result


palette = {
    'snow': (0.9, 0.96, 0.98), 'ink': (0.035, 0.06, 0.075),
    'pine': (0.045, 0.26, 0.2), 'wood': (0.29, 0.14, 0.09),
    'teal': (0.05, 0.46, 0.47), 'coral': (0.83, 0.22, 0.14),
    'mustard': (0.97, 0.62, 0.07), 'purple': (0.39, 0.23, 0.55),
    'skin': (0.95, 0.66, 0.42), 'cream': (1.0, 0.91, 0.69),
    'glass': (0.29, 0.65, 0.75), 'road': (0.21, 0.29, 0.34),
    'mountain': (0.24, 0.43, 0.51), 'distant': (0.38, 0.56, 0.63),
}
mats = {name: material(name, color) for name, color in palette.items()}


def finish(obj, name, color, parent=None):
    obj.name = name
    obj.data.materials.append(mats[color])
    if parent:
        obj.parent = parent
    return obj


def box(name, position, size, color, parent=None, bevel=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = finish(bpy.context.object, name, color, parent)
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        modifier = obj.modifiers.new('Soft corners', 'BEVEL')
        modifier.width = bevel
        modifier.segments = 2
        obj.modifiers.new('Weighted normals', 'WEIGHTED_NORMAL')
    return obj


def ball(name, position, size, color, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=1, location=position)
    obj = finish(bpy.context.object, name, color, parent)
    obj.scale = size
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def cone(name, position, radius, depth, color, parent=None, top=0):
    bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=radius, radius2=top, depth=depth, location=position)
    return finish(bpy.context.object, name, color, parent)


def group(name, location):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    obj.location = location
    return obj


font_path = Path('/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf')
font = bpy.data.fonts.load(str(font_path)) if font_path.exists() else None


def lettering(name, body, position, size, color, parent=None):
    curve = bpy.data.curves.new(name, 'FONT')
    curve.body = body
    curve.align_x = 'CENTER'
    curve.align_y = 'CENTER'
    curve.size = size
    curve.extrude = 0.008
    curve.bevel_depth = 0.003
    if font:
        curve.font = font
    obj = bpy.data.objects.new(name, curve)
    scene.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (math.pi / 2, 0, 0)
    return finish(obj, name, color, parent)


def key(obj, frame, location=None, rotation=None):
    if location is not None:
        obj.location = location
        obj.keyframe_insert(data_path='location', frame=frame)
    if rotation is not None:
        obj.rotation_euler = rotation
        obj.keyframe_insert(data_path='rotation_euler', frame=frame)


box('Snowfield', (0, 4, -0.4), (95, 70, 0.7), 'snow')
box('Main street', (0, 0, 0.01), (65, 3.8, 0.08), 'road')
for x in range(-30, 31, 3):
    box('Road stripe', (x, 0, 0.065), (1.3, 0.1, 0.015), 'cream')
for x, y, radius, height in [(-20, 24, 10, 16), (-9, 26, 11, 19), (4, 27, 12, 18), (19, 24, 11, 17)]:
    cone('Distant mountain', (x, y, height / 2 - 1), radius, height, 'distant')
    cone('Snow peak', (x, y, height * 0.79 - 1), radius * 0.42, height * 0.42, 'snow')
for x in [-18, -5, 12, 25]:
    cone('Mountain ridge', (x, 19, 4.5), 7, 11, 'mountain')
    cone('Ridge snowcap', (x, 19, 8.3), 2.16, 3.4, 'snow')


def pine(x, y, scale):
    cone('Pine trunk', (x, y, 0.7 * scale), 0.16 * scale, 1.4 * scale, 'wood', top=0.16 * scale)
    for z, radius in [(1.5, 1), (2.25, 0.8), (2.9, 0.57)]:
        cone('Evergreen tier', (x, y, z * scale), radius * scale, 1.6 * scale, 'pine')
        cone('Snow on pine', (x, y, (z + 0.33) * scale), radius * 0.64 * scale, scale, 'snow')


for index in range(32):
    pine(random.uniform(-29, 29), random.uniform(10, 16), random.uniform(0.8, 1.65))
for x, y in [(-12, -3), (12, -2), (-17, 4), (18, 5)]:
    pine(x, y, 1.3)


def building(x, width, height, color, label):
    box(label, (x, 6, height / 2), (width, 3.5, height), color, bevel=0.08)
    for side in [-1, 1]:
        roof = box('Snow roof', (x + side * width / 4, 6, height + 0.5), (width * 0.58, 4, 0.2), 'snow')
        roof.rotation_euler.y = side * 0.42
    box('Front door', (x, 4.21, 0.92), (0.78, 0.12, 1.8), 'wood', bevel=0.04)
    ball('Door knob', (x + 0.23, 4.11, 0.9), (0.06, 0.04, 0.06), 'mustard')
    for side in [-1, 1]:
        wx = x + side * width * 0.3
        box('Window frame', (wx, 4.18, 1.6), (0.95, 0.15, 1.2), 'cream')
        box('Window pane', (wx, 4.08, 1.6), (0.76, 0.03, 1.02), 'glass')
        box('Window mullion', (wx, 4.04, 1.6), (0.06, 0.04, 1.05), 'cream')
        box('Window crossbar', (wx, 4.04, 1.6), (0.78, 0.04, 0.06), 'cream')
    box('Shop sign', (x, 4.02, height - 0.5), (width * 0.92, 0.18, 0.61), 'cream', bevel=0.04)
    lettering(label + ' lettering', label, (x, 3.91, height - 0.5), 0.34, 'ink')
    box('Chimney', (x + width * 0.28, 6.4, height + 0.95), (0.5, 0.55, 1.2), 'wood')


for spec in [(-10, 4, 3.7, 'teal', 'PINE MART'), (-5, 4, 4.3, 'coral', 'HOT COCOA'), (0, 4, 3.7, 'mustard', 'TOWN HALL'), (5, 4, 4.3, 'purple', 'POST OFFICE'), (10, 4, 3.6, 'teal', 'SKI REPAIR')]:
    building(*spec)

for x in [-7.6, 7.6]:
    cone('Street lamp', (x, 2.6, 1.6), 0.075, 3.2, 'ink', top=0.075)
    box('Lamp glass', (x, 2.6, 3.25), (0.38, 0.38, 0.5), 'cream', bevel=0.04)
    cone('Lamp cap', (x, 2.6, 3.65), 0.37, 0.3, 'ink')

for x in [-8.6, -5.4]:
    box('Welcome post', (x, -2.3, 1.2), (0.16, 0.2, 2.4), 'wood')
box('Welcome board', (-7, -2.3, 2.1), (4.3, 0.25, 1.5), 'pine', bevel=0.06)
box('Welcome snow', (-7, -2.3, 2.9), (4.5, 0.32, 0.14), 'snow', bevel=0.05)
lettering('Welcome', 'WELCOME TO', (-7, -2.46, 2.48), 0.26, 'cream')
lettering('Town name', 'FROSTBITE FALLS', (-7, -2.46, 2.05), 0.34, 'snow')
lettering('Population', 'POP. 408  /  MOSTLY NICE', (-7, -2.46, 1.65), 0.17, 'cream')


def character(name, x, coat, hat):
    root = group(name, (x, -3.2, 0.15))
    for side in [-1, 1]:
        ball(name + ' boot', (side * 0.26, -0.07, 0.2), (0.29, 0.39, 0.2), 'ink', root)
    ball(name + ' parka', (0, 0, 0.87), (0.69, 0.42, 0.68), coat, root)
    box(name + ' zipper', (0, -0.424, 0.82), (0.035, 0.025, 0.82), 'ink', root)
    ball(name + ' head', (0, -0.03, 1.74), (0.73, 0.43, 0.65), 'skin', root)
    ball(name + ' beanie', (0, 0.01, 2.15), (0.75, 0.46, 0.33), hat, root)
    box(name + ' hat brim', (0, -0.38, 2.06), (1.34, 0.17, 0.18), hat, root, 0.07)
    ball(name + ' pompom', (0, 0, 2.53), (0.18, 0.18, 0.18), 'cream', root)
    for side in [-1, 1]:
        ball(name + ' eye', (side * 0.22, -0.427, 1.78), (0.22, 0.075, 0.27), 'snow', root)
        ball(name + ' pupil', (side * 0.17, -0.501, 1.76), (0.054, 0.035, 0.074), 'ink', root)
    ball(name + ' mouth', (0, -0.435, 1.4), (0.13, 0.035, 0.055), 'ink', root)
    box(name + ' scarf', (0, -0.41, 1.22), (1.03, 0.16, 0.17), hat, root, 0.07)
    box(name + ' scarf tail', (0.39, -0.45, 1.02), (0.17, 0.12, 0.46), hat, root, 0.04)
    for side in [-1, 1]:
        arm = group(name + ' arm', (side * 0.55, 0, 1.08))
        arm.parent = root
        ball(name + ' sleeve', (side * 0.08, 0, -0.2), (0.22, 0.3, 0.37), coat, arm)
        ball(name + ' mitten', (side * 0.12, -0.06, -0.46), (0.22, 0.28, 0.21), hat, arm)
        if side == 1:
            for beat in range(33):
                key(arm, 1 + beat * args.fps / 2, rotation=(0, -0.35 if beat % 2 else -1.6, 0))
    for beat in range(65):
        key(root, 1 + beat * args.fps / 4, (x, -3.2, 0.15 + (0.065 if beat % 2 else 0)))
    return root


cast = [character('Pip', -2.6, 'coral', 'teal'), character('Moss', -0.87, 'teal', 'mustard'), character('June', 0.87, 'purple', 'coral'), character('Otis', 2.6, 'mustard', 'purple')]

bus = group('School bus', (-19, 0, 0))
box('Bus body', (0, 0, 1.2), (5, 1.8, 1.4), 'mustard', bus, 0.16)
box('Bus cabin', (-0.4, 0, 2.07), (4.2, 1.75, 1.1), 'mustard', bus, 0.16)
box('Bus roof snow', (-0.4, 0, 2.64), (4.35, 1.9, 0.13), 'snow', bus, 0.06)
box('Bus bumper', (2.56, 0, 0.74), (0.2, 1.9, 0.18), 'ink', bus)
for x in [-1.9, -1.05, -0.2, 0.65]:
    for side in [-1, 1]:
        box('Bus window frame', (x, side * 0.889, 2.1), (0.75, 0.07, 0.72), 'ink', bus, 0.04)
        box('Bus window', (x, side * 0.932, 2.1), (0.62, 0.025, 0.59), 'glass', bus, 0.02)
for side in [-1, 1]:
    box('Bus stripe', (0, side * 0.915, 1.05), (4.85, 0.03, 0.13), 'ink', bus)
    for x in [-1.6, 1.55]:
        wheel = cone('Bus tire', (x, side * 0.94, 0.5), 0.5, 0.22, 'ink', bus, top=0.5)
        wheel.rotation_euler.x = math.pi / 2
        hub = cone('Wheel hub', (x, side * 1.07, 0.5), 0.22, 0.03, 'cream', bus, top=0.22)
        hub.rotation_euler.x = math.pi / 2
    ball('Headlight', (2.51, side * 0.62, 1.3), (0.07, 0.19, 0.2), 'cream', bus)
lettering('Bus lettering', 'FROSTBITE SCHOOL', (-0.3, -0.94, 1.45), 0.24, 'ink', bus)
for frame, x in [(1, -19), (4 * args.fps, -8), (8 * args.fps, 7), (10 * args.fps, 20), (16 * args.fps, 38)]:
    key(bus, frame, (x, 0, 0))

for index in range(65):
    x, y, z = random.uniform(-15, 15), random.uniform(-5, 12), random.uniform(4, 13)
    flake = ball('Drifting snow', (x, y, z), (0.045, 0.045, 0.045), 'snow')
    key(flake, 1, (x, y, z))
    key(flake, scene.frame_end, (x + 2.7, y, z - 4))


def camera(name, start, end, position, target, scale, finish_position):
    bpy.ops.object.camera_add(location=position)
    obj = bpy.context.object
    obj.name = name
    obj.data.type = 'ORTHO'
    obj.data.ortho_scale = scale
    obj.data.lens = 45
    obj.data.clip_end = 200
    for frame, location in [(start, position), (end, finish_position)]:
        rotation = (Vector(target) - Vector(location)).to_track_quat('-Z', 'Y').to_euler()
        key(obj, frame, location, rotation)
    marker = scene.timeline_markers.new(name, frame=start)
    marker.camera = obj
    return obj


fps = args.fps
camera('01 - Welcome to town', 1, 4 * fps, (15, -26, 16), (0, 4, 3.5), 32, (10, -25, 13))
camera('02 - The morning commute', 4 * fps + 1, 8 * fps, (-7, -16, 6), (0, 0, 1.4), 17, (5, -17, 6))
camera('03 - The usual suspects', 8 * fps + 1, 12 * fps, (-1.4, -16, 4.1), (0, -3.2, 1.35), 9.5, (1.2, -16, 3.6))
title_camera = camera('04 - Frostbite Falls', 12 * fps + 1, 16 * fps, (0, -22, 7), (0, 0, 2.7), 17.5, (0, -23, 7.5))
plate = box('Title panel', (0, 2.65, -12.12), (15.5, 2.2, 0.08), 'pine', title_camera, 0.15)
plate.scale = (0, 0, 0)
plate.keyframe_insert(data_path='scale', frame=12 * fps)
plate.scale = (1, 1, 1)
plate.keyframe_insert(data_path='scale', frame=12 * fps + 1)
for name, body, y, size, color in [('Main title', 'FROSTBITE FALLS', 3.05, 1.32, 'cream'), ('Title kicker', 'A LITTLE TOWN. A LOT OF ATTITUDE.', 1.96, 0.32, 'snow'), ('Title footer', 'WINTER NEVER TAKES A DAY OFF', -4.02, 0.27, 'ink')]:
    obj = lettering(name, body, (0, y, -12), size, color, title_camera)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (0, 0, 0)
    obj.keyframe_insert(data_path='scale', frame=12 * fps)
    obj.scale = (1, 1, 1)
    obj.keyframe_insert(data_path='scale', frame=12 * fps + 1)

scene['project'] = 'Frostbite Falls'
scene['duration_seconds'] = 16
scene['cast'] = json.dumps([obj.name for obj in cast])
scene.frame_set(1)
scene.camera = scene.timeline_markers[0].camera
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
            area.spaces.active.shading.color_type = 'MATERIAL'
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(output))
manifest = {'title': 'Frostbite Falls', 'width': args.width, 'height': args.height, 'fps': fps, 'frames': scene.frame_end, 'duration': 16, 'shots': [{'name': marker.name, 'frame': marker.frame} for marker in scene.timeline_markers]}
output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Saved {output}: {scene.frame_end} frames, four cameras, four characters')
