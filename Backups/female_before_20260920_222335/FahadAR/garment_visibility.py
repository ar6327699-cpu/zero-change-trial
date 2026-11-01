"""Scene-owned garment lifecycle; never hide against an unrelated view layer."""
import bpy

def studio_scene():
    scenes=[s for s in bpy.data.scenes if s.get('GS_studio_version')]
    if len(scenes)!=1:raise ValueError('Cannot identify the Studio scene. Reopen using START_GARMENT_STUDIO.cmd.')
    return scenes[0]

def hide_in_layer(obj,hidden,layer=None):
    layer=layer or bpy.context.view_layer
    layer.update()
    if layer.objects.get(obj.name) != obj:
        print('GARMENT_VISIBILITY: %s is outside view layer %s; per-layer hide skipped.'%(obj.name,layer.name))
        return False
    obj.hide_set(hidden,view_layer=layer)
    return True

def store(obj,collection):
    if collection.objects.get(obj.name)!=obj:collection.objects.link(obj)
    # Stored assets are never scene actors. These flags don't require a Base
    # in any particular view layer; exclude/monitor flags are left untouched.
    obj.hide_render=True;obj.hide_viewport=True

def show_preview(obj,scene):
    if scene.collection.objects.get(obj.name)!=obj:scene.collection.objects.link(obj)
    obj.hide_viewport=False;obj.hide_render=False
    layer=scene.view_layers[0];layer.update()
    if not hide_in_layer(obj,False,layer):
        raise ValueError('Preview object is linked but unavailable in preview view layer. Reopen the preview scene.')
