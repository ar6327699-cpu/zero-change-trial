"""Geometry Nodes modifier inputs across Blender versions.

Blender 5.2 removed ID-property access (mod[identifier]) for Nodes modifiers;
inputs live in mod.properties.inputs as RNA structs with a .value field.
Never assign inputs[key]=... there: that replaces the socket's property group
(value/type/attribute_name) with a bare array and the UI freezes on redraw.
Older builds still use mod[identifier].
"""

def _inputs(mod):
    return getattr(getattr(mod,'properties',None),'inputs',None)

def get(mod,key,default=None):
    inputs=_inputs(mod)
    if inputs is None:return mod.get(key,default)
    socket=getattr(inputs,key,None)
    if socket is None:return default
    try:return socket.value
    except AttributeError:return default

def set(mod,key,value):
    inputs=_inputs(mod)
    if inputs is None:mod[key]=value;return
    socket=getattr(inputs,key,None)
    if socket is None:raise KeyError(key)
    socket.value=value
