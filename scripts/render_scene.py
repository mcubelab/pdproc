#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:38:55 2016

@author: Alina Kloss (main author)
@author: Peter KT Yu (minor revision)

A class, Scene, for rendering depth and rgb images for the MIT push
dataset to a simple scene that consists of a flat surface, the object
and optionally a cylinder representing the pusher.

Parameters of the constructor:
    param: a dictionary with following optional arguments
        width: the width of the output images   [default: 640]
        height: the height of the output images [default: 480]

        resource_path: the path to the folder named "resources", which
                       contains resource files such as object meshes and
                       textures

        ground_w: the width of the surface upon which the object is
                  rendered in meters           [default: 0.6]
        ground_d: the depth (height) of the surface upon which the
                  object is rendered in meters [default: 0.5]

        camera_pos: a list of [x,y,z] describing an the camera position
                    in meters [default: [0, 0, 0.6]]

        texture_files: a list of names of texture files to be loaded.
                       If not supplied, all texture files in the
                       resource folder will be loaded [default: []]

        object_files: a list of names of object meshes to be loaded.
                      If not supplied, all meshes in the resource folder
                      will be loaded [default: []]

        shadows: activates shadows in the rgb image [default: True]

Usage:
    After initializing a Scene object, call draw(...) with desired scene
    parameters to generate an RGB-D image.

    You can change the camera location between calls to draws by using
    set_camera().

    Both set_camera() and get_world_2_cam() return the transformation matrix
    for transforming a 3d point in world coordinates to camera coordinates.

    Use close() to destroy the rendering context when you're done using the
    Scene object.

    A demo is implemented in the main function.

Notes:
    Object Meshes (in .obj format), textures and materials must be supplied
    in the resource folder. All files in this folder will be automatically
    loaded on initialization.

    Shadows are still experimental and artifacts may appear, especially for
    extreme camera angle or light position. Contributions are welcome!

    If the shadows look bad for your specific camera setting, try
    experimenting with different ways to compute the "bias" variable in
    ShadowCalculation(). Higher bias will help against "shadow acne"
    (weird artifacts), but might cause "peter-panning" (shadows don't
    connect to their casters).

    The camera will always look at (0, 0, 0), and the scene is rendered
    onto the camera's x-y plane. The z-coordinate thus determines the
    height of the camera relative to the surface and should therefore
    not be smaller than 0.1. Setting non-zero x and y coordinates.

References:
    The OpenGL code in this file is mostly adapted from https://learnopengl.com/
"""

import OpenGL.GL as gl
import OpenGL.GLU as glu

from OpenGL.arrays import vbo
import OpenGL.GLUT as glut
import OpenGL.GL.framebufferobjects as fbos

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from mesh_object import MeshObject
from shader import Shader

import h5py
import argparse

import logging


# setup logging
log = logging.getLogger('contact_annotation')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s: [%(name)s] [%(levelname)s] %(message)s')

# create console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
log.addHandler(ch)

# lots of shader code for OpenGL
SCENE_VS_SHADER = \
    """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 2) in vec3 aNormal;
    layout (location = 8) in vec2 aTexCoords;

    out vec2 TexCoords;

    out VS_OUT {
        vec3 FragPos;
        vec3 Normal;
        vec2 TexCoords;
        vec4 FragPosLightSpace;
    } vs_out;

    uniform mat4 projection;
    uniform mat4 view;
    uniform mat4 model;
    uniform mat4 lightSpaceMatrix;

    void main()
    {
        vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
        vs_out.Normal = transpose(inverse(mat3(model))) * aNormal;
        vs_out.TexCoords = aTexCoords;
        vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(vs_out.FragPos, 1.0);
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
    """

SCENE_F_SHADER = \
    """
    #version 330 core
    out vec4 FragColor;
    in VS_OUT {
        vec3 FragPos;
        vec3 Normal;
        vec2 TexCoords;
        vec4 FragPosLightSpace;
    } fs_in;
    uniform sampler2D diffuseTexture;
    uniform sampler2D shadowMap;
    uniform vec3 viewPos;
    struct Material {
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        float shininess;
    };
    struct Light {
        vec3 position;
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
    };
    uniform Material material;
    uniform Light light;
    uniform float scale;

    float ShadowCalculation(vec4 fragPosLightSpace) {
        // perform perspective divide
        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        // transform to [0,1] range
        projCoords = projCoords * 0.5 + 0.5;
        // get closest depth value from light's perspective
        // (using [0,1] range fragPosLight as coords)
        float closestDepth = texture(shadowMap, projCoords.xy).r;
        // get depth of current fragment from light's perspective
        float currentDepth = projCoords.z;
        // calculate bias (based on depth map resolution and slope)
        vec3 normal = normalize(fs_in.Normal);
        vec3 lightDir = normalize(light.position - fs_in.FragPos);
        float bias = scale * max(0.00025 * (1.0 - dot(normal, lightDir)), 0.0002);
        // PCF
        float shadow = 0.0;
        vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
        for(int x = -1; x <= 1; ++x) {
            for(int y = -1; y <= 1; ++y) {
                float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
                shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
            }
        }
        shadow /= 9.0;

        // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
        if(projCoords.z > 1.0)
            shadow = 0.0;
        return shadow;
    }

    void main() {
        vec3 color = texture(diffuseTexture, fs_in.TexCoords).rgb;
        vec3 normal = normalize(fs_in.Normal);
        // ambient
        vec3 ambient = light.ambient * material.ambient;
        // diffuse
        vec3 lightDir = normalize(light.position - fs_in.FragPos);
        float diff = max(dot(lightDir, normal), 0.0);
        vec3 diffuse = light.diffuse * (diff * material.diffuse);
        // specular
        vec3 viewDir = normalize(viewPos - fs_in.FragPos);
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
        vec3 specular = light.specular * (spec * material.specular);
        // calculate shadow
        float shadow = ShadowCalculation(fs_in.FragPosLightSpace);
        vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;

        FragColor = vec4(lighting, 1.0);
    }

    """

SCENE_F_SHADER_SIMPLE = \
    """
    #version 330 core
    out vec4 FragColor;
    in VS_OUT {
        vec3 FragPos;
        vec3 Normal;
        vec2 TexCoords;
        vec4 FragPosLightSpace;
    } fs_in;
    uniform sampler2D diffuseTexture;
    uniform vec3 viewPos;
    struct Material {
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        float shininess;
    };
    struct Light {
        vec3 position;
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
    };
    uniform Material material;
    uniform Light light;

    void main() {
        vec3 color = texture(diffuseTexture, fs_in.TexCoords).rgb;
        vec3 normal = normalize(fs_in.Normal);
        // ambient
        vec3 ambient = light.ambient * material.ambient;
        // diffuse
        vec3 lightDir = normalize(light.position - fs_in.FragPos);
        float diff = max(dot(lightDir, normal), 0.0);
        vec3 diffuse = light.diffuse * (diff * material.diffuse);
        // specular
        vec3 viewDir = normalize(viewPos - fs_in.FragPos);
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
        vec3 specular = light.specular * (spec * material.specular);
        vec3 lighting = (ambient + diffuse + specular) * color;

        FragColor = vec4(lighting, 1.0);
    }

    """

SHADOW_VS_SHADER = \
    """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 lightSpaceMatrix;
    uniform mat4 model;
    void main() {
        gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
    }
    """

SHADOW_F_SHADER = \
    """
    #version 330 core
    void main() {
    }
    """


class Scene:
    def __init__(self, param):
        # set the parameters
        if 'width' in param.keys():
            self.width = param["width"]
        else:
            self.width = 640
        if 'height' in param.keys():
            self.height = param["height"]
        else:
            self.height = 480
        if 'resource_path' in param.keys():
            self.resource_path = param['resource_path']
        else:
            self.resource_path = 'resources/'

        if 'ground_w' in param.keys():
            self.ground_size_w = param['ground_w']
        else:
            self.ground_size_w = 0.55

        if 'ground_d' in param.keys():
            self.ground_size_d = param['ground_d']
        else:
            self.ground_size_d = 0.5

        if 'camera_pos' in param.keys():
            # make sure that the camera pose is ok
            if np.linalg.norm(np.array(param['camera_pos'])) > 1.:
                log.error('Invalid Camera pose: The norm of the camera ' +
                          'position should not exceed 1!')
                log.warning('Setting camera position to ' + str([0., 0., 0.6]))
                self.cam_pos = [0., 0., 0.6]
            elif param['camera_pos'][2] < 0.1:
                log.error('Invalid Camera pose: The z coordinate needs ' +
                          'to be between 0.1 and 1.')
                log.warning('Setting camera position to ' + str([0., 0., 0.6]))
                self.cam_pos = [0., 0., 0.6]
            self.cam_pos = param['camera_pos']
        else:
            self.cam_pos = [0., 0., 0.6]

        if 'shadows' in param.keys():
            self.shadows = param['shadows']
        else:
            self.shadows =True

        # init a glut window as context
        glut.glutInit(sys.argv)
        glut.glutInitWindowSize(self.width, self.height)
        # Select type of Display mode:
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE |
                                 glut.GLUT_DEPTH | glut.GLUT_STENCIL)
        self.window = glut.glutCreateWindow("")
        # hide the glut window
        glut.glutHideWindow()

        # scan for available resources
        resources = [os.path.join(self.resource_path, f)
                     for f in os.listdir(self.resource_path)]
        if 'texture_files' in param.keys():
            self.texture_files = param['texture_files']
            self.texture_files = \
                filter(lambda f: os.path.basename(f), resources)
        else:  # if no textures are specified, load everything
            self.texture_files = \
                filter(lambda f: f.endswith('jpg') or f.endswith('jpeg') or
                       f.endswith('png') or f.endswith('bmp'), resources)

        self.material_files = filter(lambda f: f.endswith('mtl'), resources)

        if 'object_files' in param.keys():
            self.object_files = param['object_files']
            for f in self.object_files:
                if not os.path.exists(f):
                    raise ValueError("Object file not found: " + f)
                    sys.exit()
        else:  # if no files are specified, load all
            self.object_files = filter(lambda f: f.endswith('obj'), resources)

        self.texture_keys = dict()
        self.object_keys = dict()
        self.materials = dict()
        self.objects = []

        # load required textures
        for tex in self.texture_files:
            ind = self._load_texture(tex)
            name = os.path.basename(tex)[:os.path.basename(tex).find('.')]
            # register the texture twice, with and without file ending
            self.texture_keys[os.path.basename(tex)] = ind
            self.texture_keys[name] = ind

        # load required materials
        for mat in self.material_files:
            material = self._load_material(mat)
            name = os.path.basename(mat)[:os.path.basename(mat).find('.')]
            self.materials[name] = material
            self.materials[os.path.basename(mat)] = material

        # load object(s)
        for obj in self.object_files:
            ob = MeshObject(obj, self.texture_keys, self.materials)
            ind = len(self.objects)
            self.objects.append(ob)
            name = os.path.basename(obj)[:os.path.basename(obj).find('.')]
            self.object_keys[name] = ind

        # initialize the framebuffers that we will render to
        self._init_render()
        self._init_shadow()

        # some configuration
        self.z_far = max(1.1, np.linalg.norm(np.array(self.cam_pos)) + 1.)
        self.z_near = 0.05
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glClearColor(0.1, 0.2, 0.3, 1.0)

        # initialize the shaders
        if self.shadows:
            self.scene_shader = Shader(SCENE_VS_SHADER, SCENE_F_SHADER)
        else:
            self.scene_shader = Shader(SCENE_VS_SHADER, SCENE_F_SHADER_SIMPLE)
        self.shadow_depth_shader = Shader(SHADOW_VS_SHADER, SHADOW_F_SHADER)

        # shader configuration
        self.scene_shader.use()
        self._set_light()
        self.scene_shader.set_value("shadowMap", 1)
        self.scene_shader.set_value("diffuseTexture", 0)

    def draw(self, position, rot, texture_ground, object_shape,
             tip=None, texture_object='brushed_metal', light_pos=[0., 0.2, 1]):
        """
        Renders the scene for a given configuration
        Args:
            position: object's position [in meter]
            rot: object's rotation [in degree]
            texture_ground: which texture to use for the table surface
            object_shape: shape of the object
            tip: the tip's position [optional]
            texture_object: which texture to use for the object [default metal]

        Returns:
            im: RGB image of the scene (numpy array)
            d: depth image of the scene (numpy array)
        """
        gl.glClearDepth(1.)

        # check the light position, should not be too far off for shadows
        if self.shadows and np.linalg.norm(np.array(light_pos)) > 2.75:
            log.warning('Light might be too far away from the scene for ' +
                        'proper shadow rendering!')

        # get the light projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45., 1.0, self.z_near, 3.)

        lightProjectionMatrix = gl.glGetFloatv(gl.GL_PROJECTION_MATRIX)

        # get the light view matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluLookAt(light_pos[0], light_pos[1], light_pos[2],
                      0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        lightViewMatrix = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)

        # eval projection matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glMultMatrixf(lightProjectionMatrix)
        gl.glMultMatrixf(lightViewMatrix)
        lightSpaceMatrix = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)

        if self.shadows:
            # -----------------------------------------------------------------
            # render scene from light's perspective to depth map
            # -----------------------------------------------------------------
            # Use viewport the same size as the shadow map
            gl.glViewport(0, 0, self.shadow_size, self.shadow_size)

            # activate the right shader
            self.shadow_depth_shader.use()
            self.shadow_depth_shader.set_value("lightSpaceMatrix",
                                               lightSpaceMatrix)

            # activate the right framebuffer
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depth_texture_buffer)
            # clear it
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClear(gl.GL_STENCIL_BUFFER_BIT)

            # switching to front-face culling is supposed to help against
            # peter-panning (shadows are detached from their casting object)
            # but did not seem to help in my settings
            gl.glCullFace(gl.GL_FRONT)
            # render the scene
            self._render_scene(self.shadow_depth_shader, position, rot,
                               texture_ground, object_shape, texture_object,
                               tip)
            gl.glCullFace(gl.GL_BACK)
            # unbind the framebuffer
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

            # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # render scene
        # ---------------------------------------------------------------------
        # setup the right framebuffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        # clear it
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glClear(gl.GL_STENCIL_BUFFER_BIT)

        # configure the camera
        projection, view = self._set_camera()

        # use the right shader
        self.scene_shader.use()
        # set the transformation matrices
        self.scene_shader.set_value("projection", projection)
        self.scene_shader.set_value("view", view)
        # set light uniforms
        self.scene_shader.set_value("viewPos", self.cam_pos)
        self.scene_shader.set_value("light.position", light_pos)
        self.scene_shader.set_value("lightSpaceMatrix", lightSpaceMatrix)

        if self.shadows:
            # adapt to z_far value
            scale = min(max(float(2. / self.z_far), 0.5), 1.15)
            self.scene_shader.set_value("scale", scale)

            # bind the shadow texture
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_map)
        # render the scene
        self._render_scene(self.scene_shader, position, rot, texture_ground,
                           object_shape, texture_object, tip)
        # ---------------------------------------------------------------------

        glut.glutSwapBuffers()

        im, d = self._get_rgbd_image()

        return im, d

    def get_world_2_cam(self):
        """
        Returns the transformation matrix that converts a point in the
        world's coordinate frame (centred on the table) to the camera's frame
        """
        projection, view = self._set_camera()
        view = view.T
        gl.glLoadIdentity()
        gl.glRotate(-90, 1., 0., 0.)
        gl.glMultMatrixf(view)

        trans = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)

        return trans, projection

    def close(self):
        glut.glutDestroyWindow(self.window)

    def set_camera(self, cam_pos):
        """
        Sets the camera position to the given value and returns the updated
        transformation matrix from world to camera
        """
        if cam_pos[2] < 0.1:
            logging.error('Invalid Camera pose: The z coordinate needs to ' +
                          'be bigger than 0.1. Ignoring this request.')
            return self.get_world_2_cam()
        self.cam_pos = cam_pos

        self.z_far = max(1.1, np.linalg.norm(np.array(self.cam_pos)) + 1.)

        new_transform = self.get_world_2_cam()
        return new_transform

    def _get_rgbd_image(self):
        """
        Reads the colour and the depth buffer and returns numpy arrays
        """
        # read the colour buffer
        pixels = gl.glReadPixels(0, 0, self.width, self.height,
                                 gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        # convert to PIL image
        image = Image.frombytes('RGBA', (self.width, self.height), pixels)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        # convert to numpy array
        im = np.array(image.getdata(), np.uint8).reshape(self.height,
                                                         self.width, 4)
        # we have to get rid of the alpha channel to get the correct result
        im_res = np.copy(im[:, :, :3])
        # convert to float and divide by 255. to get values between 0 and 1
        im_res = im_res.astype(np.float32)/255.

        # now take care of the depth part
        depth_raw = gl.glReadPixels(0, 0, self.width, self.height,
                                    gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        # convert to PIL image
        depth = Image.frombytes('F', (self.width, self.height), depth_raw)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        # convert to numpy array
        d = np.array(depth.getdata(), np.float32).reshape(self.height,
                                                          self.width, 1)

        # linearize depth values
        d = 2. * d - 1.
        d = 2. * self.z_near * self.z_far / \
            (self.z_far + self.z_near - d * (self.z_far - self.z_near))
        return im_res, d

    def _load_material(self, filename):
        """
        Loads a material file and reads its values to a dictionary
        """
        if not os.path.exists(filename):
            raise ValueError("Material file not found: " + filename)
            return
        contents = {}
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] != 'newmtl':
                contents[values[0]] = map(float, values[1:])

        return contents

    def _load_texture(self, filename):
        """
        Loads an image as texture
        """
        if not os.path.exists(filename):
            raise ValueError("Texture file not found: " + filename)
            sys.exit()
        # Load the image
        image = Image.open(filename)
        ix = image.size[0]
        iy = image.size[1]
        image = image.tobytes("raw", "RGBX", 0, -1)

        # Create Texture
        ind = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, ind)

        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,
                           gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,
                           gl.GL_REPEAT)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_LINEAR_MIPMAP_NEAREST)

        glu.gluBuild2DMipmaps(gl.GL_TEXTURE_2D, 3, ix, iy, gl.GL_RGBA,
                              gl.GL_UNSIGNED_BYTE, image)

        return ind

    def _init_shadow(self):
        """
        Initializes a texture to which we render the depth-map for calculating
        shadows and creates a framebuffer object that renders the scene's depth
        component to this texture.
        """
        self.shadow_size = 2048
        # configure depth map
        self.depth_texture_buffer = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depth_texture_buffer)

        # create depth texture
        self.depth_map = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_map)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT,
                        self.shadow_size, self.shadow_size, 0,
                        gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,
                           gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,
                           gl.GL_CLAMP_TO_BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR,
                            border_color)
        # attach depth texture as FBO's depth buffer
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                  gl.GL_TEXTURE_2D, self.depth_map, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _init_render(self):
        """
        Initialize framebuffers to render offscreen, i.e. without displaying a
        window
        """
        # Framebuffer
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)

        # Color renderbuffer
        self.rbo_color = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo_color)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, self.width,
                                 self.height)
        gl.glFramebufferRenderbuffer(gl.GL_DRAW_FRAMEBUFFER,
                                     gl.GL_COLOR_ATTACHMENT0,
                                     gl.GL_RENDERBUFFER, self.rbo_color)

        # Depth and stencil renderbuffer.
        self.rbo_depth_stencil = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo_depth_stencil)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8,
                                 self.width, self.height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER,
                                     gl.GL_DEPTH_STENCIL_ATTACHMENT,
                                     gl.GL_RENDERBUFFER,
                                     self.rbo_depth_stencil)

        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)

        # Sanity check
        assert fbos.glCheckFramebufferStatusEXT(gl.GL_FRAMEBUFFER)

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

    def _set_camera(self):
        """
        Sets the camera's position relative to the scene and returns the
        projection and the view matrix
        """
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45.0, float(self.width)/float(self.height),
                           self.z_near, self.z_far)
        projection = gl.glGetFloatv(gl.GL_PROJECTION_MATRIX)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glViewport(0, 0, self.width, self.height)
        glu.gluLookAt(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
                      0.0, 0.0, 0.0, 0.0, 1.0, .0)
        view = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)
        return projection, view

    def _set_light(self):
        """
        Defines the colour of lights and registers them with the shader
        """
        self.scene_shader.use()
        diffuseColor = np.array([1, 1, 1])
        ambientColor = np.array([0.9, 0.875, 0.85])
        self.scene_shader.set_value("light.ambient", ambientColor)
        self.scene_shader.set_value("light.diffuse", diffuseColor)
        self.scene_shader.set_value("light.specular",
                                    np.array([0.95, 0.95, 1.0]))

    def _render_scene(self, shader, position, rot, texture,
                      object_shape, texture_object, tip=None):
        # activate the given shader
        shader.use()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        # we draw all objects into the x-z-plane. As the camera faces to the
        # x-y plane, we rotate the scene by 90 degree
        gl.glRotate(90, 1., 0., 0.)

        # draw the ground
        model = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)
        shader.set_value("model", model)
        self._draw_ground(texture, shader)

        # draw the object
        gl.glPushMatrix()
        gl.glTranslate(position[0], 0, position[1])
        gl.glRotate(-rot, 0, 1, 0)
        model = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)
        shader.set_value("model", model)
        obj = self.objects[self.object_keys[object_shape]]
        obj.draw(shader, texture_object, shader.id == self.scene_shader.id)
        gl.glPopMatrix()

        # draw the tip if it is given
        if tip is not None:
            gl.glTranslatef(tip[0], 0, tip[1])
            tip_obj = self.objects[self.object_keys['tip']]
            model = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)
            shader.set_value("model", model)
            tip_obj.draw(shader, 'brushed_metal',
                         shader.id == self.scene_shader.id)
        # reset the modelview matrix
        gl.glLoadIdentity()

    def _draw_ground(self, texture, shader):
        """
        Draws a simple rectangle as the ground.
        """
        # activate the right texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_keys[texture])

        # if we are not dealing with the shadow shader, we set the material
        # properties
        if shader.id != self.shadow_depth_shader.id:
            if '.' in texture:
                mat_name = texture[:texture.find('.')]
            else:
                mat_name = texture
            mtl = self.materials[mat_name]

            # define material properties
            shader.set_value("material.ambient", mtl['Ka'])
            shader.set_value("material.diffuse", mtl['Kd'])
            shader.set_value("material.specular", mtl['Ks'])
            shader.set_value("material.shininess", mtl['Ns'][0])

        # top face - gets more vertices
        vertices = [[-1, 0., -1, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [-1, 0.,  0, 0.0, 1.0, 0.0, 0.0, 0.5],
                    [0,  0.,  0, 0.0, 1.0, 0.0, 0.5, 0.5],
                    [0,  0., -1, 0.0, 1.0, 0.0, 0.5, 0.0],
                    #
                    [0,  0., -1, 0.0, 1.0, 0.0, 0.5, 0.0],
                    [0,  0.,  0, 0.0, 1.0, 0.0, 0.5, 0.5],
                    [1,  0.,  0, 0.0, 1.0, 0.0, 1.0, 0.5],
                    [1,  0., -1, 0.0, 1.0, 0.0, 1.0, 0.0],
                    #
                    [-1, 0., 0, 0.0, 1.0, 0.0, 0.0, 0.5],
                    [-1, 0., 1, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [0,  0., 1, 0.0, 1.0, 0.0, 0.5, 1.0],
                    [0,  0., 0, 0.0, 1.0, 0.0, 0.5, 0.5],
                    #
                    [0,  0., 0, 0.0, 1.0, 0.0, 0.5, 0.5],
                    [0,  0., 1, 0.0, 1.0, 0.0, 0.5, 1.0],
                    [1,  0., 1, 0.0, 1.0, 0.0, 1.0, 1.0],
                    [1,  0., 0, 0.0, 1.0, 0.0, 1.0, 0.5]]
        # bottom face
        vertices += [[-1, -0.05, -1, 0.0, -1.0, 0.0, 0.0, 0.0],
                     [-1, -0.05,  1, 0.0, -1.0, 0.0, 0.0, 1.0],
                     [1,  -0.05,  1, 0.0, -1.0, 0.0, 1.0, 1.0],
                     [1,  -0.05, -1, 0.0, -1.0, 0.0, 1.0, 0.0]]
        # front face
        vertices += [[-1, -0.05,  1, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [-1,  0.00,  1, 0.0, 0.0, 1.0, 0.0, 1.0],
                     [1,   0.00,  1, 0.0, 0.0, 1.0, 1.0, 1.0],
                     [1,  -0.05,  1, 0.0, 0.0, 1.0, 1.0, 0.0]]
        # back face
        vertices += [[-1, -0.05, -1, 0.0, 0.0, -1.0, 0.0, 0.0],
                     [-1,  0.00, -1, 0.0, 0.0, -1.0, 0.0, 1.0],
                     [1,   0.00, -1, 0.0, 0.0, -1.0, 1.0, 1.0],
                     [1,  -0.05, -1, 0.0, 0.0, -1.0, 1.0, 0.0]]
        # right face
        vertices += [[1, -0.05, -1, 1.0,  0.0, 0.0, 0.0, 0.0],
                     [1,  0.00,  1, 1.0,  0.0, 0.0, 0.0, 1.0],
                     [1, -0.05,  1, 1.0,  0.0, 0.0, 1.0, 1.0],
                     [1,  0.00, -1, 1.0,  0.0, 0.0, 1.0, 0.0]]
        # left face
        vertices += [[-1, -0.05, -1, -1.0, 0.0, 0.0, 0.0, 0.0],
                     [-1,  0.00,  1, -1.0, 0.0, 0.0, 0.0, 1.0],
                     [-1, -0.05,  1, -1.0, 0.0, 0.0, 1.0, 1.0],
                     [-1,  0.00, -1, -1.0, 0.0, 0.0, 1.0, 0.0]]

        # we use different texture coordinates for the different materials
        tex_scale_x = 1.
        tex_scale_y = 1.
        if 'abs' in texture:
            tex_scale_x = 4.
            tex_scale_y = 4.
        elif 'pu'in texture:
            tex_scale_x = 6.
            tex_scale_y = 6.
        elif 'delrin'in texture:
            tex_scale_x = 2.
            tex_scale_y = 2.
        elif 'plywood.'in texture:
            tex_scale_x = 2.5

        def scale(x):
            x[0] *= self.ground_size_w/2.
            x[2] *= self.ground_size_d/2.
            x[6] *= tex_scale_x
            x[7] *= tex_scale_y
            return x
        vertices = map(scale, vertices)

        tmp_indices = [[0, 3, 2], [0, 1, 2]]
        indices = []
        for i in np.arange(9):
            indices += map(lambda x: map(lambda y: y + 4 * i, x), tmp_indices)
        vb = vbo.VBO(np.array(vertices, dtype=np.float32))
        eb = vbo.VBO(np.array(indices, dtype=np.int32),
                     target=gl.GL_ELEMENT_ARRAY_BUFFER)

        vb.bind()
        eb.bind()
        try:
            # vetrices contain vertex position (3), normal (3) and texture
            # coordinates (2)
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 4 * 8, vb)
            gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
            gl.glTexCoordPointer(2, gl.GL_FLOAT, 4 * 8, vb+24)
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glNormalPointer(gl.GL_FLOAT, 4 * 8, vb+12)

            gl.glDrawElements(gl.GL_TRIANGLES, 2*3*9, gl.GL_UNSIGNED_INT, None)
        finally:
            eb.unbind()
            vb.unbind()
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
            gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)


def demo(argv=None):
    """
    Demo: For each object, draws a random datafile from the push dataset and
        renders the scene with random camera and light position.

    Args:
        source_dir: path to the MIT push dataset (preprocessed)
        out-dir: where to store the output files
        resource-dir: path to the directory containing mesh files etc.
        shadows: Render with shadows or not [default: True]
    """
    parser = argparse.ArgumentParser('render_scene')
    parser.add_argument('--source-dir', dest='source_dir', type=str,
                        help='Directory holding the preprocessed MIT Push ' +
                            'dataset.')
    parser.add_argument('--out-dir', dest='out_dir', type=str,
                        help='Where to store the images.')
    parser.add_argument('--resource-dir', dest='resource_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '../resources'),
                        help='where to find textures etc')
    parser.add_argument('--shadows', dest='shadows', type=str,
                        default=True, help='render with shadows?')

    args = parser.parse_args(argv)

    ps = Scene({'resource_path': args.resource_dir,
                'ground_d': 0.5, 'ground_w': 0.6, 'shadows': args.shadows})

    #all objects and surfaces:
    #obs = ['ellip1', 'ellip2', 'ellip3', 'rect1', 'rect2', 'rect3',
    #       'tri1', 'tri2', 'tri3', 'butter', 'hex']
    #surfaces = ['delrin', 'abs', 'pu', 'plywood']

    # Single example
    obs = ['rect1']
    surfaces = ['abs']

    plt.ioff()
    for o in obs:
        # chose a random surface
        s_int = np.random.randint(len(surfaces))
        s = surfaces[s_int]
        path = os.path.join(args.source_dir, s, o)

        # chose a random data-file
        fs = [os.path.join(path, f) for f in os.listdir(path)]
        filename = fs[np.random.randint(len(fs))]

        # sample a random camera position
        tilt = np.random.ranf() - 0.5
        if tilt > 0:
            camera_pos = (np.random.ranf(size=(3)) - 0.5) / 2.
        else:
            camera_pos = np.zeros(3)
        camera_pos[2] = 0.3 + np.random.ranf() * 0.6
        ps.set_camera(camera_pos)

        # sample a random offset for the object
        off = ((np.random.ranf(2) - 0.5) / 5.) * camera_pos[2]

        # sample a random light position
        light = (np.random.ranf(size=(3)) - 0.5) * 2
        light[2] = max(np.random.ranf()*1.5, 0.3)

        # load the data
        try:
            data_in = h5py.File(filename, "r", driver='core')
            data = {}
            for key, val in data_in.iteritems():
                data[key] = val.value
        except:
            log.error('Could not load '+ filename)
            return

        # render and save an image of the scene every 10 steps
        for ind in np.arange(len(data['object']))[::10]:
            pos = data["object"][ind]
            tip = data['tip'][ind] + off

            tr = pos[:2] + off
            rot = pos[2]

            # render the scene
            im, d = ps.draw(tr, rot * 180. / np.pi, s, o, tip=tip,
                            light_pos=light)

            # plot and save
            fig, ax = plt.subplots(2, figsize=(8, 13))
            ax[0].imshow(im)
            ax[0].set_ylim([0, 480])
            ax[0].set_xlim([0, 640])
            ax[1].imshow(d.reshape(480, 640), cmap='gray', vmin=0.2, vmax=1.)
            ax[1].set_ylim([0, 480])
            ax[1].set_xlim([0, 640])
            # hide the ticks but keep the axis as frame
            ax[0].get_xaxis().set_ticks([])
            ax[0].get_yaxis().set_ticks([])
            ax[1].get_xaxis().set_ticks([])
            ax[1].get_yaxis().set_ticks([])
            name = s + '_' + o + '_' + str(ind) + ".jpg"
            fig.tight_layout()
            plt.savefig(os.path.join(args.out_dir, name), dpi=90)
            plt.close(fig)
    ps.close()

if __name__ == "__main__":
    demo()
