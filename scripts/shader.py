#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:30:06 2017
Wrapper class for OpenGl shaders

Code adapted from https://learnopengl.com/
@author: Alina Kloss
"""

import OpenGL.GL as gl
import numpy as np
import os
import OpenGL.GL.shaders as shaders


class Shader:
    def __init__(self, vertex_code=None, fragment_code=None,
                 vertex_file=None, fragment_file=None):
        if vertex_file is not None:
            vs = self.vertex_from_file(vertex_file)
        elif vertex_code is not None:
            vs = self.vertex_from_string(vertex_code)
        else:
            raise ValueError('You need to supply a vertex shader!')

        if fragment_file is not None:
            fs = self.fragment_from_file(vertex_file)
        elif fragment_code is not None:
            fs = self.fragment_from_string(fragment_code)
        else:
            raise ValueError('You need to supply a fragment shader!')

        self.id = shaders.glCreateProgram()
        shaders.glAttachShader(self.id, vs)
        shaders.glAttachShader(self.id, fs)

        shaders.glLinkProgram(self.id)
        if shaders.glGetProgramiv(self.id, shaders.GL_LINK_STATUS) != gl.GL_TRUE:
            print shaders.glGetProgramInfoLog(self.id)
            raise RuntimeError(shaders.glGetProgramInfoLog(self.id))

        # cleanup
        shaders.glDeleteShader(vs)
        shaders.glDeleteShader(fs)

    def vertex_from_file(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError('File ' + filepath + 'does not exist!')

        with open('Path/to/file', 'r') as content_file:
            content = content_file.read()
            vs = self.create_shader(shaders.GL_VERTEX_SHADER, content)
        return vs

    def vertex_from_string(self, content):
        vs = self.create_shader(shaders.GL_VERTEX_SHADER, content)
        return vs

    def fragment_from_file(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError('File ' + filepath + 'does not exist!')

        with open('Path/to/file', 'r') as content_file:
            content = content_file.read()
            fs = self.create_shader(shaders.GL_FRAGMENT_SHADER, content)
        return fs

    def fragment_from_string(self, content):
        fs = self.create_shader(shaders.GL_FRAGMENT_SHADER, content)
        return fs

    def create_shader(self, shader_type, source):
        """compile a shader."""
        shader = shaders.glCreateShader(shader_type)
        shaders.glShaderSource(shader, source)
        shaders.glCompileShader(shader)
        if shaders.glGetShaderiv(shader, shaders.GL_COMPILE_STATUS) != gl.GL_TRUE:
            raise RuntimeError(shaders.glGetShaderInfoLog(shader))
        return shader

    def use(self):
        shaders.glUseProgram(self.id)

    # utility uniform functions
    def set_value(self, name, val):
        if type(val) in [int, bool]:
            gl.glUniform1i(gl.glGetUniformLocation(self.id, name), int(val))
        elif type(val) == float:
            gl.glUniform1f(gl.glGetUniformLocation(self.id, name), val)
        elif type(val) == list:
            if len(val) == 2:
                gl.glUniform2fv(gl.glGetUniformLocation(self.id, name), 1, val)
            elif len(val) == 3:
                gl.glUniform3fv(gl.glGetUniformLocation(self.id, name), 1, val)
            elif len(val) == 4:
                gl.glUniform4fv(gl.glGetUniformLocation(self.id, name), 1, val)
        elif type(val) == np.ndarray:
            shape = val.shape
            if len(shape) == 1:  # vector
                if shape[0] == 2:
                    gl.glUniform2fv(gl.glGetUniformLocation(self.id, name),
                                    1, val)
                elif shape[0] == 3:
                    gl.glUniform3fv(gl.glGetUniformLocation(self.id, name),
                                    1, val)
                elif shape[0] == 4:
                    gl.glUniform4fv(gl.glGetUniformLocation(self.id, name),
                                    1, val)
            elif len(shape) == 2:
                if shape[0] == 2 and shape[1] == 2:
                    gl.glUniformMatrix2fv(gl.glGetUniformLocation(self.id,
                                                                  name),
                                          1, gl.GL_FALSE, val)
                elif shape[0] == 3 and shape[1] == 3:
                    gl.glUniformMatrix3fv(gl.glGetUniformLocation(self.id,
                                                                  name),
                                          1, gl.GL_FALSE, val)
                elif shape[0] == 4 and shape[1] == 4:
                    gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.id,
                                                                  name),
                                          1, gl.GL_FALSE, val)
        else:
            raise ValueError('Cannot set uniform with value ' + str(val))
