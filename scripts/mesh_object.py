#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Class for loading and drawing mesh objects in .obj format

@author: Alina Kloss
"""

import OpenGL.GL as gl
from OpenGL.arrays import vbo
import numpy as np
import sys
import os

class MeshObject:
    class Texture:
        def __init__(self):
            self.id = -1
            self.type = None
            self.path = None

    class Face:
        def __init__(self):
            self.vertexIndices = []
            self.texIndices = []
            self.normals = []

    def __init__(self, filepath, textures, materials):
        self.num_faces = 0
        self.texture_keys = textures
        self.materials = materials

        self.max_y = 0.
        self.min_y = 1000.
        self.min_x = 1000.
        self.min_z = 1000.

        self.load(filepath)

    def load(self, filename):
        if not filename.endswith('.obj'):
            raise ValueError("Mesh file needs to be in .obj format")
            sys.exit()

        faces = []
        vs = []
        ns = []
        ts = []

        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            # collect the vertices
            # save the minimum y coordinate while at it
            if values[0] == 'v':
                v = map(float, values[1:4])
                vs.append(v)
                self.min_y = min(self.min_y, v[1])
                self.min_x = min(self.min_x, v[0])
                self.min_z = min(self.min_z, v[2])
                self.max_y = max(self.max_y, v[1])
            # collect the normals
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                ns.append(v)
            # collect texture coords
            elif values[0] == 'vt':
                vals = map(float, values[1:3])
                ts.append(vals)
            elif values[0] == 'mtllib':
                self.mtl = self.materials[values[1]]
            elif values[0] == 'f':
                face = MeshObject.Face()
                tex = []
                norms = []
                vertices = []
                for v in values[1:]:
                    w = v.split('/')
                    vertices.append(int(w[0]) - 1)  # indexing starts at 1 in obj files
                    if len(w) >= 2 and len(w[1]) > 0:
                        tex.append(int(w[1]) - 1)
                    else:
                        tex.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]) - 1)
                    else:
                        norms.append(0)
                face.normals = norms
                face.texIndices = tex
                face.vertexIndices = vertices
                faces.append(face)
        self.num_faces = len(faces)

        # adapt the origin of the mesh to lie on
        # its centre of mass projected to the bottom
        faces, vs = self.set_origin(filename, faces, vs)

        # reorganize the data to be usable with shaders
        self.reorganize(faces, vs, ns, ts)

        return

    """
    Ensure that allmeshes are drawn around their centre of mass.
    This means we have to calculate it and then shift all point coordinates to
    be relative to this new origin.

    The idea is to calculate the centroid of each face and then calculate the
    centre of mass of the mesh as the average of face centroids weighted by
    face area.
    Method taken from
    https://troymcconaghy.com/2015/06/25/how-blender-calculates-center-of-mass/

    In the last step, we want the object to lie on the ground plane, so we
    shift all points up such that 0 is the minimum y coordinate.
    """
    def set_origin(self, filename, faces, vertices):
        total_area = 0.
        centroid = np.zeros(3)

        for f in faces:
            a = self.get_area(f, vertices)
            c = self.get_centroid(f, vertices)

            total_area += a
            centroid += a * c
        # calculate the area-weighted average of face-centroids
        centroid /= total_area

        # if the centre of mass is not the origin already, subtract the
        # centroid from every vertex to move it there
        if 'tri' not in os.path.basename(filename):
            if np.any(np.abs(centroid)) > 0.0001:
                for v in vertices:
                    v[0] -= centroid[0]
                    v[1] -= centroid[1]
                    v[2] -= centroid[2]
                # apply the shift to the minimum y coordinate we got
                # when loading the mesh
                self.min_y -= centroid[1]
                self.min_x -= centroid[0]
                self.min_z -= centroid[2]
                self.max_y -= centroid[1]

        # now lift up the object to have minimum y-coordinate = 0
        for v in vertices:
            v[1] -= self.min_y

        self.min_x -= self.min_y
        self.min_z -= self.min_y
        self.max_y -= self.min_y
        self.min_y -= self.min_y
        return faces, vertices

    def get_area(self, f, vertices):
        """ calculate area using herons formular """
        p1 = np.array(vertices[f.vertexIndices[0]])
        p2 = np.array(vertices[f.vertexIndices[1]])
        p3 = np.array(vertices[f.vertexIndices[2]])

        ab = np.linalg.norm(p1 - p2)
        ac = np.linalg.norm(p1 - p3)
        bc = np.linalg.norm(p3 - p2)

        s = 0.5 * (ab + bc + ac)
        val = s * (s - ab) * (s - ac) * (s - bc)
        return np.sqrt(val)

    def get_centroid(self, f, vertices):
        """ the centroid of a triangle is the mean of its vertices"""
        p1 = np.array(vertices[f.vertexIndices[0]])
        p2 = np.array(vertices[f.vertexIndices[1]])
        p3 = np.array(vertices[f.vertexIndices[2]])

        c = (p1 + p2 + p3) / 3.

        return c

    def reorganize(self, faces, vs, ns, ts):
        """
        We have to reorganize the data such that we can index everything
        with one index array. For this we create vertices that contain
        texture and noral information (possibly duplicating data) and an
        index array that indexes the vertices for each face
        """
        last_ind = 0
        vertices = []
        vertex_indices = []
        for f in faces:
            v1 = vs[f.vertexIndices[0]]
            v2 = vs[f.vertexIndices[1]]
            v3 = vs[f.vertexIndices[2]]
            t1 = ts[f.texIndices[0]]
            t2 = ts[f.texIndices[1]]
            t3 = ts[f.texIndices[2]]
            n1 = ns[f.normals[0]]
            n2 = ns[f.normals[1]]
            n3 = ns[f.normals[2]]
            vertex1 = [v1 + n1 + t1]
            vertex2 = [v2 + n2 + t2]
            vertex3 = [v3 + n3 + t3]

            if vertex1 in vertices:
                ind1 = vertices.index(vertex1)
            else:
                vertices.append(vertex1)
                ind1 = last_ind
                last_ind += 1

            if vertex2 in vertices:
                ind2 = vertices.index(vertex2)
            else:
                vertices.append(vertex2)
                ind2 = last_ind
                last_ind += 1

            if vertex3 in vertices:
                ind3 = vertices.index(vertex3)
            else:
                vertices.append(vertex3)
                ind3 = last_ind
                last_ind += 1

            vertex_indices += [[ind1, ind2, ind3]]

        self.vbo = vbo.VBO(np.array(vertices, dtype=np.float32))
        self.ebo = vbo.VBO(np.array(vertex_indices, dtype=np.int32),
                           target=gl.GL_ELEMENT_ARRAY_BUFFER)

    def draw(self, shader, texture, lighting=True):
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_keys[texture])

        if lighting:
            # define material properties
            shader.set_value("material.ambient", self.mtl['Ka'])
            shader.set_value("material.diffuse", self.mtl['Kd'])
            shader.set_value("material.specular", self.mtl['Ks'])
            shader.set_value("material.shininess", self.mtl['Ns'][0])
        self.ebo.bind()
        self.vbo.bind()
        try:
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 4 * 8, self.vbo)
            gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
            gl.glTexCoordPointer(2, gl.GL_FLOAT, 4 * 8, self.vbo + (4*6))
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glNormalPointer(gl.GL_FLOAT, 4 * 8, self.vbo + (4*3))

            gl.glDrawElements(gl.GL_TRIANGLES, self.num_faces * 3,
                              gl.GL_UNSIGNED_INT, None)

        finally:
            self.ebo.unbind()
            self.vbo.unbind()
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
            gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)

