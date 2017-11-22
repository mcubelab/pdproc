#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:57:18 2017

Annotate the dataset with contact points and normals. 

Warining: This code is terribly slow for ellipses!
@author: Alina Kloss
"""

import h5py
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as p
import sympy as sp
from sympy import Ellipse, Circle, Point

from joblib import Parallel, delayed

import logging
import sys

import pickle
import argparse


# setup logging
log = logging.getLogger('annotate')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s: [%(name)s] [%(levelname)s] %(message)s')

# create console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
log.addHandler(ch)


# we use sympy to calculate the contact points. For line-circle intersections,
# we first solve the equations symbolically and then plug in the values when
# needed

# intersection line - circle
x, y, m, c, r = sp.symbols('x y m c r', real=True)
# solving for x
eq_line_circle_x = sp.Eq((m * x + c)**2, (r**2 - x**2))
expr_line_circle_x = sp.solve(eq_line_circle_x, x)
# solving for y
eq_line_circle_y = sp.Eq(((y - c) / m)**2, (r**2 - y**2))
expr_line_circle_y = sp.solve(eq_line_circle_y, y)

# Define the objects as sympy ellipses and circles
# Warning: this can lead to an error with sympy versions < 1.1.
ellip1 = Circle(Point(0., 0.), 0.0525)
ellip2 = Ellipse(Point(0., 0.), 0.0525, 0.065445)
ellip3 = Ellipse(Point(0., 0.), 0.0525, 0.0785)


def get_contact_annotation(source_dir, out_dir, num_jobs=10, debug=True,
                           criteria=['a=0']):
    if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    mats = [f for f in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, f)) if 'delrin' in f]
    for mat in mats:
        objects = [f for f in os.listdir(os.path.join(source_dir, mat))
                   if 'zip' not in f]
        if not os.path.exists(os.path.join(out_dir, mat)):
            os.mkdir(os.path.join(out_dir, mat))
        for ob in objects:
            files = [os.path.join(source_dir, mat, ob, f)
                     for f in os.listdir(os.path.join(source_dir, mat, ob))]

            # filter out files that don't fullfill the criteria
            for crit in criteria:
                files = filter(lambda x: crit in os.path.basename(x), files)

            target_dir = os.path.join(out_dir, mat, ob)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            # don't override what we already did
            files = \
                filter(lambda x:
                       not os.path.exists(os.path.join(target_dir,
                                                       os.path.basename(x))),
                       files)

            if debug:
                debug_dir = os.path.join(out_dir, 'visualization')
                if not os.path.exists(debug_dir):
                    os.mkdir(debug_dir)
            # process multiple files in parallel
            Parallel(n_jobs=num_jobs)(delayed(annotate)
                                      (ob, mat, num, f, target_dir, len(files),
                                       debug, debug_dir)
                                      for num, f in enumerate(files))


def annotate(ob, mat, num, filename, target, num_files, debug, debug_dir):
    name = os.path.basename(filename)[:os.path.basename(filename).rfind('.',)]
    log.info(filename)
    try:
        data_in = h5py.File(filename, "r", driver='core')
        data = {}
        for key, val in data_in.iteritems():
            data[key] = val[:]
    except:
        log.exception("message")
        return None

    tips = data['tip']
    positions = data['object']
    contacts = []
    normals = []

    for ind in np.arange(len(positions)):
        tip = np.array(tips[ind])
        pos = np.array(positions[ind][:2])
        ang = positions[ind][2]

        # find the contact point at this timestep and the surface normal there
        contact, normal = get_contact_point(ob, tip, pos, ang, ind, num, name,
                                            debug, debug_dir)
        # contact and normal contain two different estimates: contact[0] is
        # the contact point for the real tip radius (with normal normal[0]),
        # contact[1] is an estimate using a bigger tip radius

        # if we found a contact point with the real tip radius, use it
        if np.any(contact[0]) != 0:
            contacts += [contact[0].astype(np.float32)]
            if np.linalg.norm(normal[0]) == 0.:
                log.warning('File ' + os.path.basename(filename) +
                            ': [Found no normal for contact point]')
            # sometimes we get very big values for the normal, in that case we
            # normalise its length
            elif np.linalg.norm(normal[0]) >= 100:
                normal[0] /= np.linalg.norm(normal[0])
            normals += [normal[0].astype(np.float32)]
        # we only use contact[1] if we did not find a contact point with the
        # real tip radius, but the object moved a significant distance
        elif ind != len(positions) - 1 and \
                (np.linalg.norm(np.array(positions[ind+1][:2]) - pos) > 5e-4 or
                 np.linalg.norm(ang - positions[ind+1][2]) > 0.001):
            contacts += [contact[1].astype(np.float32)]
            if np.any(contact[1]) != 0 and np.linalg.norm(normal[1]) == 0:
                log.warning('File ' + os.path.basename(filename) +
                            ': [Found no normal for contact point]')
            elif np.linalg.norm(normal[1]) >= 100:
                normal[1] /= np.linalg.norm(normal[1])
            normals += [normal[1].astype(np.float32)]
        else:
            contacts += [contact[0].astype(np.float32)]
            normals += [normal[0].astype(np.float32)]

    data['contact_points'] = np.array(contacts)
    data['contact_normals'] = np.array(normals)

    with h5py.File(os.path.join(target,
                                os.path.basename(filename)), 'w') as h5:
        for key, val in data.iteritems():
            h5.create_dataset(key, data=val)

    return


def get_contact_point(ob, tip, pos, rot, image_num, file_num, name,
                      debug, debug_dir):
    pos = pos.reshape(2, 1)

    # we'll look at everything in a coordinate system relative to the object
    # so we have to move and rotate the tip to undo this
    rot_mat = np.zeros((2, 2))
    rot_mat[0, 0] = np.cos(rot)
    rot_mat[0, 1] = -np.sin(rot)
    rot_mat[1, 0] = np.sin(rot)
    rot_mat[1, 1] = np.cos(rot)

    rot_mat2 = np.zeros((2, 2))
    rot_mat2[0, 0] = np.cos(-rot)
    rot_mat2[0, 1] = -np.sin(-rot)
    rot_mat2[1, 0] = np.sin(-rot)
    rot_mat2[1, 1] = np.cos(-rot)

    tip = np.reshape(tip, (2, 1)) - pos
    tip = np.dot(rot_mat2, tip)

    # we use two radii for the tip
    # the first is slightly bigger than the real tip radius, to account for
    # inaccuracies in the tracking and temporal resampling of the data
    # the second is even bigger. Contact points found with this radius will
    # only be used if the object moved, but no contact point was found for the
    # smaller radius
    tip_r = 0.0048
    tip_r2 = 0.0068

    # if desired, we'll plot a few examples and save them for inspection
    debug = debug and file_num < 10 and image_num % 30 == 0

    if debug:
        fig, ax = plt.subplots()
        ax.set_ylim([-0.1, 0.1])
        if 'tri' in ob:
            ax.set_ylim([-0.15, 0.1])
            ax.set_xlim([-0.15, 0.1])
        else:
            ax.set_ylim([-0.1, 0.1])
            ax.set_xlim([-0.1, 0.1])
        plt.gca().set_aspect('equal', adjustable='box')
        ax.add_patch(p.Ellipse((tip[0], tip[1]), 2*tip_r, 2*tip_r, 0,
                               alpha=0.2, facecolor='g', edgecolor='g'))
        ax.add_patch(p.Ellipse((tip[0], tip[1]), 2*tip_r2, 2*tip_r2, 0,
                               fill=False, edgecolor='c'))

    # depending on the object's shape, we calculate normals and contact
    # points in different ways
    if 'rect' in ob:
        # c-----d
        # |     |
        # a-----b
        # get the positions of the corner points
        if '1' in ob:
            a = np.array([[-0.045], [-0.045]])
            b = np.array([[0.045], [-0.045]])
            c = np.array([[-0.045], [0.045]])
            d = np.array([[0.045], [0.045]])
        if '2' in ob:
            a = np.array([[-0.044955], [-0.05629]])
            b = np.array([[0.044955], [-0.05629]])
            c = np.array([[-0.044955], [0.05629]])
            d = np.array([[0.044955], [0.05629]])
        if '3' in ob:
            a = np.array([[-0.067505], [-0.04497]])
            b = np.array([[0.067505], [-0.04497]])
            c = np.array([[-0.067505], [0.04497]])
            d = np.array([[0.067505], [0.04497]])

        out = [np.zeros((2, 1)), np.zeros((2, 1))]
        normal = [np.zeros((2, 1)), np.zeros((2, 1))]
        # if the tip is "above" the rectangle, it most likely intersects with
        # edge cd
        if tip[1] > d[1]:
            out = intersects_line(c, d, tip, tip_r, tip_r2)
            normal = [np.array([[0], [-1]]), np.array([[0], [-1]])]
        # if it is right of the rectangle, it most likely intersects with
        # edge bd
        if np.linalg.norm(out) == 0 and tip[0] > d[0]:
            normal = [np.array([[-1], [0]]), np.array([[-1], [0]])]
            out = intersects_line(b, d, tip, tip_r, tip_r2)
        # if it is below the triangle, it most likely intersects with edge ab
        if np.linalg.norm(out) == 0 and tip[1] < a[1]:
            out = intersects_line(a, b, tip, tip_r, tip_r2)
            normal = [np.array([[0], [1]]), np.array([[0], [1]])]
        # if it is left of the triangle, it most likely intersects with edge ca
        if np.linalg.norm(out) == 0 and tip[0] < a[0]:
            normal = [np.array([[1], [0]]), np.array([[1], [0]])]
            out = intersects_line(c, a, tip, tip_r, tip_r2)

        if debug:
            ax.add_patch(p.Rectangle((-np.linalg.norm(c-d)/2.,
                                      -np.linalg.norm(c-a)/2.),
                                     np.linalg.norm(c-d), np.linalg.norm(c-a),
                                     alpha=0.2, facecolor='b', edgecolor='b'))
    if 'tri' in ob:
        # b ----- a
        #         |
        #         |
        #         c
        # get the positions of the points
        if '1' in ob:
            a = np.array([[0.045], [0.045]])
            b = np.array([[-0.0809], [0.045]])
            c = np.array([[0.045], [-0.08087]])
        if '2' in ob:
            a = np.array([[0.045], [0.045]])
            b = np.array([[-0.106], [0.045]])
            c = np.array([[0.045], [-0.08087]])
        if '3' in ob:
            a = np.array([[0.045], [0.045]])
            b = np.array([[-0.1315], [0.045]])
            c = np.array([[0.045], [-0.08061]])

        out = [np.zeros((2, 1)), np.zeros((2, 1))]
        normal = [np.zeros((2, 1)), np.zeros((2, 1))]
        # if the tip is "above" the triangle, it most likely intersects with
        # edge ab
        if tip[1] > a[1]:
            out = intersects_line(a, b, tip, tip_r, tip_r2)
            normal = [np.array([[0], [-1]]), np.array([[0], [-1]])]
        # if it is right of the triangle, it it most likely intersects with
        # edge ac
        if np.linalg.norm(out) == 0 and tip[0] > a[0]:
            normal = [np.array([[-1], [0]]), np.array([[-1], [0]])]
            out = intersects_line(a, c, tip, tip_r, tip_r2)
        if np.linalg.norm(out) == 0:
            out = intersects_line(b, c, tip, tip_r, tip_r2)
            # get the slope of the normal (negative slope of bc)
            m = - (c[0] - b[0]) / (c[1] - b[1])
            # normal points to the object centre -> positive x direction
            normal = [np.array([[1], [m]]), np.array([[1], [m]])]

        if debug:
            ax.add_patch(p.Polygon([a.reshape(2), b.reshape(2), c.reshape(2)],
                                   alpha=0.2, facecolor='b', edgecolor='b'))
    if 'ellip' in ob:
        if '1' in ob:
            a = 0.0525
            b = 0.0525
        elif '2' in ob:
            a = 0.0525
            b = 0.065445
        elif '3' in ob:
            a = 0.0525
            b = 0.0785
        out = intersects_ellipse(ob, a, b, tip, tip_r, tip_r2)
        normal1 = np.zeros((2, 1), dtype=np.float32)
        # if we found a contact point for the smaller radius, calculate the
        # normal
        if np.linalg.norm(out[0]) > 0:
            # get the slope of the normal at this point
            if not out[0][0] == 0:
                m = (a**2 * out[0][1]) / (b**2 * out[0][0])
                # normal points to the object centre (which is at (0, 0))
                if out[0][0] < 0:
                    normal1 = np.array([[1], [m]])
                else:
                    normal1 = np.array([[-1], [-m]])
            else:
                # if the x-coordinate of the contact point is zero, the normal
                # is parallel to the y-axis
                if out[0][1] < 0:
                    normal1 = np.array([[0], [1]])
                else:
                    normal1 = np.array([[0], [-1]])

        # same for the bigger tip-radius
        normal2 = np.zeros((2, 1), dtype=np.float32)
        if np.linalg.norm(out[1]) > 0:
            if not out[1][0] == 0:
                m = (a**2 * out[1][1]) / (b**2 * out[1][0])
                if out[1][0] < 0:
                    normal2 = np.array([[1], [m]])
                else:
                    normal2 = np.array([[-1], [-m]])
            else:
                if out[1][1] < 0:
                    normal2 = np.array([[0], [1]])
                else:
                    normal2 = np.array([[0], [-1]])

        normal = [normal1, normal2]
        if debug:
            ax.add_patch(p.Ellipse((0., 0.), 2*a, 2*b, 0,
                                   alpha=0.2, facecolor='b', edgecolor='b'))
    if 'hex' in ob:
        # get the corners of the hexagon
        points = []
        for i in range(6):
            theta = (np.pi/3)*i
            points += [np.array([0.06050*np.cos(theta),
                                 0.06050*np.sin(theta)])]
        # add the first point at the end again
        points += [points[0]]

        normal = [np.zeros((2, 1), dtype=np.float32),
                  np.zeros((2, 1), dtype=np.float32)]
        # we test the edges sequentially
        for i in np.arange(0, len(points)-1):
            start = points[i]
            end = points[i+1]

            out = intersects_line(start, end, tip, tip_r, tip_r2)
            # if we got a contact point for the smaller radius, calculate
            # the normal
            if np.any(out[0]) != 0:
                # get the slope of the normal at this point
                if np.abs(end[0] - start[0]) < 1e-6:
                    # the line runs vertically, so the normal is horizontal
                    if out[0][0] < 0:
                        normal[0] = np.array([[1], [0]])
                    else:
                        normal[0] = np.array([[-1], [0]])
                elif np.abs(end[1] - start[1]) < 1e-6:
                    # the line runs horizontally, so the normal is vertical
                    if out[0][1] < 0:
                        normal[0] = np.array([[0], [1]])
                    else:
                        normal[0] = np.array([[0], [-1]])
                else:
                    if (start[0] > end[0]):
                        m = - (start[0] - end[0]) / (start[1] - end[1])
                    else:
                        m = - (end[0] - start[0]) / (end[1] - start[1])
                    # the normal points to the object centre
                    if out[0][0] < 0:
                        normal[0] = np.array([[1], [m]])
                    else:
                        normal[0] = np.array([[-1], [-m]])
            # same for the bigger tip radius
            if np.any(out[1]) != 0:
                if np.abs(end[0] - start[0]) < 1e-6:
                    if out[1][0] < 0:
                        normal[1] = np.array([[1], [0]])
                    else:
                        normal[1] = np.array([[-1], [0]])
                elif np.abs(end[1] - start[1]) < 1e-6:
                    if out[1][1] < 0:
                        normal[1] = np.array([[0], [1]])
                    else:
                        normal[1] = np.array([[0], [-1]])
                else:
                    if (start[0] > end[0]):
                        m = - (start[0] - end[0]) / (start[1] - end[1])
                    else:
                        m = - (end[0] - start[0]) / (end[1] - start[1])
                    if out[1][0] < 0:
                        normal[1] = np.array([[1], [m]])
                    else:
                        normal[1] = np.array([[-1], [-m]])
            if np.any(out[0] != 0) or np.any(out[1] != 0):
                break

        if debug:
            ax.add_patch(p.Polygon(points, alpha=0.2,
                                   facecolor='b', edgecolor='b'))

    if 'butter' in ob:
        # the butterfly-shaped object is defined by a large number of points
        # that we load from file
        points = pickle.load('../../resource/butter_pickle.pkl')
        points = np.array(points)

        # since the edges between the points are very short, the tip might
        # intersect with many neighbouring segments, and we take the average of
        # all contact points and normals we found
        tmp_out = []
        tmp_norm = []
        # to save time, we only use every fourth point and approximate the
        # shape around it by a streigth line throuh two neighbouring points
        for i in np.arange(0, len(points) - 2, 4):
            start = points[i-2]
            end = points[i+2]

            out = intersects_line(start, end, tip, tip_r, tip_r2)
            if np.any(out[0]) != 0 or np.any(out[1]) != 0:
                tmp_out += [out]
                normal = [np.zeros((2, 1), dtype=np.float32),
                          np.zeros((2, 1), dtype=np.float32)]
                # if we got a contact point for the smaller radius, calculate
                # the normal
                if np.any(out[0]) != 0:
                    # get the slope of the normal at this point
                    if end[1] == start[1]:
                        # the line runs vertially, so the normal is horizontal
                        if out[0][0] < 0:
                            normal[0] = np.array([[1], [0]])
                        else:
                            normal[0] = np.array([[-1], [0]])
                    elif end[0] == start[0]:
                        # the line runs horizontally, so the normal is vertical
                        if out[0][1] < 0:
                            normal[0] = np.array([[0], [1]])
                        else:
                            normal[0] = np.array([[0], [-1]])
                    else:
                        if (start[0] > end[0]):
                            m = - (start[0] - end[0]) / (start[1] - end[1])
                        else:
                            m = - (end[0] - start[0]) / (end[1] - start[1])
                        # the normal points to the object centre
                        if out[0][0] < 0:
                            normal[0] = np.array([[1], [m]])
                        else:
                            normal[0] = np.array([[-1], [-m]])
                # same for the bigger tip radius
                if np.any(out[1]) != 0:
                    if end[1] == start[1]:
                        if out[1][0] < 0:
                            normal[1] = np.array([[1], [0]])
                        else:
                            normal[1] = np.array([[-1], [0]])
                    elif end[0] == start[0]:
                        if out[1][1] < 0:
                            normal[1] = np.array([[0], [1]])
                        else:
                            normal[1] = np.array([[0], [-1]])
                    else:
                        if (start[0] > end[0]):
                            m = - (start[0] - end[0]) / (start[1] - end[1])
                        else:
                            m = - (end[0] - start[0]) / (end[1] - start[1])
                        if out[1][0] < 0:
                            normal[1] = np.array([[1], [m]])
                        else:
                            normal[1] = np.array([[-1], [-m]])
                tmp_norm += [normal]
        # average over the results in tmp
        out = [np.zeros((2, 1), dtype=np.float32),
               np.zeros((2, 1), dtype=np.float32)]
        normal = [np.zeros((2, 1), dtype=np.float32),
                  np.zeros((2, 1), dtype=np.float32)]
        count0 = 0
        count1 = 0
        for ind, el in enumerate(tmp_out):
            if np.any(el[0]) != 0:
                count0 += 1
                out[0] += el[0]
                normal[0] += tmp_norm[ind][0]
            if np.any(el[1]) != 0:
                count1 += 1
                out[1] += el[1]
                normal[1] += tmp_norm[ind][1]
        if count0 != 0:
            out[0] /= count0
            normal[0] /= count0
        if count1 != 0:
            out[1] /= count1
            normal[1] /= count1

        if debug:
            ax.add_patch(p.Polygon(points, alpha=0.2,
                                   facecolor='b', edgecolor='b'))

    if debug:
        # plot the contact point and the normal
        if not np.all(out[0]) == 0:
            ax.plot(out[0][0], out[0][1], 'mo')
            ax.plot([out[0][0], out[0][0] + normal[0][0]],
                    [out[0][1], out[0][1] + normal[0][1]], 'm')
        if not np.all(out[1]) == 0:
            ax.plot(out[1][0], out[1][1], 'bx')
            ax.plot([out[1][0], out[1][0] + normal[1][0]],
                    [out[1][1], out[1][1] + normal[1][1]], '--b')

    # redo the rotation and translation to transfer the contact point
    # and normal from the object's coordinate frame int the global frame
    # first rotate everything relative to the object
    out[0] = np.dot(rot_mat, out[0])
    out[1] = np.dot(rot_mat, out[1])
    normal[0] = np.dot(rot_mat, normal[0])
    normal[1] = np.dot(rot_mat, normal[1])

    # then undo the translation for the contact point
    if not np.all(out[0]) == 0:
        out[0] += pos
    if not np.all(out[1]) == 0:
        out[1] += pos

    # for debugging, we plot the first 10 files for each object
    if debug:
        path = os.path.join(debug_dir, ob, name)
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, str(image_num) + '.jpg')
        fig.savefig(file_path)
        plt.close()
    return out, normal


def intersects_line(p1, p2, tip, tip_r, tip_r2):
    """
    Calculates the intersection of a line segment and two circles
    (representing the tip) with different radii
    Args:
        p1: start point of the line segment
        p2: end point of the line segment
        tip: position of the circles (one for both)
        tip_r: first radius
        tip_r2: second (bigger) radius
    """
    # move everything such that the tip is in the centre
    p1 = p1.reshape(2, 1)
    p2 = p2.reshape(2, 1)
    l1 = p1 - tip
    l2 = p2 - tip

    # get the line equation
    if not l1[0] == l2[0]:
        if (l1[0] > l2[0]):
            m_val = (l1[1] - l2[1]) / (l1[0] - l2[0])
        elif l1[0] < l2[0]:
            m_val = (l2[1] - l1[1]) / (l2[0] - l1[0])

        c_val = l1[1] - l1[0] * m_val

        # plug in the values in the symbolic sympy equations defined at the
        # beginnign
        vals = [(c, c_val), (m, m_val), (r, tip_r)]
        c1 = np.zeros((2, 1), dtype=np.float32)
        c2 = np.zeros((2, 1), dtype=np.float32)
        c1x = expr_line_circle_x[0].subs(vals)
        c2x = expr_line_circle_x[1].subs(vals)
        c1y = expr_line_circle_y[0].subs(vals)
        c2y = expr_line_circle_y[1].subs(vals)
        if not c1x.is_Boolean and c1x.is_real:
            c1[0] = c1x.evalf()
            c1[1] = c1y.evalf()
        if not c2x.is_Boolean and c2x.is_real:
            c2[0] = c2x.evalf()
            c2[1] = c2y.evalf()

        # most of the time, the circle will intersect the line in two places,
        # and the real contact point is in the middle between them
        contact1 = (c1 + c2) / 2
        if np.any(contact1) != 0:
            contact1 += tip

        # try again with the bigger tip radius
        vals = [(c, c_val), (m, m_val), (r, tip_r2)]
        c1 = np.zeros((2, 1), dtype=np.float32)
        c2 = np.zeros((2, 1), dtype=np.float32)
        c1x = expr_line_circle_x[0].subs(vals)
        c2x = expr_line_circle_x[1].subs(vals)
        c1y = expr_line_circle_y[0].subs(vals)
        c2y = expr_line_circle_y[1].subs(vals)
        if not c1x.is_Boolean and c1x.is_real:
            c1[0] = c1x.evalf()
            c1[1] = c1y.evalf()
        if not c2x.is_Boolean and c2x.is_real:
            c2[0] = c2x.evalf()
            c2[1] = c2y.evalf()

        contact2 = (c1 + c2) / 2
        if np.any(contact2) != 0:
            contact2 += tip
    else:
        # the line runs vertically -> x is given and we really don't need
        # sympy here
        x_val = l1[0]
        contact1 = np.zeros((2, 1), dtype=np.float32)
        # if there are one or two intersections, the y coordinate of the
        # (average) contact point is at y=0 (relative to the tip)
        if np.abs(x_val) <= tip_r:
            contact1[0] = x_val
            contact1[1] = 0
        if np.any(contact1) != 0:
            contact1 += tip

        # again with a slightly bigger probe
        contact2 = np.zeros((2, 1), dtype=np.float32)
        if np.abs(x_val) <= tip_r2:
            contact2[0] = x_val
            contact2[1] = 0
        if np.any(contact2) != 0:
            contact2 += tip

    # if an intersection was found, we need to make sure that it lies on the
    # line segment
    if np.any(contact1) != 0:
        if contact1[0, 0] > max(p1[0, 0], p2[0, 0]) + 1e-5 \
                or contact1[0, 0] < min(p1[0, 0], p2[0, 0]) - 1e-5 \
                or contact1[1, 0] > max(p1[1, 0], p2[1, 0]) + 1e-5 \
                or contact1[1, 0] < min(p1[1, 0], p2[1, 0]) - 1e-5:
            contact1 = np.zeros((2, 1), dtype=np.float32)
    if np.any(contact2) != 0:
        if contact2[0] > max(p1[0], p2[0]) + 1e-5 \
                or contact2[0] < min(p1[0], p2[0]) - 1e-5 \
                or contact2[1] > max(p1[1], p2[1]) + 1e-5 \
                or contact2[1] < min(p1[1], p2[1]) - 1e-5:
            contact2 = np.zeros((2, 1), dtype=np.float32)

    return [contact1, contact2]


def intersects_ellipse(ob, major, minor, tip, tip_r, tip_r2):
    """
    Calculates the intersection of an ellipse and two circles
    (representing the tip) with different radii
    Args:
        major: major axis of the ellipse
        minor: minor axis of the ellipse
        tip: position of the circles (one for both)
        tip_r: first radius
        tip_r2: second (bigger) radius
    """
    # check if there is any chance of contact
    if tip[0] + tip_r2 < - major:
        # the tip's maximum x is below the object's minimum x
        return [np.zeros((2, 1), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32)]
    elif tip[0] - tip_r2 > major:
        # the tip's minimum x is above the object's maximum x
        return [np.zeros((2, 1), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32)]
    elif tip[1] + tip_r2 < - minor:
        # the tip's maximum y is below the object's minimum y
        return [np.zeros((2, 1), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32)]
    elif tip[1] - tip_r2 > minor:
        # the tip's minimum y is above the object's maximum y
        return [np.zeros((2, 1), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32)]

    # try with the bigger probe
    tp = Circle(Point(tip[0], tip[1]), tip_r2)
    if '1' in ob:
        cps = ellip1.intersection(tp)
    elif '2' in ob:
        cps = ellip2.intersection(tp)
    elif '3' in ob:
        cps = ellip3.intersection(tp)
    count = 0.
    avg = np.zeros((2, 1), dtype=np.float32)
    for cp in cps:
        c = np.zeros((2, 1), dtype=np.float32)
        c[0] = cp.x
        c[1] = cp.y
        count += 1
        avg += c

    if count != 0:
        contact2 = avg / count
    else:
        contact2 = avg

    # only check the smaller one if the bigger radius had a contact
    if np.linalg.norm(contact2) > 0:
        tp = Circle(Point(tip[0], tip[1]), tip_r)
        if '1' in ob:
            cps = ellip1.intersection(tp)
        elif '2' in ob:
            cps = ellip2.intersection(tp)
        elif '3' in ob:
            cps = ellip3.intersection(tp)
        count = 0.
        avg = np.zeros((2, 1), dtype=np.float32)
        for cp in cps:
            c = np.zeros((2, 1), dtype=np.float32)
            c[0] = cp.x
            c[1] = cp.y
            count += 1
            avg += c

        if count != 0:
            contact1 = avg / count
        else:
            contact1 = avg
    else:
        contact1 = np.zeros((2, 1), dtype=np.float32)

    return [contact1, contact2]


def main(argv=None):
    parser = argparse.ArgumentParser('annotate')
    parser.add_argument('--source-dir', dest='source_dir', type=str,
                        required=True,
                        help='Directory holding the preprocessed MIT Push ' +
                            'dataset.')
    parser.add_argument('--out-dir', dest='out_dir', type=str,
                        help='Where to store results.')
    parser.add_argument('--num-jobs', dest='num_jobs', type=int,
                        default=10,
                        help='Number of files to process in parallel.')
    parser.add_argument('--visualize', dest='visualize', type=int,
                        default=1, choices=[0, 1],
                        help='Visualize some files?')
    parser.add_argument('--criteria', dest='criteria', type=str, nargs='*',
                        default=[],
                        help='Limit annotations to datafiles containing the ' +
                            'strings in this list. E.g. criteria=[a=0] will ' +
                            'only consider files with acceleration (a) = 0')

    args = parser.parse_args(argv)

    get_contact_annotation(args.source_dir, args.out_dir, args.num_jobs,
                           args.visualize, args.criteria)


if __name__ == "__main__":
    main()
