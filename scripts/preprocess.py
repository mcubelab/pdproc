#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:35:54 2016

@author: Alina Kloss
    code mostly copied from
    https://github.com/peterkty/pnpush/blob/master/catkin_ws/src/pnpush_planning/src/analyze/extract_data_training.py

    This code preprocesses the hdf5 files of the pushing dataset.

    Input:
        source_dir: Top-level directory with one subfolder for each
            surface material. In each surface folder, there needs to be one
            folder for each object with zip archive containing hdf5 files.
            For example ~/pd/abs/rect1/rect1_h5.zip. You should get this
            structure automatically when you download (and unzip) the archives
            for the different surface materials into the same folder.
            Here, source_dir is ~/pd.
        out_dir: Optional, output preprocessed data to a different location
        frequency: Desired frequency for resampling the data during
            synchronization, default: 180 Hz

    Output:
        Preprocesses the data by:
            - removing redundant data entries
            - treating some (not all) of the jumps in object orientation
            - transforming the orientation to [-pi, pi]
            - synchronizing the data by resampling to a given frequency
            - setting the initial object position and orientation to zero
            - adding information about the push (angle, velocity...) to the
                h5 datafile
        The data is saved to out_dir in the format [surface]/[object]/[file].h5
        The unzipped archives in source_dir are removed after preprocessing.
        A list with all datafiles that contain errors or large jumps in the
        object's orientation is saved to out_dir/error.log
"""

import os
import argparse
import sys
import numpy as np
import h5py
import pandas as pd
import zipfile
import shutil
import logging
import csv


class Preprocess:
    def __init__(self, args):
        # setup logging
        self.log = logging.getLogger('data_parser')
        self.log.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s: [%(name)s] ' +
                                      '[%(levelname)s] %(message)s')
        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.log.addHandler(ch)
        # create file handler which logs warnings errors and criticals
        if os.path.exists(os.path.join(args.source_dir, 'error.log')):
            os.remove(os.path.join(args.source_dir, 'error.log'))
        fh = logging.FileHandler(os.path.join(args.source_dir, 'error.log'))
        fh.setLevel(logging.WARNING)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)

        self.source_dir = args.source_dir
        if not os.path.exists(self.source_dir):
            raise ValueError('source directory does not exist: ' +
                             self.source_dir)

        # if no output directory is supplied, we operate in place
        if args.out_dir is not None:
            self.out_dir = args.out_dir
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
                self.log.info('Creating output directory ' + self.out_dir)
        else:
            self.out_dir = args.source_dir
            self.log.info('Writing output to source directory')

    def preprocess_data(self, dt='5556us'):
        surfaces = [os.path.join(self.source_dir, f)
                    for f in os.listdir(self.source_dir) if
                    os.path.isdir(os.path.join(self.source_dir, f))]
        for s in surfaces:
            self.log.info(s)
            surface_name = os.path.basename(s)

            # object folders
            objects = [os.path.join(s, f) for f in os.listdir(s) if
                       os.path.isdir(os.path.join(s, f))]

            # each object folder contains zipped json, h5 and png files
            # we want the h5 files
            datafiles = []
            for o in objects:
                datafiles += [os.path.join(o, f) for f in os.listdir(o) if
                              f.endswith('zip') and 'h5' in f]

            # unzip the object folders into the surface-directory
            for d in datafiles:
                data_name = os.path.basename(d)
                data_name = data_name[:data_name.find('.')]
                # see if the folder has already been unzipped, if not do so
                if not os.path.exists(os.path.join(s, data_name)):
                    self.log.debug('Extracting ' + d + ' to ' +
                                   os.path.join(s, data_name))
                    zip_ref = zipfile.ZipFile(d, 'r')
                    zip_ref.extractall(os.path.join(s, data_name))
                    zip_ref.close()

            # this time collect the unzipped files
            datafiles = [os.path.join(s, f) for f in os.listdir(s)
                         if os.path.isdir(os.path.join(s, f)) and
                         f.endswith('h5')]

            surface_out = os.path.join(self.out_dir, surface_name)
            if not os.path.exists(surface_out):
                os.mkdir(surface_out)

            for d in datafiles:
                # each directory corresponds to one object type
                data_name = os.path.basename(d)
                object_name = data_name[:data_name.find('_')]

                # make a new directory for the object if necessary
                target = os.path.join(surface_out, object_name)
                if not os.path.exists(target):
                    shutil.os.makedirs(target)

                # get all the datafiles
                filenames = [os.path.join(d, f) for f in os.listdir(d) if
                             f.endswith('h5')]
                # account for different folder structures
                if filenames == []:
                    dirname = [os.path.join(d, f) for f in os.listdir(d) if
                               os.path.isdir(os.path.join(d, f))]
                    for di in dirname:
                        filenames += [os.path.join(di, f)
                                      for f in os.listdir(di) if
                                      f.endswith('h5')]

                self.log.info('Preprocessing data for ' + d)
                for i, f in enumerate(filenames):
                    # get the push velocity
                    v_ind = f.find('_v=') + 3
                    vel = int(float(f[v_ind:f.find('_', v_ind)]))

                    # read and preprocess the data
                    data = self._process_data(f, vel, dt)
                    if data is None:
                        continue

                    # we collect the information from the file-name and save
                    # it into the h5 structure
                    a_ind = f.find('_a=') + 3
                    acc = int(float(f[a_ind:f.find('_', a_ind)]))
                    data['acc'] = acc

                    data['push_velocity'] = vel

                    t_ind = f.find('_t=') + 3
                    ang = float(f[t_ind:f.find('.h5')])
                    data['push_angle'] = ang * 180 / np.pi

                    i_ind = f.find('_i=') + 3
                    side = int(float(f[i_ind:f.find('_', i_ind)]))
                    data['push_side'] = side

                    s_ind = f.find('_s=') + 3
                    point = float(f[s_ind:f.find('_', s_ind)])
                    data['push_point'] = point

                    # save file
                    name = os.path.basename(f)
                    if os.path.exists(os.path.join(target, name)):
                        os.remove(os.path.join(target, name))
                    # I'm shortening the filename by object and surface,
                    # since this info is already in the folder structure
                    with h5py.File(os.path.join(target,
                                                name[name.find('_a=')+1:]),
                                   "w") as h5:
                        for key, val in data.iteritems():
                            h5.create_dataset(key, data=val)

                # remove the unzipped folder to save diskspace
                shutil.rmtree(d)

    def _process_data(self, filename, vel, dt):
        try:
            data_h5 = h5py.File(filename, "r", driver='core')
        except:
            self.log.exception("message")
            return None

        # get rid of redundant entries and ensuring temporal ordering
        data = self._order_data(data_h5, filename)
        data_h5.close()
        if data is None:
            return None
        # treat orientation jumps, limit the range of orientation values
        # to [-pi, pi]
        # TODO this is pretty hacky
        data = self._correct_orientation(data, filename, vel)
        if data is None:
            return None

        # the data is not synchronized, so we need to resample
        # this step also discards the timestamps
        data = self._resample(data, filename, dt)
        if data is None:
            return None

        # set initial orientation and position to zero
        data = self._zero_initial_position(data)

        return data

    def _order_data(self, data, filename):
        tip_pose = data['tip_pose']
        object_pose = data['object_pose']
        force = data['ft_wrench']

        # checking for redundant entries and ensuring correct sorting of data
        object_pose_2d = []
        tip_pose_2d = []
        for i in (range(0, len(object_pose), 1)):
            time = object_pose[i][0]
            # don't add redundant data entry with the same time
            if not (len(object_pose_2d) > 0 and time == object_pose_2d[-1][0]):
                pose = object_pose[i][1:3]
                ang = object_pose[i][3]
                object_pose_2d.append([time] + pose.tolist() + [ang])
        # sort the object poses by time
        object_pose_2d.sort(key=lambda x: x[0])

        # same for the tips
        for i in (range(0, len(tip_pose), 1)):
            time = tip_pose[i][0]
            # don't add redundant data entry with the same time
            if(not(len(tip_pose_2d) > 0 and time == tip_pose_2d[-1][0])):
                pose = tip_pose[i][1:3]
                ang = tip_pose[i][3]
                tip_pose_2d.append([time] + pose.tolist() + [ang])
        tip_pose_2d.sort(key=lambda x: x[0])

        # ft, no redundency
        ft_2d = np.array(force).tolist()
        ft_2d.sort(key=lambda x: x[0])

        # all data needs to have the same length
        if len(ft_2d) < 2 or len(tip_pose_2d) < 2 or len(object_pose_2d) < 2:
            self.log.error('File: ' + os.path.basename(filename) +
                           ': [Missing data]:  force: ' +
                           str(len(ft_2d)) + ', object:' +
                           str(len(object_pose_2d)) + ', tip:' +
                           str(len(tip_pose_2d)))
            return None
        else:
            data = {"object": object_pose_2d, "tip": tip_pose_2d,
                    'force': ft_2d}

            return data

    def _correct_orientation(self, data, filename, vel):
        # correct for jumping angles
        last = 0.
        last_t = 0
        # this indicates if an uncorrected jump in orientation occured and is
        # used to avoid logging the jump multiple times for the same file
        blacklisted = False
        for ind, dat in enumerate(data['object']):
            a = dat[-1]
            t = dat[0]

            diff = last - a
            dt = t - last_t
            # define a threshold for the maximum allowed orientation jump,
            # depending on the time to the last measurement and the velocity
            # of the pusher, minimum 8 degree, maximum 15 degree
            thresh = self._get_threshold(dt, vel)

            if ind != 0 and np.abs(diff) > thresh:
                # a jump by 2 pi is no problem due to the periodicity
                if np.abs(np.abs(diff) - (2 * np.pi)) > thresh:
                    # if the jump is close to a multiple of pi, it's a tracking
                    # error and we correct it
                    if np.abs(diff - np.pi/2) < thresh:  # 90
                        data['object'][ind][-1] = a + np.pi/2
                    elif np.abs(diff + np.pi/2) < thresh:  # -90
                        data['object'][ind][-1] = a - np.pi/2
                    elif np.abs(diff - np.pi) < thresh:  # 180
                        data['object'][ind][-1] = a + np.pi
                    elif np.abs(diff + np.pi) < thresh:  # -180
                        data['object'][ind][-1] = a - np.pi
                    elif np.abs(diff - 3*np.pi/2) < thresh:  # 270
                        data['object'][ind][-1] = a + 3*np.pi/2
                    elif np.abs(diff + 3*np.pi/2) < thresh:  # - 270
                        data['object'][ind][-1] = a - 3*np.pi/2
                    elif ind != len(data['object']) - 1:
                        # if the jump does not fit with the above, it could be
                        # streched over multiple timesteps, so we look ahead a
                        # bit to see if the jump gets bigger
                        next_ind = ind + 1
                        next_diff = a - data['object'][next_ind][-1]
                        next_dt = data['object'][next_ind][0] - t
                        next_thresh = self._get_threshold(next_dt, vel)
                        while next_ind + 1 < len(data['object']) and \
                                np.abs(next_diff) > next_thresh and \
                                np.sign(next_diff) == np.sign(diff):
                            next_ind += 1
                            next_diff = data['object'][next_ind-1][-1] - \
                                data['object'][next_ind][-1]
                            next_dt = data['object'][next_ind][0] - \
                                data['object'][next_ind-1][0]
                            next_thresh = self._get_threshold(next_dt, vel)

                        # next_ind is either the end of the file or the first
                        # index where the jump did not increase in magnitude
                        # anymore. In the latter case, we subtract 1 to get the
                        # index of the jump's end
                        if next_ind != len(data['object'])-1 or \
                                (next_ind == len(data['object'])-1 and
                                 (a - data['object'][next_ind][-1]) < thresh):
                            next_ind -= 1

                        if next_ind > ind:
                            # correct the full-value jump for the last index
                            corrected = True
                            total_diff = a - data['object'][next_ind][-1]
                            if np.abs(total_diff - np.pi/2) < thresh:  # 90
                                data['object'][next_ind][-1] = \
                                    data['object'][next_ind][-1] + np.pi/2
                            elif np.abs(total_diff + np.pi/2) < thresh:  # -90
                                data['object'][next_ind][-1] = \
                                    data['object'][next_ind][-1] - np.pi/2
                            elif np.abs(total_diff - np.pi) < thresh:  # 180
                                data['object'][next_ind][-1] = \
                                    data['object'][next_ind][-1] + np.pi
                            elif np.abs(total_diff + np.pi) < thresh:  # -180
                                data['object'][next_ind][-1] = \
                                    data['object'][next_ind][-1] - np.pi
                            elif np.abs(total_diff - 3*np.pi/2) < thresh:  # 270
                                data['object'][next_ind][-1] = \
                                    data['object'][next_ind][-1] + 3*np.pi/2
                            elif np.abs(total_diff + 3*np.pi/2) < thresh:  # - 270
                                data['object'][next_ind][-1] = \
                                    data['object'][next_ind][-1] - 3*np.pi/2
                            else:
                                corrected = False

                            if not corrected and not blacklisted:
                                error = 'Large orientation jump: ' + \
                                    'difference in degree: ' + \
                                    str(180 * total_diff / np.pi) + \
                                    ', step: ' + str(ind)
                                self.log.error('File: ' +
                                               os.path.basename(filename) +
                                               ': ' + error)
                                blacklisted = True
                            else:
                                self.log.debug('correcting longer jump: ' +
                                               'File: ' +
                                               os.path.basename(filename))
                                # correct the other values by linearly fitting
                                # the total angle jump
                                # get the slope of the orientation
                                m = (data['object'][next_ind][-1] - last) / \
                                    (data['object'][next_ind][0] - last_t)
                                for l in np.arange(ind, next_ind):
                                    dt_l = data['object'][l][0] - last_t
                                    data['object'][l][-1] = last + dt_l * m
                        else:
                            if not blacklisted:
                                error = ': Large orientation jump: ' + \
                                    'difference in degree: ' + \
                                    str(180*diff/np.pi) + ', step: ' + str(ind)
                                self.log.error('File: ' +
                                               os.path.basename(filename) +
                                               error)
                                blacklisted = True
                    elif not blacklisted:
                        self.log.debug('not in pi and eof')
                        error = 'Large orientation jump: ' + \
                            'difference in degree: ' + \
                            str(180*diff/np.pi) + ', step: ' + str(ind)
                        self.log.error('File: ' + os.path.basename(filename) +
                                       ': ' + error)
                        blacklisted = True

            last = data['object'][ind][-1]
            last_t = t

        # limit the range of the orientation to [-pi, pi]
        for ind, dat in enumerate(data['object']):
            a = dat[-1]
            # correcting orientation-values above 360 degree
            if np.abs(a) > 2 * np.pi:
                self.log.warning('too big angle: ' + str(a) + ' -> ' +
                                 str(a % (2 * np.pi)))
                a = a % (2 * np.pi)
            # limit range
            if a < -np.pi:
                a += 2 * np.pi
            elif a > np.pi:
                a -= 2 * np.pi
            data['object'][ind][-1] = a

        return data

    def _get_threshold(self, dt, vel):
        thresh = max(min(0.25, 0.14 * (dt/0.004)), 0.07)
        if vel > 75 and vel < 200:
            thresh *= 1.25
        elif vel >= 200:
            thresh *= 1.5
        elif vel < 0:
            thresh *= 1.5
        return thresh

    def _resample(self, data, f, interval):
        object_poses_2d = data['object']
        tip_poses_2d = data['tip']
        force_2d = data['force']

        starttime = max(tip_poses_2d[0][0], object_poses_2d[0][0],
                        force_2d[0][0])
        endtime = min(tip_poses_2d[-1][0], object_poses_2d[-1][0],
                      force_2d[-1][0])

        pd_starttime = pd.to_datetime(starttime, unit='s')
        pd_endtime = pd.to_datetime(endtime, unit='s')
        if pd_endtime <= pd_starttime:
            self.log.error('File: ' + os.path.basename(f) +
                           ': [Incompatible timestamps]')
            return None

        # the number of datapoints between start and end should be
        # approximately the same in each recording
        o_in = filter(lambda x: x[0] > starttime and x[0] < endtime,
                      object_poses_2d)
        t_in = filter(lambda x: x[0] > starttime and x[0] < endtime,
                      object_poses_2d)
        f_in = filter(lambda x: x[0] > starttime and x[0] < endtime,
                      object_poses_2d)
        if np.abs(len(o_in) - len(f_in)) > 20 or  \
                np.abs(len(f_in) - len(t_in)) > 20 or \
                np.abs(len(t_in) - len(o_in)) > 20:
            error = 'Unbalanced data:  force: ' + str(len(f_in)) + \
                ', object:' + str(len(o_in)) + ', tip:' + str(len(t_in)) + \
                ' timesteps between start and end'
            self.log.error('File: ' + os.path.basename(f) + ': ' + error)

        tip_poses_2d_dt = pd.to_datetime(np.array(tip_poses_2d)[:, 0].tolist(),
                                         unit='s')
        tip_poses_2d = pd.DataFrame(np.array(tip_poses_2d)[:, 1:3].tolist(),
                                    index=tip_poses_2d_dt)
        tip_poses_2d_resampled = tip_poses_2d.resample(interval, how='mean')
        tip_poses_2d_interp = tip_poses_2d_resampled.interpolate()

        start_ = tip_poses_2d_interp.index.searchsorted(pd_starttime)
        end_ = tip_poses_2d_interp.index.searchsorted(pd_endtime)
        tip_poses_2d_interp = tip_poses_2d_interp.ix[start_:end_]
        tip_poses_2d_interp_list = tip_poses_2d_interp.values.tolist()

        object_poses_2d_dt = \
            pd.to_datetime(np.array(object_poses_2d)[:, 0].tolist(), unit='s')
        object_poses_2d = \
            pd.DataFrame(np.array(object_poses_2d)[:, 1:4].tolist(),
                         index=object_poses_2d_dt)
        object_poses_2d_resampled = object_poses_2d.resample(interval,
                                                             how='mean')
        object_poses_2d_interp = object_poses_2d_resampled.interpolate()
        start_ = object_poses_2d_interp.index.searchsorted(pd_starttime)
        end_ = object_poses_2d_interp.index.searchsorted(pd_endtime)
        object_poses_2d_interp = object_poses_2d_interp.ix[start_:end_]
        object_poses_2d_interp_list = object_poses_2d_interp.values.tolist()

        force_dt = pd.to_datetime(np.array(force_2d)[:, 0].tolist(), unit='s')
        force_2d = pd.DataFrame(np.array(force_2d)[:, 1:4].tolist(),
                                index=force_dt)
        force_2d_resampled = force_2d.resample(interval, how='mean')
        force_2d_interp = force_2d_resampled.interpolate()
        start_ = force_2d_interp.index.searchsorted(pd_starttime)
        end_ = force_2d_interp.index.searchsorted(pd_endtime)
        force_2d_interp = force_2d_interp.ix[start_:end_]
        force_2d_interp_list = force_2d_interp.values.tolist()

        data_resample = {}
        data_resample['tip'] = tip_poses_2d_interp_list
        data_resample['object'] = object_poses_2d_interp_list
        data_resample['force'] = force_2d_interp_list

        if len(data_resample['tip']) == 0 or \
                len(data_resample['object']) == 0 or \
                len(data_resample['force']) == 0:
            self.log.error('File: ' + f + ': [Resampling failed]')
            return None
        else:
            return data_resample

    def _zero_initial_position(self, data):
        num = len(data['object'])
        new_data = {'object': np.zeros((num, 3), dtype=np.float32),
                    'tip': np.zeros((num, 2), dtype=np.float32),
                    'force': np.zeros((num, 3), dtype=np.float32)}
        # set the initial object position to and oreinatation to zero. This way
        # it can be changed easier when rendering images
        init_ob_pose = np.copy(np.array(data['object'][0]))
        ang = init_ob_pose[2]

        # create a rotation matrix to yero the object's rotaion
        rot = np.zeros((2, 2))
        rot[0, 0] = np.cos(-ang)
        rot[0, 1] = -np.sin(-ang)
        rot[1, 0] = np.sin(-ang)
        rot[1, 1] = np.cos(-ang)

        for i, pos in enumerate(data['object']):
            # get the vector from initial position to object position
            p = np.array(pos[:2]) - init_ob_pose[:2]
            # rotate the object
            p = np.dot(rot, p)
            # subtract the initial orientation
            a = (pos[2] - ang)

            # this could take the orientation out of [-pi, pi] again, so we
            # readjust
            if a < -np.pi:
                a += 2 * np.pi
            elif a > np.pi:
                a -= 2 * np.pi
            assert (a >= -np.pi and a <= np.pi), \
                ('after zero: ' + str(a) + ' ' + str(ang) + ' ' + str(pos[2]))
            new_data['object'][i, :2] = p[:2].astype(np.float32)
            new_data['object'][i, 2] = a.astype(np.float32)

        # as we rotate the object around the initial pose, we also have to
        # rotate the tip position around it
        for i, pos in enumerate(data['tip']):
            # get the vector from initial position to tip position
            p = np.array(pos) - init_ob_pose[:2]
            # rotate
            p = np.dot(rot, p)
            new_data['tip'][i, :] = p[:2].astype(np.float32)

        # rotate the force vectors
        for i, f in enumerate(data['force']):
            # get the linear force vector
            p = np.array(f)[:2]
            # rotate
            p = np.dot(rot, p)
            out = np.array([p[0], p[1], np.array(f)[2]])
            new_data['force'][i, :] = out.astype(np.float32)

        return new_data


def main(argv=None):
    parser = argparse.ArgumentParser('preprocess')
    parser.add_argument('--source-dir', dest='source_dir', type=str,
                        required=True,
                        help='Directory holding the MIT Push dataset. The ' +
                            'archives for the surfaces must be extracted ' +
                            'manually, the script will take care of the rest.')
    parser.add_argument('--out-dir', dest='out_dir', type=str,
                        help='Where to store results. If omitted, results ' +
                            'will be stored into the source directory.')
    parser.add_argument('--frequency', dest='frequency', type=int,
                        default='180',
                        help='Desired data-frequency after synchronizing.')
    args = parser.parse_args(argv)

    # turn the frequency in a time interval string for pandas
    dt = 1. / args.frequency
    dt *= 1000000
    dt = str(int(np.round(dt))) + 'us'

    pre = Preprocess(args)
    pre.preprocess_data(dt)


if __name__ == "__main__":
    main()
