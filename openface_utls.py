"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
Utls for handling OpenFace related stuffs.
"""
# pylint: disable=no-member, import-error, wrong-import-position
# pylint: disable=too-many-locals, too-many-arguments, unspecified-encoding
import numpy as np
import pandas

class OpenFaceCSVReader():
    """Reader for openface generated CSV files."""
    def __init__(self, csv_path, num_landmarks=68):
        self.num_landmarks = num_landmarks
        data_tmp = pandas.read_csv(csv_path)
        self._data = {}
        # A hack to remove the left space in the column names
        for _d in data_tmp:
            self._data[_d.lstrip()] = data_tmp[_d] # type: ignore

    def get_landmarks2d(self):
        """Get 2d landmarks."""
        landmarks = []
        for i in range(self.num_landmarks):
            lm_x = self._data['x_' + str(i)].to_list()
            lm_y = self._data['y_' + str(i)].to_list()
            lm_ = np.stack([lm_x, lm_y], axis=-1)
            landmarks.append(lm_)
        landmarks = np.array(landmarks)
        landmarks = np.transpose(landmarks, (1, 0, 2))
        landmarks = np.reshape(landmarks, (landmarks.shape[0], -1))
        return landmarks.astype(np.float32)

    # pylint: disable=invalid-name
    # from OpenFace: RotationHelpers.h Ln47, R = Rx * Ry * Rz
    @staticmethod
    def _rotation_xyz(x_rad, y_rad, z_rad):
        s1 = np.sin(x_rad)
        s2 = np.sin(y_rad)
        s3 = np.sin(z_rad)
        c1 = np.cos(x_rad)
        c2 = np.cos(y_rad)
        c3 = np.cos(z_rad)
        return np.array([
            [c2 * c3, -c2 * s3, s2],
            [c1 * s3 + c3 * s1 * s2,  c1 * c3 - s1 * s2 * s3,  -c2 * s1],
            [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]]).astype(np.float32)

    def get_landmarks3d(self):
        """Get 3d landmarks."""
        landmarks = []
        for i in range(self.num_landmarks):
            lm_x = self._data['X_' + str(i)].to_list()
            lm_y = self._data['Y_' + str(i)].to_list()
            lm_z = self._data['Z_' + str(i)].to_list()
            lm_ = np.stack([lm_x, lm_y, lm_z], axis=-1)
            landmarks.append(lm_)
        landmarks = np.array(landmarks)
        landmarks = np.transpose(landmarks, (1, 0, 2)).astype(np.float32)

        poses_t = [self._data['pose_Tx'].to_list(),
                   self._data['pose_Ty'].to_list(),
                   self._data['pose_Tz'].to_list()]
        poses_r = [self._data['pose_Rx'].to_list(),
                   self._data['pose_Ry'].to_list(),
                   self._data['pose_Rz'].to_list()]
        poses_t = np.stack(poses_t, axis=-1).astype(np.float32)
        poses_r = np.stack(poses_r, axis=-1).astype(np.float32)
        num_frames = poses_t.shape[0]
        landmarks_unposed = []
        for i in range(num_frames):
            _t = poses_t[i]
            _lm = landmarks[i]
            _lm = _lm - _t
            # rot_x = self._rotation_x(-poses_r[i, 0])
            # rot_y = self._rotation_y(-poses_r[i, 1])
            # rot_z = self._rotation_z(-poses_r[i, 2])
            # rot_0 = np.matmul(rot_z, np.matmul(rot_y, rot_x))
            rot_ = self._rotation_xyz(poses_r[i, 0], poses_r[i, 1], poses_r[i, 2])
            rot_ = np.linalg.inv(rot_)
            _lm = np.matmul(_lm, rot_.T)
            landmarks_unposed.append(_lm)
        landmarks_unposed = np.array(landmarks_unposed)

        landmarks = np.reshape(landmarks, (num_frames, -1))
        landmarks_unposed = np.reshape(landmarks_unposed, (num_frames, -1))
        return landmarks.astype(np.float32), landmarks_unposed.astype(np.float32)

    def get_confidence(self):
        """Get confidence."""
        conf = self._data['confidence'].to_list()
        return np.array(conf).astype(np.float32)

    def get_timestamp(self):
        """Get confidence."""
        conf = self._data['timestamp'].to_list()
        return np.array(conf).astype(np.float32)

    def get_head_pose(self):
        """Get head pose."""
        t_x = self._data['pose_Tx'].to_list()
        t_y = self._data['pose_Ty'].to_list()
        t_z = self._data['pose_Tz'].to_list()
        r_x = self._data['pose_Rx'].to_list()
        r_y = self._data['pose_Ry'].to_list()
        r_z = self._data['pose_Rz'].to_list()
        poses = np.stack([t_x, t_y, t_z, r_x, r_y, r_z], axis=-1)
        return poses.astype(np.float32)

    def get_landmarks2d_5pts(self):
        """Get 2d landmarks (5 pts)."""
        assert self.num_landmarks == 68
        landmarks = self.get_landmarks2d()
        landmarks = np.reshape(landmarks, (-1, 68, 2))
        lm_left_eye = np.mean(landmarks[:, 36:42, :], axis=1, keepdims=True)
        lm_right_eye = np.mean(landmarks[:, 42:48, :], axis=1, keepdims=True)
        lm_nose = landmarks[:, 30:31, :]
        lm_mouth_left = landmarks[:, 48:49, :]
        lm_mouth_right = landmarks[:, 54:55, :]
        landmarks = np.concatenate(
            [lm_left_eye, lm_right_eye, lm_nose,
             lm_mouth_left, lm_mouth_right],
            axis=1)
        return landmarks.astype(np.float32)
