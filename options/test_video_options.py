"""This script contains the test options for Deep3DFaceRecon_pytorch
"""

from .base_options import BaseOptions


class TestVideoOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
        parser.add_argument('--video_folder', type=str, default=None, help='cropped video folder.')
        parser.add_argument('--openface_folder', type=str, default=None, help='openface csv folder.')
        parser.add_argument('--output_folder', type=str, default=None, help='output folder.')
        parser.add_argument('--n_parts', type=int, default=1, help='number of parts to split data')
        parser.add_argument('--part_id', type=int, default=0, help='part id to process')
        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
