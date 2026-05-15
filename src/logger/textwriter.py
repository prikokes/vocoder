from datetime import datetime

import numpy as np
import pandas as pd


class TextWriter:
    """
    Class for experiment tracking via CometML.

    See https://www.comet.com/docs/v2/.
    """

    def __init__(
        self,
        logger,
        *args,
        **kwargs,
    ):
        self.writer = logger
        pass


    def set_step(self, step, mode="train"):
        pass
    def _object_name(self, object_name):
        pass

    def add_checkpoint(self, checkpoint_path, save_dir):
        pass

    def add_scalar(self, scalar_name, scalar):
        pass

    def add_scalars(self, scalars):
        pass

    def add_image(self, image_name, image):
        pass

    def add_audio(self, audio_name, audio, sample_rate=None):
        pass

    def add_text(self, text_name, text):
        pass

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        pass

    def add_table(self, table_name, table: pd.DataFrame):
        pass

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()
