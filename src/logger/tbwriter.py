import os
from datetime import datetime
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        log_dir = os.path.join("tensorboard_logs", run_name if run_name else "run")
        
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard initialized. Logs will be saved to: {os.path.abspath(log_dir)}")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

        try:
            config_str = yaml.dump(project_config, default_flow_style=False)
            self.writer.add_text("Config", f"```yaml\n{config_str}\n```", 0)
        except Exception as e:
            logger.warning(f"Failed to log config to TensorBoard: {e}")

    def set_step(self, step, mode="train"):
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            if duration.total_seconds() > 0:
                self.add_scalar(
                    "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
                )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        return f"{self.mode}/{object_name}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        pass

    def add_scalar(self, scalar_name, scalar):
        self.writer.add_scalar(self._object_name(scalar_name), scalar, self.step)

    def add_scalars(self, scalars):
        for scalar_name, scalar in scalars.items():
            self.add_scalar(scalar_name, scalar)

    def add_image(self, image_name, image):
        if hasattr(image, 'detach'):
            image = image.detach().cpu().numpy()
        self.writer.add_image(self._object_name(image_name), image, self.step)

    def add_audio(self, audio_name, audio, sample_rate=22050):
        if hasattr(audio, 'detach'):
            audio = audio.detach().cpu().numpy()

        audio = np.squeeze(audio)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        if sample_rate is None:
            sample_rate = 22050

        self.writer.add_audio(
            self._object_name(audio_name),
            audio,
            self.step,
            sample_rate=sample_rate,
        )

    def add_text(self, text_name, text):
        self.writer.add_text(self._object_name(text_name), text, self.step)

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        if hasattr(values_for_hist, 'detach'):
            values_for_hist = values_for_hist.detach().cpu().numpy()
            
        self.writer.add_histogram(self._object_name(hist_name), values_for_hist, self.step)

    def add_table(self, table_name, table: pd.DataFrame):
        md_table = table.to_markdown()
        self.writer.add_text(self._object_name(table_name), md_table, self.step)

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()
