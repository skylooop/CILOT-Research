import imageio
import os
from gymnasium.utils.save_video import save_video

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = root_dir
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
                # height=self.height,
                # width=self.width
            )
            self.frames.append(frame)

    def save(self, file_name, fps):
        save_video(
                self.frames,
                file_name,
                fps=fps,
            )
        # if self.enabled:
        #     path = os.path.join(self.save_dir, file_name)
        #     imageio.mimsave(path, self.frames, fps=self.fps)
