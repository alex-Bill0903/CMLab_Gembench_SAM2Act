import os
from typing import Type
import numpy as np

from pathlib import Path
from typing import Tuple, Dict, List
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor


class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy, speed: float):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians

    def step(self):
        self.origin.rotate([0, 0, self.speed])


class StaticCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor):
        super().__init__(cam)

    def step(self):
        pass

class AttachedCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, parent_cam: VisionSensor):
        super().__init__(cam)
        self.parent_cam = parent_cam

    def step(self):
        self.cam.set_pose(self.parent_cam.get_pose())


class TaskRecorder(object):

    def __init__(self, cams_motion: Dict[str, CameraMotion], fps=30):
        self._cams_motion = cams_motion
        self._fps = fps
        self._snaps = {cam_name: [] for cam_name in self._cams_motion.keys()}

    def take_snap(self):
        for cam_name, cam_motion in self._cams_motion.items():
            cam_motion.step()
            self._snaps[cam_name].append(
                (cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))
    # original code
    # def save(self, path):
    #     print('Converting to video ...')
    #     path = Path(path)
    #     path.mkdir(exist_ok=True)
    #     # OpenCV QT version can conflict with PyRep, so import here
    #     import cv2
    #     for cam_name, cam_motion in self._cams_motion.items():
    #         video = cv2.VideoWriter(
    #                 str(path / f"{cam_name}.avi"), cv2.VideoWriter_fourcc(*'mp4v'), self._fps,
    #                 tuple(cam_motion.cam.get_resolution()))
    #         for image in self._snaps[cam_name]:
    #             video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #         video.release()

    #     self._snaps = {cam_name: [] for cam_name in self._cams_motion.keys()}
    
    def save(self, path):
        import shutil
        import cv2
        import uuid
        print('Converting to video ...')
        path = Path(path)
        path.mkdir(exist_ok=True)

        for cam_name, cam_motion in self._cams_motion.items():
            images = self._snaps[cam_name]
            resolution = tuple(cam_motion.cam.get_resolution())
            tmp_folder = path / f"tmp_{uuid.uuid4().hex}"
            tmp_folder.mkdir(exist_ok=True)

            # 儲存成一系列暫存圖片
            for idx, image in enumerate(images):
                # 轉成 BGR 給 OpenCV 寫檔
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(tmp_folder / f"{idx:04d}.png"), img_bgr)

            # 用 ffmpeg 轉成 mp4
            output_mp4 = str(path / f"{cam_name}.mp4")
            os.system(
                f"ffmpeg -y -framerate {self._fps} -i {tmp_folder}/%04d.png "
                f"-c:v libx264 -pix_fmt yuv420p {output_mp4} -hide_banner -loglevel error"
            )

            # 刪除暫存圖片
            shutil.rmtree(tmp_folder)

        # 重設 snaps
        self._snaps = {cam_name: [] for cam_name in self._cams_motion.keys()}
    
    
 
