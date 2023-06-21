import cv2

class Camera(object):
    _instance = None
    @staticmethod
    def get_instance():
        if Camera._instance is None:
            Camera._instance = Camera()
        return Camera._instance
    def __init__(self):
        if Camera._instance != None:
            raise Exception("This class is a singleton!")
        else:
            Camera._instance = self
            self.video = cv2.VideoCapture(0, cv2.CAP_V4L)
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_FPS, 30)
            self.grabbed, self.frame = self.video.read()
    def __del__(self):
        self.video.release()

    def get_frame(self):
        image=self.frame
        _, jpeg=cv2.imencode('.jpg',image)
        return jpeg.tobytes()
    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

    def capture(self):
        return self.frame
    
    def is_opened(self):
        return self.video.isOpened()