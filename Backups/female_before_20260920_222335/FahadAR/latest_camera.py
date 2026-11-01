"""Drain camera continuously; inference consumes only the most recent frame."""
import threading,queue,time

class LatestCamera:
    def __init__(self,cap):
        self.cap=cap;self.q=queue.Queue(maxsize=1);self.running=True;self.error=None
        self.thread=threading.Thread(target=self._read,daemon=True,name='latest_camera')
        self.thread.start()
    def _read(self):
        while self.running:
            ok,frame=self.cap.read();stamp=time.time()
            if not ok:self.error=RuntimeError('Camera stopped returning frames');break
            try:self.q.get_nowait()
            except queue.Empty:pass
            try:self.q.put_nowait((frame,stamp))
            except queue.Full:pass
    def read(self):
        try:return self.q.get(timeout=2)
        except queue.Empty:raise self.error or RuntimeError('No fresh camera frame')
    def close(self):
        self.running=False;self.thread.join(timeout=.4)
        self.cap.release();self.thread.join(timeout=.4)
