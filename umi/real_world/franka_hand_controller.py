import time
import multiprocessing as mp
import zerorpc

class FrankaHandController(mp.Process):
    def __init__(self, host="129.97.71.27", port=4242, speed=0.05, force=20.0, update_rate=30):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.speed = speed
        self.force = force
        self.update_dt = 1.0 / update_rate
        self.command_queue = mp.Queue()
        self._stop_event = mp.Event()

    def get_state(self):
        client = zerorpc.Client()
        client.connect(f"tcp://{self.host}:{self.port}")
        state = client.get_gripper_state()
        return state
        
    def send_target(self, width):
        self.command_queue.put(width)

    def stop(self):
        self._stop_event.set()
        self.join()
        print("[GripperStreamer] Stopped.")

    def run(self):
        client = zerorpc.Client()
        client.connect(f"tcp://{self.host}:{self.port}")
        print("[GripperStreamer] Connected to gripper RPC.")

        last_width = None

        while not self._stop_event.is_set():
            try:
                while True:
                    width = self.command_queue.get_nowait()
                    last_width = width
            except Exception:
                pass

            if last_width is not None:
                print(f"[Streaming] Sending width: {last_width:.3f}")
                client.gripper_goto(last_width, self.speed, self.force)

            time.sleep(self.update_dt)


# ========== Example usage ==========
if __name__ == "__main__":
    streamer = FrankaHandController()
    streamer.start()

    try:
        print("State:", streamer.get_state()['timestamp']['seconds'])
        widths = [0.08, 0.06, 0.07, 0.02, 0.05, 0.07, 0.1]
        for w in widths:
            streamer.send_target(w)
            time.sleep(1.5)
    finally:
        streamer.stop()


# Test client
# client = zerorpc.Client()
# client.connect("tcp://129.97.71.27:4242")

# for _ in range(30):  # adjust loop count or use a time-based stop
# 		state_dict = client.get_gripper_state()
# 		print("Gripper State:", state_dict)
# 		time.sleep(0.05)