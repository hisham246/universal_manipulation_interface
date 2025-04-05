import time
import multiprocessing as mp
import zerorpc
import numpy as np

class FrankaHandController(mp.Process):
    def __init__(self, host="129.97.71.27", port=4242, speed=0.05, force=20.0, update_rate=30):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.speed = speed
        self.force = force
        self.update_dt = 1.0 / update_rate
        self.command_queue = mp.Queue()
        self.ready_event = mp.Event()
        self._stop_event = mp.Event()
        self._client = None
        self._connected = False

    def connect(self):
        if not self._connected:
            try:
                self._client = zerorpc.Client()
                self._client.connect(f"tcp://{self.host}:{self.port}")
                self._connected = True
                print("[FrankaHandController] Connected to gripper RPC.")
            except Exception as e:
                print(f"[FrankaHandController] Failed to connect: {e}")
                self._connected = False

    def get_state(self):
        self.connect()
        try:
            return self._client.get_gripper_state()
        except Exception as e:
            print(f"[FrankaHandController] Error getting gripper state: {e}")
            return {'width': 0.0}

    def get_all_state(self):
        state = self.get_state()
        try:
            timestamp = state['timestamp']['seconds'] + state['timestamp']['nanoseconds'] * 1e-9
        except KeyError:
            timestamp = time.time()

        return {
            'gripper_position': [state['width']],
            'gripper_timestamp': [timestamp],
        }
    
    def schedule_waypoint(self, pos):
        if isinstance(pos, np.ndarray):
            pos = float(pos)
        self.send_target(pos)

    def send_target(self, width):
        self.command_queue.put(width)

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True):
        super().start()
        if wait:
            time.sleep(0.5)

    def stop(self, wait=True):
        self._stop_event.set()
        if wait:
            self.join()
        print("[FrankaHandController] Stopped.")

    def start_wait(self):
        pass

    def stop_wait(self):
        pass

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self):
        self.stop()

    def run(self):
        self.connect()
        if not self._connected:
            print("[FrankaHandController] Failed to start gripper RPC.")
            return

        self.ready_event.set()
        print("[FrankaHandController] Ready and running.")

        last_width = None

        while not self._stop_event.is_set():
            try:
                # Flush queue to get latest command
                while True:
                    width = self.command_queue.get_nowait()
                    last_width = width
            except mp.queues.Empty:
                pass  # No new command, proceed with last one

            if last_width is not None:
                try:
                    # print(f"[FrankaHandController] Streaming width: {last_width:.3f}")
                    self._client.gripper_goto(last_width, self.speed, self.force)
                except Exception as e:
                    # print(f"[FrankaHandController] RPC error during gripper_goto: {e}")
                    self._connected = False
                    self.ready_event.clear()
                    break  # Exit loop or try reconnecting optionally

            time.sleep(self.update_dt)

        print("[FrankaHandController] Exiting run loop.")

# import time
# import multiprocessing as mp
# import zerorpc

# class FrankaHandController(mp.Process):
#     def __init__(self, host="129.97.71.27", port=4242, speed=0.05, force=20.0, update_rate=30):
#         super().__init__(daemon=True)
#         self.host = host
#         self.port = port
#         self.speed = speed
#         self.force = force
#         self.update_dt = 1.0 / update_rate
#         self.command_queue = mp.Queue()
#         self._stop_event = mp.Event()

#     def get_state(self):
#         client = zerorpc.Client()
#         client.connect(f"tcp://{self.host}:{self.port}")
#         state = client.get_gripper_state()
#         return state
        
#     def send_target(self, width):
#         self.command_queue.put(width)

#     def stop(self):
#         self._stop_event.set()
#         self.join()
#         print("[GripperStreamer] Stopped.")

#     def run(self):
#         client = zerorpc.Client()
#         client.connect(f"tcp://{self.host}:{self.port}")
#         print("[GripperStreamer] Connected to gripper RPC.")

#         last_width = None

#         while not self._stop_event.is_set():
#             try:
#                 while True:
#                     width = self.command_queue.get_nowait()
#                     last_width = width
#             except Exception:
#                 pass

#             if last_width is not None:
#                 print(f"[Streaming] Sending width: {last_width:.3f}")
#                 client.gripper_goto(last_width, self.speed, self.force)

#             time.sleep(self.update_dt)


# # ========== Example usage ==========
# if __name__ == "__main__":
#     streamer = FrankaHandController()
#     streamer.start()

#     try:
#         print("State:", streamer.get_state()['timestamp']['seconds'])
#         widths = [0.08, 0.06, 0.07, 0.02, 0.05, 0.07, 0.1]
#         for w in widths:
#             streamer.send_target(w)
#             time.sleep(1.5)
#     finally:
#         streamer.stop()


# Test client
# client = zerorpc.Client()
# client.connect("tcp://129.97.71.27:4242")

# for _ in range(30):  # adjust loop count or use a time-based stop
# 		state_dict = client.get_gripper_state()
# 		print("Gripper State:", state_dict)
# 		time.sleep(0.05)