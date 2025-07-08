import zerorpc
from polymetis import GripperInterface
import scipy.spatial.transform as st

class FrankaInterface:
    def __init__(self):
        self.gripper = GripperInterface('localhost')

    # Gripper methods
    def get_gripper_state(self):
        gripper_state = self.gripper.get_state()
        gripper_state_dict = {
            "timestamp": {"seconds": gripper_state.timestamp.seconds, "nanoseconds": gripper_state.timestamp.nanos},
            "width": gripper_state.width
        }
        return gripper_state_dict
    
    def gripper_goto(self, width, speed, force):
        self.gripper.goto(width, speed, force)
    
    def gripper_grasp(self, speed, force, grasp_width):
        self.gripper.grasp(speed, force, grasp_width)


s = zerorpc.Server(FrankaInterface())
s.bind("tcp://0.0.0.0:4242")
s.run()