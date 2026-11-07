from keyboard import KeyboardPose
from tag import TagDetector
from util import testingWebcam

testingWebcam(KeyboardPose("calibration_stores/calibration_robot_cam.pckl", "keyboard_data/demo.json").testingProcessor, True, 1)

# recordTest
# testingWebcam(lambda x: x, True, 1, True)
