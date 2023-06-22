import numpy as np
import sensor_msgs.msg

from rosbag_utils.export_hdf5 import main, reader


# some readers are already defined ...
# define readers for every message type you want to import
# like here below. Each reader should take the corresponding message as input
# and ouputs a numpy array

@reader(sensor_msgs.msg.BatteryState)
def battery_state(msg: sensor_msgs.msg.BatteryState) -> np.ndarray:
    return np.array([msg.voltage, msg.percentage])


main()
