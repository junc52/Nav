INS_MESSAGE_TYPE = ['UncalAccel', 'UncalGyro', 'UncalMag']


class NavIns:
    def __init__(self, args, meas):
        self.args_ = args
        self.acc_meas_ = meas[meas['MessageType'] == INS_MESSAGE_TYPE[0]]
        self.gyr_meas_ = meas[meas['MessageType'] == INS_MESSAGE_TYPE[1]]
        self.mag_meas_ = meas[meas['MessageType'] == INS_MESSAGE_TYPE[2]]


