from PostProcess.engine.inertial_navigation import NavIns, NavInsEci

# TODO : refactoring for message type and data frame columns
INS_MESSAGE_TYPE = ['UncalAccel', 'UncalGyro', 'UncalMag']
INS_COLS = ['MessageType', 'utcTimeMillis','MeasurementX', 'MeasurementY', 'MeasurementZ', 'BiasX', 'BiasY', 'BiasZ']

class NavSys:
    def __init__(self, args, meas_ins):
        self.args_ = args
        self.acc_meas_ = meas_ins[meas_ins['MessageType'] == INS_MESSAGE_TYPE[0]].reset_index()
        self.gyr_meas_ = meas_ins[meas_ins['MessageType'] == INS_MESSAGE_TYPE[1]].reset_index()
        self.mag_meas_ = meas_ins[meas_ins['MessageType'] == INS_MESSAGE_TYPE[2]].reset_index()

        # TODO : instantiate inertial navigation object based on user selected coordinate system
        self.nav_ins_ = NavInsEci(args)

    def compute_nav_solution(self):
        # TODO : resolve synchronization of measurement time of validity
        no_epochs = min(self.acc_meas_.shape[0], self.gyr_meas_.shape[0])

        for i in range(no_epochs):
            # update measurement, measurement time
            self.nav_ins_.f_ib_b = self.acc_meas_.iloc[i][['MeasurementX', 'MeasurementY', 'MeasurementZ']]
            self.nav_ins_.omega_ib_b = self.gyr_meas_.iloc[i][['MeasurementX', 'MeasurementY', 'MeasurementZ']]
            self.nav_ins_.utc_time_millis_a_ = self.acc_meas_['utcTimeMillis'].iloc[i]
            self.nav_ins_.utc_time_millis_g_ = self.gyr_meas_['utcTimeMillis'].iloc[i]
            if i == 0 :
                self.nav_ins_.last_utc_time_millis_a_ = self.acc_meas_['utcTimeMillis'].iloc[i]
                self.nav_ins_.last_utc_time_millis_g_ = self.gyr_meas_['utcTimeMillis'].iloc[i]

            # update attitude
            self.nav_ins_.update_attitude()


