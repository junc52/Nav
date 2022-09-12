from PostProcess.engine.inertial_navigation import NavIns, NavInsEci

# TODO : refactoring for message type and data frame columns
INS_MESSAGE_TYPE = ['UncalAccel', 'UncalGyro', 'UncalMag']
INS_COLS = ['MessageType', 'utcTimeMillis','MeasurementX', 'MeasurementY', 'MeasurementZ', 'BiasX', 'BiasY', 'BiasZ']

class NavSys:
    def __init__(self, args, meas_ins):
        self._args = args
        self._acc_meas = meas_ins[meas_ins['MessageType'] == INS_MESSAGE_TYPE[0]].reset_index()
        self._gyr_meas = meas_ins[meas_ins['MessageType'] == INS_MESSAGE_TYPE[1]].reset_index()
        self._mag_meas = meas_ins[meas_ins['MessageType'] == INS_MESSAGE_TYPE[2]].reset_index()

        # TODO : instantiate inertial navigation object based on user selected coordinate system
        self._nav_ins = NavInsEci(args)

    def compute_nav_solution(self):
        # TODO : resolve synchronization of measurement time of validity
        no_epochs = min(self._acc_meas.shape[0], self._gyr_meas.shape[0])

        for i in range(no_epochs):
            # update measurement, measurement time
            self._nav_ins.f_ib_b = self._acc_meas.iloc[i][['MeasurementX', 'MeasurementY', 'MeasurementZ']]
            self._nav_ins.omega_ib_b = self._gyr_meas.iloc[i][['MeasurementX', 'MeasurementY', 'MeasurementZ']]
            self._nav_ins._utc_time_millis_a_ = self._acc_meas['utcTimeMillis'].iloc[i]
            self._nav_ins._utc_time_millis_g_ = self._gyr_meas['utcTimeMillis'].iloc[i]
            if i == 0 :
                self._nav_ins._last_utc_time_millis_a_ = self._acc_meas['utcTimeMillis'].iloc[i]
                self._nav_ins._last_utc_time_millis_g_ = self._gyr_meas['utcTimeMillis'].iloc[i]

            # update attitude
            self._nav_ins.update_attitude()


