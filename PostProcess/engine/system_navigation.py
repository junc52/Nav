from PostProcess.engine.inertial_navigation import NavIns


class NavSys:
    def __init__(self, args, meas_ins):
        self.args_ = args
        self.nav_ins_ = NavIns(args, meas_ins)

    def compute_nav_solution(self):
        pass
