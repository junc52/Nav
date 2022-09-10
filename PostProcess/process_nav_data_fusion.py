import argparse
import pandas as pd
from PostProcess.engine.system_navigation import NavSys


def run_nav_data_fusion(args):
    # load INS measurements
    if args.ins:
        df_meas_ins = pd.read_csv(args.ins_file)

    # instantiate navigation system
    nav_system = NavSys(args, df_meas_ins)

    # compute navigation solution
    nav_system.compute_nav_solution()



if __name__ == '__main__':
    """
    Peform post processing of navigation data fusion algorithm
    - Read navigation data to estimate navigation solution
    - Compute navigation solution
    - Plot parameters of interest set by users
    
    Author : Jun Choi
    Open source code for Android GNSS, INS measurements 
    """
    parser = argparse.ArgumentParser(description='Process Nav Data Fusion Algorithm')
    parser.add_argument("ins_file", help="add ins nav data file name")
    parser.add_argument("-t", "--truth", help="add truth data file name")
    parser.add_argument("-i", "--ins", help="process inertial navigation component", action="store_true")

    args = parser.parse_args()

    run_nav_data_fusion(args)



