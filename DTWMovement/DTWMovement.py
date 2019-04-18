import pandas as pd
import numpy as np
import math
import fastdtw as fdtw
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches


def xyz_plotter(mvmnt_list, x_axis, y_axis, unit):
    """
    Plots xyz-values (Location, Velocity or Acceleration) of movement.

    :param mvmnt_list: 2d list of values of multiple movements
    :param x_axis: x-axis of plot
    :param y_axis: y-axis of plot
    :param unit: specified unit of movement list
    """
    plt.plot(mvmnt_list[unit][x_axis], mvmnt_list[unit][y_axis + '_x'], color='red')
    plt.plot(mvmnt_list[unit][x_axis], mvmnt_list[unit][y_axis + '_y'], color='green')
    plt.plot(mvmnt_list[unit][x_axis], mvmnt_list[unit][y_axis + '_z'], color='blue')
    plt.title("Unit {}'s axis {} over {}".format(unit, y_axis, x_axis))
    x_patch = mpatches.Patch(color='red', label=y_axis + ' x')
    y_patch = mpatches.Patch(color='green', label=y_axis + ' y')
    z_patch = mpatches.Patch(color='blue', label=y_axis + ' z')
    plt.legend(handles=[x_patch, y_patch, z_patch])
    plt.show()


def true_vel_accel_calculator(df):
    """
    Calculates the axis independent velocity and acceleration through the theorem of pythagoras.

    :param df: Dataframe to be calculated.
    :return: Dataframe with additional vel and accel columns
    """
    true_velocities = []
    for i in range(len(df.index)):
        true_velocity = np.sqrt(math.fabs(df.iat[i, 0]) ** 2 + math.fabs(df.iat[i, 1]) ** 2)
        true_velocities.append(true_velocity)

    series_true_velocity = pd.Series(true_velocities)

    true_accelerations = []
    for i in range(len(df.index)):
        true_acceleration = np.sqrt(math.fabs(df.iat[i, 3]) ** 2 + math.fabs(df.iat[i, 4]) ** 2)
        true_accelerations.append(true_acceleration)

    series_true_acceleration = pd.Series(true_accelerations)

    df = pd.concat([df, series_true_velocity, series_true_acceleration], axis=1, sort=False)
    df = df.rename(columns={0: 'vel_true', 1: 'acc_true'})

    return df


def mvmnt_splitter(df):
    """
    Splits a dataframe of multiple movements in singular ones and saves them in a list.

    :param df: Dataframe with movement data
    :return: List of movement units
    """
    _ = df.index[df['tick'] == 1].tolist()
    mvmnt_units = []

    for i in range(len(_)):
        if i == len(_) - 1:
            temp_df = df.loc[_[i]:len(df.index)]
        else:
            temp_df = df.loc[_[i]:_[i + 1] - 1]

        mvmnt_units.append(temp_df)

    return mvmnt_units


def distance_calculator(mvmnt_units, ref_unit):
    """
    Calculates distance of given movement unit to an given reference unit.

    :param mvmnt_units:
    :param ref_unit:
    :return:
    """

    distance_list = []
    for i in range(len(mvmnt_units) - 1):
        distance, path = fdtw.fastdtw(mvmnt_units[i], ref_unit)
        distance_list.append(distance)

    return np.mean(distance_list)


def movement_classifier(distances_list, test_name, threshold):
    """
    Classifies the move by the lowest value and a threshold.

    :param distances_list: List of dtw calculated distances of different movements
    :param test_name: name of the reference movement
    """
    best_fit = min(distances_list)
    if best_fit[0] < threshold:
        print("Movement is a {} with the lowest distance value of {}".format(test_name, best_fit[0]))
    else:
        print("No movement fits the {} movement, the distances were as follows".format(test_name))
        for i in range(len(distances_list)):
            print("Distance to {} movement: {}".format(distances_list[i][1], distances_list[i][0]))


def comparator(mvmnt_comparators, test_df, mvmnt_names, test_name, threshold=2500):
    """
    Uses all functions to compare different movements to a reference movement. mvmnt_names can be left empty, only
    important for a better overview on how the algorithm works.

    :param mvmnt_comparators: List of dataframes with movement values
    :param test_df: Dataframe with reference movement values
    :param mvmnt_names: Names of the comparators
    :param test_name: Name of the reference movement
    """
    comparator_units = []
    for i in range(len(mvmnt_comparators)):
        temp_units = mvmnt_splitter(mvmnt_comparators[i])
        comparator_units.append(temp_units)

    comparator_distances = []
    for i in range(len(mvmnt_comparators)):
        temp_distance = distance_calculator(comparator_units[i], test_df)
        comparator_distances.append(temp_distance)

    if mvmnt_names is not None:
        comparator_distances_list = []
        for i in range(len(mvmnt_comparators)):
            temp_unit = [comparator_distances[i], mvmnt_names[i]]
            comparator_distances_list.append(temp_unit)

        movement_classifier(comparator_distances_list, test_name, threshold)
    else:
        movement_classifier(comparator_distances, test_name, threshold)


"""For testing let threshold variable stay at 0, otherwise change to desired value (e.g. 2500)"""
threshold = 0

fall_df = pd.read_csv("Movements/fall_movement.csv")
jump_df = pd.read_csv("Movements/jump_movement.csv")
run_df = pd.read_csv("Movements/run_movement.csv")
slope_down_df = pd.read_csv("Movements/slope_run_movement_down.csv")          # running movement
slope_up_df = pd.read_csv("Movements/slope_run_movement_up.csv")
test_fall_df = pd.read_csv("Movements/test_fall_movement.csv")
test_jump_df = pd.read_csv("Movements/test_jump_movement.csv")
test_run_df = pd.read_csv("Movements/test_run_movement.csv")

"""Commented part is for plotting, num can be changed to the desired movement-unit of the df"""
# mvmnt_units = mvmnt_splitter(fall_df)
# num = 0
# xyz_plotter(mvmnt_units, 't_total', 'loc', num)
# xyz_plotter(mvmnt_units, 't_total', 'vel', num)
# xyz_plotter(mvmnt_units, 't_total', 'acc', num)
#
# mvmnt_units = mvmnt_splitter(jump_df)
# xyz_plotter(mvmnt_units, 't_total', 'loc', num)
# xyz_plotter(mvmnt_units, 't_total', 'vel', num)
# xyz_plotter(mvmnt_units, 't_total', 'acc', num)
#
# mvmnt_units = mvmnt_splitter(run_df)
# xyz_plotter(mvmnt_units, 't_total', 'loc', num)
# xyz_plotter(mvmnt_units, 't_total', 'vel', num)
# xyz_plotter(mvmnt_units, 't_total', 'acc', num)

"""Drops the location columns, since they are not needed"""
fall_df = fall_df.drop(columns=['loc_x', 'loc_y', 'loc_z'])
jump_df = jump_df.drop(columns=['loc_x', 'loc_y', 'loc_z'])
run_df = run_df.drop(columns=['loc_x', 'loc_y', 'loc_z'])
test_fall_df = test_fall_df.drop(columns=['loc_x', 'loc_y', 'loc_z'])
test_jump_df = test_jump_df.drop(columns=['loc_x', 'loc_y', 'loc_z'])
test_run_df = test_run_df.drop(columns=['loc_x', 'loc_y', 'loc_z'])
slope_down_df = slope_down_df.drop(columns=['loc_x', 'loc_y', 'loc_z'])
slope_up_df = slope_up_df.drop(columns=['loc_x', 'loc_y', 'loc_z'])

"""Beginning of movement comparisons to different reference movements"""
print("Comparison to fall movement:")

fall_to_fall_df = fall_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
jump_to_fall_df = jump_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
run_to_fall_df = run_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
slope_down_to_fall_df = slope_down_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
slope_up_fall_df = slope_up_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
test_fall_df = test_fall_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])


comparator(mvmnt_comparators=[fall_to_fall_df, jump_to_fall_df, run_to_fall_df, slope_down_to_fall_df, slope_up_fall_df],
           test_df=test_fall_df, mvmnt_names=["fall", "jump", "run", "slope down", "slope up"], test_name="fall",
           threshold=threshold)

print("\nComparison to jump movement:")

fall_to_jump_df = fall_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
jump_to_jump_df = jump_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
run_to_jump_df = run_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
slope_down_jump_df = slope_down_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
slope_up_to_jump_df = slope_up_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
test_jump_df = test_jump_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])

comparator(mvmnt_comparators=[fall_to_jump_df, jump_to_jump_df, run_to_jump_df, slope_down_to_fall_df, slope_up_fall_df],
           test_df=test_jump_df, mvmnt_names=["fall", "jump", "run", "slope down", "slope up"], test_name="jump",
           threshold=threshold)

print("\nComparison to run movement:")

# TODO: For some reason, the calculated true velocity/acceleration dramatically skews the results
# fall_to_run_df = true_vel_accel_calculator(fall_df)
# jump_to_run_df = true_vel_accel_calculator(jump_df)
# run_to_run_df = true_vel_accel_calculator(run_df)
# test_run_df = true_vel_accel_calculator(test_run_df)

# Since that is the case, I tested if running up or down a slope can still be classified as running, when choosing the columns I choose
fall_to_run_df = fall_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
jump_to_run_df = jump_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
run_to_run_df = run_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
slope_down_to_run_df = slope_down_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
slope_up_to_run_df = slope_up_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])
test_run_df = test_run_df.drop(columns=['acc_x', 'acc_y', 'vel_x', 'vel_y'])

comparator(mvmnt_comparators=[fall_to_run_df, jump_to_run_df, run_to_run_df, slope_down_to_run_df, slope_up_to_run_df],
           test_df=test_run_df, mvmnt_names=["fall", "jump", "run", "slope down", "slope up"], test_name="run",
           threshold=threshold)



