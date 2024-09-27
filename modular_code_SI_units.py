# -*- coding: utf-8 -*-
"""
Same as 2d_meshgrid_test_2_with_neutrals.py, but using functions for the different parts to make it easier
"""
# import the necessary libraries and modules
import numpy as np
from scipy import interpolate
import cmath # imports a maths module that allows for complex numbers, so that maths errors don't occur
import matplotlib.pyplot as plt # imports matplotlib under the name plt
import math
from scipy import constants
from matplotlib import ticker, cm

# FONT SIZES
title = 25
label = 20
legend = 20
maj_ticks = 19
min_ticks = 17

def create_meshgrid():
    # set minimum and maximum x and y values
    xmin = -300
    xmax = 200
    ymin = -300
    ymax = 300

    # set the size of the step for the numpy arrays
    dx = 0.5
    dy = 0.5

    # make the x and y numpy arrays
    x_init = np.arange(xmin, xmax + 1, dx)
    y_init = np.arange(ymin, ymax + 1, dy)

    # make the meshgrid
    xbox, ybox = np.meshgrid(x_init, y_init)
    
    return xbox, ybox, xmin, xmax, dx, dy



def boundaries(xmin, dx):
    # calculate the x and y values for the bow shock and magnetopause
    '''THE CALCULATIONS ARE COPIED FROM more_efficient_arrays.py'''

    # create arrays for x-values
    x = np.arange(-200, 200, 0.5) # creates the array of initial x-values
    x_plot_vals = np.repeat(x, 2) # creates a new array with each of the initial x-values repeated twice, so that the graph can be plotted
    x_divided = x/120 # creates an array of the original x-values divided by 120, for the calculations to be made with

    # create arrays for the values of the dynamic pressure
    #bow_shock_pressure = [0.07, 0.315]
    #mag_pause_pressure = [0.039, 0.306]
    
    #bow_shock_pressure = [0.07, 0.315] #ignore these
    mag_pause_pressure = [0.021, 0.167] # new dynamic pressures for using in the phd paper

    a = 0 # variable that changes which conditions are being modelled: 0 = expanded, 1 = compressed

    # get the dynamic pressure values from the arrays:
    bs_dyn_pressure = mag_pause_pressure[a]
    mp_dyn_pressure = mag_pause_pressure[a]

    # array_format_functions = [A, B, C, D, E, F]
    bs_pressure_functions = [-1.107 + 1.591*(bs_dyn_pressure)**(-0.25), -0.566 - 0.812*(bs_dyn_pressure)**(-0.25), 0.048 - 0.059*(bs_dyn_pressure)**(-0.25), 0.077 - 0.038*(bs_dyn_pressure), -0.874 - 0.299*(bs_dyn_pressure), -0.055 + 0.124*(bs_dyn_pressure)]
    mp_pressure_functions = [-0.134 + 0.488*(mp_dyn_pressure)**(-0.25), -0.581 - 0.225*(mp_dyn_pressure)**(-0.25), -0.186 - 0.016*(mp_dyn_pressure)**(-0.25), -0.014 + 0.096*(mp_dyn_pressure), -0.814 - 0.811*(mp_dyn_pressure), -0.050 + 0.168*(mp_dyn_pressure)]

    # create empty arrays to store calculated y-values
    y_values_bs_dawn = []
    y_values_bs_dusk = []
    y_values_mp_dawn = []
    y_values_mp_dusk = []

    for i in x_divided:

        # calculate the "G" part of the equation:
        G_bs = bs_pressure_functions[0] + bs_pressure_functions[1] * i + bs_pressure_functions[2] * i**2
        G_mp = mp_pressure_functions[0] + mp_pressure_functions[1] * i + mp_pressure_functions[2] * i**2
        
        # calculate the "H" part of the equation:
        H_bs = bs_pressure_functions[5] * i
        H_mp = mp_pressure_functions[5] * i
        
        # set the coefficients of the quadratic:
        a_bs = bs_pressure_functions[4]
        b_bs = bs_pressure_functions[3] + H_bs
        c_bs = G_bs
        a_mp = mp_pressure_functions[4]
        b_mp = mp_pressure_functions[3] + H_mp
        c_mp = G_mp
        
        # calculate the sets of y values using the quadratic formula:
        y_bs_plus = ((-b_bs)+cmath.sqrt(b_bs**2 - 4*a_bs*c_bs))/(2*a_bs)
        y_bs_minus = ((-b_bs)-cmath.sqrt(b_bs**2 - 4*a_bs*c_bs))/(2*a_bs)
        y_mp_plus = ((-b_mp)+cmath.sqrt(b_mp**2 - 4*a_mp*c_mp))/(2*a_mp)
        y_mp_minus = ((-b_mp)-cmath.sqrt(b_mp**2 - 4*a_mp*c_mp))/(2*a_mp)
        
        # append the calculated y-values to the corresponding arrays
        y_values_bs_dawn.append(y_bs_plus) 
        y_values_bs_dusk.append(y_bs_minus)
        y_values_mp_dawn.append(y_mp_plus)
        y_values_mp_dusk.append(y_mp_minus)

    # convert the y-value arrays into numpy arrays    
    y_mp_dawn_divided = np.array(y_values_mp_dawn)
    y_mp_dusk_divided = np.array(y_values_mp_dusk)
    y_bs_dawn_divided = np.array(y_values_bs_dawn)
    y_bs_dusk_divided = np.array(y_values_bs_dusk)

    # multiply the calculated y-values by 120 to get the coordinates to plot
    y_mp_dawn = y_mp_dawn_divided*120
    y_mp_dusk = y_mp_dusk_divided*120
    y_bs_dawn = y_bs_dawn_divided*120
    y_bs_dusk = y_bs_dusk_divided*120

    # take only the real parts of the y-value arrays
    y_mp_dawn = np.real(y_mp_dawn)
    y_mp_dusk = np.real(y_mp_dusk)
    y_bs_dawn = np.real(y_bs_dawn)
    y_bs_dusk = np.real(y_bs_dusk)

    # make separate arrays for the bow shock and magnetopause x values (they are initially the same - might need to change this)
    x_mp = x
    x_bs = x

    # find the x-coordinate for the nose of the bow shock and magnetopause
    bs_nose_div = (- bs_pressure_functions[1] - cmath.sqrt((bs_pressure_functions[1] **2)-4*bs_pressure_functions[0]*bs_pressure_functions[2]))/(2 * bs_pressure_functions[2])
    mp_nose_div = (- mp_pressure_functions[1] - cmath.sqrt((mp_pressure_functions[1] **2)-4*mp_pressure_functions[0]*mp_pressure_functions[2]))/(2 * mp_pressure_functions[2])

    bs_nose = bs_nose_div.real * 120
    mp_nose = mp_nose_div.real * 120

    # create arrays for the range of x values for the bow shock and magnetopause
    x_bs_range = np.arange(xmin, int(bs_nose), dx)
    x_mp_range = np.arange(xmin, int(mp_nose), dx)

    # create interpolation functions for the bow shock
    interpolation_funct = interpolate.InterpolatedUnivariateSpline(x_bs, y_bs_dawn)
    new_y_bs_dawn = interpolation_funct(x_bs_range)
    interpolation_funct = interpolate.InterpolatedUnivariateSpline(x_bs, y_bs_dusk)
    new_y_bs_dusk = interpolation_funct(x_bs_range)

    # create interpolation functions for the magnetopause
    interpolation_funct = interpolate.InterpolatedUnivariateSpline(x_mp, y_mp_dawn)
    new_y_mp_dawn = interpolation_funct(x_mp_range)
    interpolation_funct = interpolate.InterpolatedUnivariateSpline(x_mp, y_mp_dusk)
    new_y_mp_dusk = interpolation_funct(x_mp_range)

    # create arrays for the whole bow shock and magnetopause
    x_whole_bs = np.concatenate((x_bs_range, [int(bs_nose)], np.flip(x_bs_range)))
    x_whole_mp = np.concatenate((x_mp_range, [int(mp_nose)], np.flip(x_mp_range)))

    y_whole_bs = np.concatenate((new_y_bs_dusk, [0], np.flip(new_y_bs_dawn)))
    y_whole_mp = np.concatenate((new_y_mp_dusk, [0], np.flip(new_y_mp_dawn)))
    
    return x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk
    


def proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk):
    # set protpn densities of the different regions in m^-3
    msheath_dens_val = 0.98 * (10**(6))
    msphere_dens_val = 0.2 * (10**(6))

    # make array to store proton density values
    proton_density = xbox*ybox*0 + 0.5

    # assign the different points in the density meshgrid the correct density value
    for i in range(0, np.shape(x_bs_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_bs_dawn[i] and ybox[j,i] < new_y_bs_dusk[i]):
                r_comp = cmath.sqrt((xbox[0][i] ** 2 ) + (ybox[j][0] ** 2))
                r = r_comp.real
                proton_density[j,i] = 5*r + msheath_dens_val
                #proton_density[j,i] = msheath_dens_val #no gradient
                
    for i in range(0, np.shape(x_mp_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_mp_dawn[i] and ybox[j,i] < new_y_mp_dusk[i]):
                proton_density[j,i] = msphere_dens_val
                
    return proton_density

    
            
def neutral_dens(xbox, ybox):
    # make array to store neutral density values
    neutrals = xbox*ybox*0 + 0.5
    neutral_plot = xbox*ybox*0 + 0.5 #stores plotted neutral density
    # assign each point in the neutral density meshgrid the correct value
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            r_comp = cmath.sqrt((xbox[0][j] ** 2 ) + (ybox[i][0] ** 2))
            r = r_comp.real
            if r>0:
                neutral_dens = 102.1 * r**(-2.4)
            
            """
            # determines which equation to use to calculate the density depending on radial distance
            if r > 14:
                neutral_dens = (-0.00284) * r + 0.5
            elif 10 < r <= 14:
                neutral_dens = (1)/((r - 9)**2) + 0.1
            elif 6 < r <= 8:
                neutral_dens = (2)/((r-10)**2) + 0.1
            elif 8 < r <= 9:
                neutral_dens = 1.04*r - 7.76
            elif 9 < r <= 10:
                neutral_dens = 6.91 - 0.59*r
            elif 3 < r <= 6:
                neutral_dens = ((math.sin((0.55 * r)*1.2))/2) -0.15
            elif 0 < r <= 3:
                neutral_dens = 0.0735 * r - 0.1455"""
            neutrals[i, j] = neutral_dens * (10**(6)) # stores the neutral density in m^-3
            
            if r>50:
                neutral_plot[i,j] = neutral_dens * (10**(6))
        
                 
    # remove negative neutral density values
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            if neutrals[i,j] < 0:
                neutrals[i,j] = 0
            if neutral_plot[i,j] < 0:
                neutral_plot[i,j] = 0
    return neutrals, neutral_plot



def temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk):
    # make array to store the ion temperature
    temperature = xbox*ybox*0 + 0.5

    # set temperature values for the different regions in eV
    msheath_temp = 197
    msphere_temp = 100 # this is a dummy value

    # assign each point in the temperature meshgrid the correct temperature
    for i in range(0, np.shape(x_bs_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_bs_dawn[i] and ybox[j,i] < new_y_bs_dusk[i]):
                r_comp = cmath.sqrt((xbox[0][i] ** 2 ) + (ybox[j][0] ** 2))
                r = r_comp.real
                temperature[j,i] = -0.07*r + msheath_temp
                #temperature[j,i] = msheath_temp #no grad
                
    for i in range(0, np.shape(x_mp_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_mp_dawn[i] and ybox[j,i] < new_y_mp_dusk[i]):
                temperature[j,i] = msphere_temp
    
    return temperature
    


def thermal_vels(xbox, ybox, temperature):
    # make array to store thermal velocity
    thermal_vel = xbox*ybox*0 + 0.5

    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            thermal_vel[i,j] = math.sqrt((3 * temperature[i,j] * 11605 * constants.k)/(constants.proton_mass)) # converts the temperature into K
            
    return thermal_vel
           
 

def bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk):
    # make array to store bulk velocity
    bulk_vel = xbox*ybox*0 + 400

    # set velocity values for the different regions in m/s
    msheath_vel = 348 * 1000
    msphere_vel = 10 * 1000 # this is a dummy value

    # assign each point in the bulk velocity meshgrid the correct value
    for i in range(0, np.shape(x_bs_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_bs_dawn[i] and ybox[j,i] < new_y_bs_dusk[i]):
                r_comp = cmath.sqrt((xbox[0][i] ** 2 ) + (ybox[j][0] ** 2))
                r = r_comp.real
                bulk_vel[j,i] = -250 * r + msheath_vel
                #bulk_vel[j,i] = msheath_vel #no grad
                
    for i in range(0, np.shape(x_mp_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_mp_dawn[i] and ybox[j,i] < new_y_mp_dusk[i]):
                bulk_vel[j,i] = msphere_vel
                
    return bulk_vel
               
 

def relative_vel(xbox, ybox, thermal_vel, bulk_vel):
    
    # create an array to store the relative velocity in m/s
    v_rel = xbox*ybox*0 + 0.5

    # assign each point in the relative velocity meshgrid the correct value
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            rel_velocity = cmath.sqrt((thermal_vel[i,j])**2 + (bulk_vel[i,j])**2)
            v_rel[i,j] = rel_velocity.real
            
    return v_rel



def O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk):
    # create array to store O8+ density
    o8_dens = xbox*ybox*0

    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            dens = 7.01145882e-5 * proton_density[i,j]
            o8_dens[i,j] = dens
    
    #print(max(o8_dens[0]))
    #print(max(o8_dens[1]))
    # cross section of the O8+ z transition in 10^-16 cm^2
    o8z_cross_sect = xbox*ybox*0 + 10
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            if v_rel[i,j] <= 300000:
                o8z_cross_sect[i,j] = 37
            elif v_rel[i,j] <= 500000:
                o8z_cross_sect[i,j] = 34
            elif v_rel[i,j] <= 700000:
                o8z_cross_sect[i,j] = 33
            elif v_rel[i,j] > 900000:
                o8z_cross_sect[i,j] = 32       
    # create array to store integral product
    zproduct = xbox*ybox*0 + 0.1
    # calculate the emission rate of the z transition
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            mult = neutrals[i,j]*o8_dens[i,j]*v_rel[i,j]*o8z_cross_sect[i,j]*(10**(-20)) # converts the cross section into m^2
            zproduct[i,j] = mult
    # crop product array to just msheath
    O8z_emission_rate = xbox*ybox*0
    for i in range(0, np.shape(x_bs_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_bs_dawn[i] and ybox[j,i] < new_y_bs_dusk[i]):
                O8z_emission_rate[j,i] = zproduct[j,i]
    for i in range(0, np.shape(x_mp_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_mp_dawn[i] and ybox[j,i] < new_y_mp_dusk[i]):
                O8z_emission_rate[j,i] = 0
    
    # cross section of the O8+ x,y transitions in 10^-16 cm^2
    o8xy_cross_sect = xbox*ybox*0 + 10
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            if v_rel[i,j] <= 700000:
                o8xy_cross_sect[i,j] = 10
            elif v_rel[i,j] <= 900000:
                o8xy_cross_sect[i,j] = 9.9
            elif v_rel[i,j] > 900000:
                o8xy_cross_sect[i,j] = 9.07
    # create array to store integral product
    xyproduct = xbox*ybox*0 + 0.1
    # calculate the emission rate of the z transition
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            mult = neutrals[i,j]*o8_dens[i,j]*v_rel[i,j]*o8xy_cross_sect[i,j]*(10**(-20)) # converts the cross section into m^2
            xyproduct[i,j] = mult
    # crop product array to just msheath
    O8xy_emission_rate = xbox*ybox*0
    for i in range(0, np.shape(x_bs_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_bs_dawn[i] and ybox[j,i] < new_y_bs_dusk[i]):
                O8xy_emission_rate[j,i] = xyproduct[j,i]
    for i in range(0, np.shape(x_mp_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_mp_dawn[i] and ybox[j,i] < new_y_mp_dusk[i]):
                O8xy_emission_rate[j,i] = 0
    
    # cross section of the O8+ w transitions in 10^-16 cm^2
    o8w_cross_sect = xbox*ybox*0 + 10
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            if v_rel[i,j] <= 300000:
                o8w_cross_sect[i,j] = 9.9
            elif v_rel[i,j] > 900000:
                o8w_cross_sect[i,j] = 11        
    # create array to store integral product
    wproduct = xbox*ybox*0 + 0.1
    # calculate the emission rate of the z transition
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            mult = neutrals[i,j]*o8_dens[i,j]*v_rel[i,j]*o8w_cross_sect[i,j]*(10**(-20)) # converts the cross section into m^2
            wproduct[i,j] = mult
    # crop product array to just msheath
    O8w_emission_rate = xbox*ybox*0
    for i in range(0, np.shape(x_bs_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_bs_dawn[i] and ybox[j,i] < new_y_bs_dusk[i]):
                O8w_emission_rate[j,i] = xyproduct[j,i]
    for i in range(0, np.shape(x_mp_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_mp_dawn[i] and ybox[j,i] < new_y_mp_dusk[i]):
                O8w_emission_rate[j,i] = 0
    
    # sum all the O8+ transition emission rates
    O8_emission_rate = O8z_emission_rate + O8xy_emission_rate + O8w_emission_rate
    
    return o8_dens, O8_emission_rate


def C6(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk):
    # create array to store C6+ density
    c6_dens = xbox*ybox*0

    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            dens = 1.440679973e-4 * proton_density[i,j]
            c6_dens[i,j] = dens
    
    # cross section of the C6+ z transition in 10^-16 cm^2
    c6z_cross_sect = xbox*ybox*0 + 10
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            if v_rel[i,j] <= 300000:
                c6z_cross_sect[i,j] = 8.7
            elif v_rel[i,j] <= 500000:
                c6z_cross_sect[i,j] = 12
            elif v_rel[i,j] <= 700000:
                c6z_cross_sect[i,j] = 16
            elif v_rel[i,j] <= 900000:
                c6z_cross_sect[i,j] = 18 
            elif v_rel[i,j] > 900000:
                c6z_cross_sect[i,j] = 20
    # create array to store integral product
    zproduct = xbox*ybox*0 + 0.1
    # calculate the emission rate of the z transition
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            mult = neutrals[i,j]*c6_dens[i,j]*v_rel[i,j]*c6z_cross_sect[i,j]*(10**(-20)) # converts the cross section into m^2
            zproduct[i,j] = mult
    # crop product array to just msheath
    C6z_emission_rate = xbox*ybox*0
    for i in range(0, np.shape(x_bs_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_bs_dawn[i] and ybox[j,i] < new_y_bs_dusk[i]):
                C6z_emission_rate[j,i] = zproduct[j,i]
    for i in range(0, np.shape(x_mp_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_mp_dawn[i] and ybox[j,i] < new_y_mp_dusk[i]):
                C6z_emission_rate[j,i] = 0
    
    # cross section of the C6+ x,y transitions in 10^-16 cm^2
    c6xy_cross_sect = xbox*ybox*0 + 10
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            if v_rel[i,j] <= 300000:
                c6z_cross_sect[i,j] = 0.65
            elif v_rel[i,j] <= 500000:
                c6z_cross_sect[i,j] = 1
            elif v_rel[i,j] <= 700000:
                c6z_cross_sect[i,j] = 1.5
            elif v_rel[i,j] <= 900000:
                c6z_cross_sect[i,j] = 1.7
            elif v_rel[i,j] > 900000:
                c6z_cross_sect[i,j] = 1.8
    # create array to store integral product
    xyproduct = xbox*ybox*0 + 0.1
    # calculate the emission rate of the z transition
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            mult = neutrals[i,j]*c6_dens[i,j]*v_rel[i,j]*c6xy_cross_sect[i,j]*(10**(-20)) # converts the cross section into m^2
            xyproduct[i,j] = mult
    # crop product array to just msheath
    C6xy_emission_rate = xbox*ybox*0
    for i in range(0, np.shape(x_bs_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_bs_dawn[i] and ybox[j,i] < new_y_bs_dusk[i]):
                C6xy_emission_rate[j,i] = xyproduct[j,i]
    for i in range(0, np.shape(x_mp_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_mp_dawn[i] and ybox[j,i] < new_y_mp_dusk[i]):
                C6xy_emission_rate[j,i] = 0
    
    # cross section of the C6+ w transitions in 10^-16 cm^2
    c6w_cross_sect = xbox*ybox*0 + 10
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            if v_rel[i,j] <= 300000:
                c6z_cross_sect[i,j] = 1.8
            elif v_rel[i,j] <= 500000:
                c6z_cross_sect[i,j] = 3
            elif v_rel[i,j] <= 700000:
                c6z_cross_sect[i,j] = 4.1
            elif v_rel[i,j] <= 900000:
                c6z_cross_sect[i,j] = 4.8
            elif v_rel[i,j] > 900000:
                c6z_cross_sect[i,j] = 5.2      
    # create array to store integral product
    wproduct = xbox*ybox*0 + 0.1
    # calculate the emission rate of the z transition
    for i in range(0, np.shape(ybox)[0]):
        for j in range(0, np.shape(xbox)[1]):
            mult = neutrals[i,j]*c6_dens[i,j]*v_rel[i,j]*c6w_cross_sect[i,j]*(10**(-20)) # converts the cross section into m^2
            wproduct[i,j] = mult
    # crop product array to just msheath
    C6w_emission_rate = xbox*ybox*0
    for i in range(0, np.shape(x_bs_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_bs_dawn[i] and ybox[j,i] < new_y_bs_dusk[i]):
                C6w_emission_rate[j,i] = xyproduct[j,i]
    for i in range(0, np.shape(x_mp_range)[0]):
        for j in range(0, np.shape(xbox)[0]):
            if (ybox[j,i] > new_y_mp_dawn[i] and ybox[j,i] < new_y_mp_dusk[i]):
                C6w_emission_rate[j,i] = 0
    
    # sum all the C6+ transition emission rates
    C6_emission_rate = C6z_emission_rate + C6xy_emission_rate + C6w_emission_rate
    
    return c6_dens, C6_emission_rate


def plot_boundaries(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "black")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Magnetosphere Boundaries", fontsize = title)
    plt.show()


def plot_neutrals(xbox, ybox, neutral_plot, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, neutral_plot, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Neutral Density (m^-3)", fontsize = label)
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Neutral Hydrogen Density - Compressed", fontsize = title)
    plt.show()
    


def plot_protons(xbox, ybox, proton_density, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, proton_density, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Proton Density (m^-3)", fontsize = label)
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Proton Density - Expanded", fontsize = title)
    plt.show()



def plot_o8_dens(xbox, ybox, o8_dens, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, o8_dens, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("O7+ density (m^-3)", fontsize = label)
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Density of O7+ ions - Compressed", fontsize = title)
    plt.show()


def plot_c6_dens(xbox, ybox, c6_dens, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, c6_dens, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label("C6+ density (m^-3)")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 2, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 2, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    plt.xlabel('x (R_J)') # the axes are labelled
    plt.ylabel('y (R_J)')
    plt.legend(fontsize = legend)
    plt.title("Density of C6+ ions")
    plt.show()


def plot_temp(xbox, ybox, temperature, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, temperature, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Ion Temperature (eV)", fontsize = label)
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Ion Temperature - Expanded", fontsize = title)
    plt.show()



def plot_therm_vel(xbox, ybox, thermal_vel, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, thermal_vel, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Ion Thermal Velocity (m/s)", fontsize = label)
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Ion Thermal Velocity - Compressed", fontsize = title)
    plt.show()
    
    
    
def plot_bulk_vel(xbox, ybox, bulk_vel, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, bulk_vel, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Bulk Velocity (m/s)", fontsize = label)
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Bulk Velocity - Compressed", fontsize = title)
    plt.show()
    
    
    
def plot_relative_vel(xbox, ybox, v_rel, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, v_rel, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Relative Velocity (m/s)", fontsize = label)
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Relative Velocity of Ions and Neutrals - Compressed", fontsize = title)
    plt.show()
    
    
def plot_O8_emission_rate(xbox, ybox, O8_emission_rate, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, O8_emission_rate, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Emission Rate (photons per second per cubic metre)", fontsize = label)
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("X-ray Emission Rate", fontsize = title)
    plt.show()

"""    
# THIS FUNCTION MAKES WEIRD GRAPHS
def plot_O8_emission_rate_log(xbox, ybox, O8_emission_rate, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    O8_emission_rate_log = np.log(O8_emission_rate)
    im = plt.contourf(xbox, ybox, O8_emission_rate_log, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label("Log of Emission Rate (photons per second per cubic metre)")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 2, color = "black")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 2, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    plt.xlabel('x (R_J)') # the axes are labelled
    plt.ylabel('y (R_J)')
    plt.legend()
    plt.title("Logarithmic Magnetosheath Emission Rate due to O8+ charge exchange")
    plt.show()"""


def plot_C6_emission_rate(xbox, ybox, C6_emission_rate, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, C6_emission_rate, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label("Emission Rate (photons per second per cubic metre)")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 2, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 2, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    plt.xlabel('x (R_J)') # the axes are labelled
    plt.ylabel('y (R_J)')
    plt.legend(fontsize = legend)
    plt.title("Magnetosheath Emission Rate due to C6+ charge exchange")
    plt.show()


def plot_line_of_sight_1(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    m = 0
    c = 0
    x_coord = []
    y_coord = []
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x_coord.append(x)
        y_coord.append(y)

    fig, ax = plt.subplots()
    ax.plot(x_coord, y_coord, label = "Line of Sight 1", linewidth = 3, color = "black", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "black")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Magnetosphere Boundaries with Line of Sight 1", fontsize = title)
    plt.show()
    
    
def contour_los_1(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    m = 0
    c = 0
    
    x_coord = []
    y_coord = []
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x_coord.append(x)
        y_coord.append(y)
    
    
    l_o_s = xbox * ybox * 0
    
    for i in range(0, np.shape(xbox)[1]):
        l_o_s[600,i] = 100

    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, l_o_s, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Line of Sight", fontsize = 16)
    ax.plot(x_coord, y_coord, label = "Line of Sight 1", linewidth = 3, color = "white", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Line of Sight 1", fontsize = title)
    plt.show()


def los_1_emission_rate(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp, O8_emission_rate):
    x_coord = []
    y_coord = []
    m = 0
    c = 0
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x_coord.append(x)
        y_coord.append(y)
    
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, O8_emission_rate, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("O7+ Emission Rate (photons per second per cubic metre)", fontsize = label)
    ax.plot(x_coord, y_coord, label = "Line of Sight 1", linewidth = 3, color = "white", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Line of Sight 1 with O8+ Emission Rate", fontsize = title)
    plt.show()


def los_1_intensity(xbox, ybox, O8_emission_rate):
    tot_intensity = 0
    
    for i in range(0, np.shape(xbox)[1]):
        intensity = O8_emission_rate[600,i]
        tot_intensity += intensity
        
    tot_intensity = tot_intensity * ((7.149*10**(7))) # convert the distances to metres from Jovian radii
    
    print("Intensity along Line of Sight 1 is: ", tot_intensity, " photons per second per square metre.")


def plot_line_of_sight_2(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    m = 1
    c = 0
    x_coord = []
    y_coord = []
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x_coord.append(x)
        y_coord.append(y)
    
    fig, ax = plt.subplots()
    ax.plot(x_coord, y_coord, label = "Line of Sight 2", linewidth = 3, color = "black", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "black")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Magnetosphere Boundaries with Line of Sight 2", fontsize = title)
    plt.show()
    
    
def contour_los_2(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    m = 1
    c = 0
    
    x_coord = []
    y_coord = []
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x_coord.append(x)
        y_coord.append(y)
    
    
    l_o_s = xbox * ybox * 0
    
    for i in range(0, np.shape(xbox)[1]):
        l_o_s[i,i] = 100


    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, l_o_s, 100, cmap = "hot")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Line of Sight", fontsize = label)
    ax.plot(x_coord, y_coord, label = "Line of Sight", linewidth = 3, color = "white", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Line of Sight", fontsize = title)
    plt.show()


def los_2_emission_rate(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp, O8_emission_rate):
    x_coord = []
    y_coord = []
    m = 1
    c = 0
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x_coord.append(x)
        y_coord.append(y)
    
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, O8_emission_rate, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("O8+ Emission Rate (photons per second per cubic metre)", fontsize = label)
    ax.plot(x_coord, y_coord, label = "Line of Sight 2", linewidth = 2, color = "white", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 2, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 2, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Line of Sight 2 with O8+ Emission Rate", fontsize = title)
    plt.show()


def los_2_intensity(xbox, ybox, O8_emission_rate):
    tot_intensity = 0
    
    for i in range(0, np.shape(xbox)[1]):
        intensity = O8_emission_rate[i,i]
        tot_intensity += intensity
        
    tot_intensity = tot_intensity * (7.149*10**(7)) # convert the distances to metres from Jovian radii
    
    print("Intensity along Line of Sight 2 is: ", tot_intensity, " photons per second per square metre.")


def plot_line_of_sight_3(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    m = -1
    c = 0
    x_coord = []
    y_coord = []
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x_coord.append(x)
        y_coord.append(y)

    fig, ax = plt.subplots()
    ax.plot(x_coord, y_coord, label = "Line of Sight 3", linewidth = 3, color = "black", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "black")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "black", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Magnetosphere Boundaries with Line of Sight 3", fontsize = title)
    plt.show()


def contour_los_3(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp):
    m = -1
    c = 0
    
    x_coord = []
    y_coord = []
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x_coord.append(x)
        y_coord.append(y)
    
    l_o_s = xbox * ybox * 0
    
    print(np.shape(l_o_s))
    
    j_ind = []
    for a in range(0, 1200):
        j_ind.append(a)
    rev_j = j_ind[::-1]
    
    print(rev_j[501])
    
    for i in range(0, np.shape(xbox)[1] - 1):
        x = i
        
        l_o_s[rev_j[i], i+1] = 100

    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, l_o_s, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Line of Sight", fontsize = label)
    ax.plot(x_coord, y_coord, label = "Line of Sight", linewidth = 3, color = "white", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Line of Sight", fontsize = title)
    plt.show()


def los_3_emission_rate(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp, O8_emission_rate):
    x_coord = []
    y_coord = []
    m = -1
    c = 0
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x_coord.append(x)
        y_coord.append(y)
    
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, O8_emission_rate, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("O8+ Emission Rate (photons per second per cubic metre)", fontsize = label)
    ax.plot(x_coord, y_coord, label = "Line of Sight 3", linewidth = 3, color = "white", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Line of Sight 3 with O8+ Emission Rate", fontsize = title)
    plt.show()


def los_3_intensity(xbox, ybox, O8_emission_rate):
    tot_intensity = 0
    
    j_ind = []
    for a in range(0, 1200):
        j_ind.append(a)
    rev_j = j_ind[::-1]
    
    for i in range(0, np.shape(xbox)[1] - 1):
        intensity = O8_emission_rate[rev_j[i], i+1]
        tot_intensity += intensity
    
    tot_intensity = tot_intensity * (7.149*10**(7)) # convert the distances to metres from Jovian radii
    
    print("Intensity along Line of Sight 3 is: ", tot_intensity, " photons per second per square metre.")


def all_los_emission_rate(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp, O8_emission_rate):
    x1_coord = []
    y1_coord = []
    m = 0
    c = 0
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x1_coord.append(x)
        y1_coord.append(y)
    
    m = 1
    c = 0
    x2_coord = []
    y2_coord = []
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x2_coord.append(x)
        y2_coord.append(y)
    
    x3_coord = []
    y3_coord = []
    m = -1
    c = 0
    for i in range(0, np.shape(xbox)[1]):
        x = xbox[0,i]
        y = m*x + c
        x3_coord.append(x)
        y3_coord.append(y)
    
    fig, ax = plt.subplots()
    im = plt.contourf(xbox, ybox, O8_emission_rate, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("O8+ Emission Rate (photons per second per cubic metre)", fontsize = label)
    ax.plot(x1_coord, y1_coord, label = "Line of Sight 1", linewidth = 5, color = "orangered", linestyle = "dotted")
    ax.plot(x2_coord, y2_coord, label = "Line of Sight 2", linewidth = 5, color = "deepskyblue", linestyle = "dotted")
    ax.plot(x3_coord, y3_coord, label = "Line of Sight 3", linewidth = 5, color = "lime", linestyle = "dotted")
    ax.plot(x_whole_bs, y_whole_bs, label = "Bow Shock", linewidth = 3, color = "white")
    ax.plot(x_whole_mp, y_whole_mp, label = "Magnetopause", linewidth = 3, color = "white", linestyle = "dashed")
    ax.set_xlim([np.min(xbox[0,:]),np.max(xbox[0,:])])
    ax.set_ylim([np.min(ybox[:,0]),np.max(ybox[:,0])])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('x (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('y (R_J)', fontsize = label)
    plt.legend(fontsize = legend)
    plt.title("Three Lines of Sight with Emission Rate - Compressed", fontsize = title)
    plt.show()
    
def intensity_bar(xbox, ybox, O8_emission_rate):
    intensities = []
    intensity = 0
    for i in range(0, np.shape(ybox)[0]):
        intensity = 0
        for j in range(0, np.shape(xbox)[1]):
            point = O8_emission_rate[i,j]
            intensity += point
        intensities.append(intensity)
    intensities = [i * (7.149*10**(7)) for i in intensities]
    return intensities

def plot_intensity_bar(intensities):
    # set minimum and maximum x and y values
    ymin = -1
    ymax = 0.5
    xmin = -300
    xmax = 300

    # set the size of the step for the numpy arrays
    dx = 0.5
    dy = 0.5

    # make the x and y numpy arrays
    x_init = np.arange(xmin, xmax + 1, dx)
    y_init = np.arange(ymin, ymax + 1, dy)

    # make the meshgrid
    xbox_bar, ybox_bar = np.meshgrid(x_init, y_init)
    
    int_bar = xbox_bar * ybox_bar * 0
    for i in range(0, np.shape(xbox_bar)[1]):
        for j in range(0, np.shape(ybox_bar)[0]):
            int_bar[j,i] = intensities[i]
            
         #THIS PART PLOTS THE BOWSHOCK BOUNDARY  
    # create empty arrays to store calculated z-values
    z_values_bs = []
    z_values_mp = []
    
    y = np.arange(-300, 300, 0.1) # creates the array of initial y-values
    y_divided = y/120
    # create arrays for the values of the dynamic pressure
    bow_shock_pressure = [0.07, 0.315]
    mag_pause_pressure = [0.039, 0.306]

    a = 0 # variable that changes which conditions are being modelled: 0 = expanded, 1 = compressed

    # get the dynamic pressure values from the arrays:
    bs_dyn_pressure = mag_pause_pressure[a]
    mp_dyn_pressure = mag_pause_pressure[a]

    # array_format_functions = [A, B, C, D, E, F]
    bs_pressure_functions = [-1.107 + 1.591*(bs_dyn_pressure)**(-0.25), -0.566 - 0.812*(bs_dyn_pressure)**(-0.25), 0.048 - 0.059*(bs_dyn_pressure)**(-0.25), 0.077 - 0.038*(bs_dyn_pressure), -0.874 - 0.299*(bs_dyn_pressure), -0.055 + 0.124*(bs_dyn_pressure)]
    mp_pressure_functions = [-0.134 + 0.488*(mp_dyn_pressure)**(-0.25), -0.581 - 0.225*(mp_dyn_pressure)**(-0.25), -0.186 - 0.016*(mp_dyn_pressure)**(-0.25), -0.014 + 0.096*(mp_dyn_pressure), -0.814 - 0.811*(mp_dyn_pressure), -0.050 + 0.168*(mp_dyn_pressure)]
    
    
    for i in y_divided:
        
        z_bs = cmath.sqrt(bs_pressure_functions[0] + bs_pressure_functions[3]*i + bs_pressure_functions[4]*i**2)
        z_mp = cmath.sqrt(mp_pressure_functions[0] + mp_pressure_functions[3]*i + mp_pressure_functions[4]*i**2)
        
        z_values_bs.append(z_bs)
        z_values_mp.append(z_mp)
        
    z_mp_divided = np.array(z_values_mp)
    z_bs_divided = np.array(z_values_bs)
    
    z_mp = z_mp_divided*120
    z_bs = z_bs_divided*120
    
    z_bs_n = z_bs * -1        
    
    a = 1040
    b = 5080
    
    bs_plot = z_bs[a:b]
    y_bs_plot = y[a:b]
    bs_plot_n = z_bs_n[a:b]
    
    
    fig, ax = plt.subplots()
    im = plt.contourf(xbox_bar, ybox_bar, int_bar, 100, cmap = "viridis")
    cbar = fig.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Intensity (photons per second per square metre)", fontsize = label)
    #plt.plot(y_bs_plot, bs_plot, "black",  label='bowshock') # the bowshock coordinates are plotted
    #plt.plot(y_bs_plot, bs_plot_n, "black") # the bowshock coordinates are plotted
    ax.set_xlim([-300,300])
    ax.set_ylim([-2,2])
    ax.tick_params(axis='both', which='major', labelsize=maj_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=min_ticks)
    plt.xlabel('y (R_J)', fontsize = label) # the axes are labelled
    plt.ylabel('z (R_J)', fontsize = label)
    plt.title("Intensity of Equatorial Plane in the y-z plane", fontsize = title)
    plt.show()
    
