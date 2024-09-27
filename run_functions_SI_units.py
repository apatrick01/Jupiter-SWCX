# -*- coding: utf-8 -*-
"""
Calls functions from modular_code.py
"""
from modular_code_SI_units import *



def msphere_bounds():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    plot_boundaries(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)

def neut_dens():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    plot_neutrals(xbox, ybox, neutral_plot, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)
    
def pro_dens():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    plot_protons(xbox, ybox, proton_density, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)

def O8_ion_dens():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    plot_o8_dens(xbox, ybox, o8_dens, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)

def C6_ion_dens():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    c6_dens, C6_emission_rate = C6(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    plot_c6_dens(xbox, ybox, c6_dens, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)
    
def temp():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    plot_temp(xbox, ybox, temperature, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)

def thermal_velocity():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    plot_therm_vel(xbox, ybox, thermal_vel, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)

def bulk_velocity():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    plot_bulk_vel(xbox, ybox, bulk_vel, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)

def relative_velocity():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    plot_relative_vel(xbox, ybox, v_rel, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)

def O8_emission():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    f = open("emission_rate_array.txt", "w")
    for i in range(0, len(O8_emission_rate)):
        f.write("[")
        for j in range(0, len(O8_emission_rate[i])):
            f.write(str(O8_emission_rate[i][j]) + ", ")
        f.write("]")
        f.write("\n")
    f.close()
    plot_O8_emission_rate(xbox, ybox, O8_emission_rate, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)

    # THIS MAKES WEIRD GRAPHS
def O8_emission_log():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    plot_O8_emission_rate_log(xbox, ybox, O8_emission_rate, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)
    
def C6_emission():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    c6_dens, C6_emission_rate = C6(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    plot_C6_emission_rate(xbox, ybox, C6_emission_rate, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)
    
def los_1():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    plot_line_of_sight_1(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)
    
def los_2():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    plot_line_of_sight_2(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)
    
def los_3():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    plot_line_of_sight_3(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp)
    
def los_1_int():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    los_1_intensity(xbox, ybox, O8_emission_rate)
    
def los_2_int():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    los_2_intensity(xbox, ybox, O8_emission_rate)
    
def los_3_int():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    los_3_intensity(xbox, ybox, O8_emission_rate)
    
def los_1_emission():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    los_1_emission_rate(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp, O8_emission_rate)
    
def los_2_emission():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    los_2_emission_rate(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp, O8_emission_rate)
    
def los_3_emission():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    los_3_emission_rate(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp, O8_emission_rate)
    
def all_los_emission():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    all_los_emission_rate(xbox, ybox, x_whole_bs, y_whole_bs, x_whole_mp, y_whole_mp, O8_emission_rate)
    
def int_bar():
    xbox, ybox, xmin, xmax, dx, dy = create_meshgrid()
    x_bs_range, x_mp_range, x_whole_bs, x_whole_mp, y_whole_bs, y_whole_mp, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk = boundaries(xmin, dx)
    proton_density = proton_dens(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    neutrals, neutral_plot = neutral_dens(xbox, ybox)
    temperature = temps(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    thermal_vel = thermal_vels(xbox, ybox, temperature)
    bulk_vel = bulk_vels(xbox, ybox, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    v_rel = relative_vel(xbox, ybox, thermal_vel, bulk_vel)
    o8_dens, O8_emission_rate = O8(xbox, ybox, proton_density, v_rel, neutrals, x_bs_range, x_mp_range, new_y_bs_dawn, new_y_bs_dusk, new_y_mp_dawn, new_y_mp_dusk)
    intensities = intensity_bar(xbox, ybox, O8_emission_rate)
    f = open("slice_intensity_array.txt", "w")
    f.write("[")
    for i in range(0, len(intensities)):
        f.write(str(intensities[i]) + ", ")
    f.write("]")
    f.close()
    plot_intensity_bar(intensities)