import numpy as np

# Given data
weight_kg = 22  # child's weight in kg
force_multiplier = 3.2  # times body weight
gravity = 9.81  # acceleration due to gravity in m/s^2
force_distribution_radius = 0.83  # percentage of force on radius
outer_radius_mm = 12  # outer radius of the bone in mm
cortical_thickness_mm = 2.4  # thickness of the cortical bone in mm
distance_to_critical_point_mm = 33  # distance from force application to critical location in mm
angle_degrees = 21  # angle of force application

# Yield stresses in MPa
sigma_yield_tension = 93
sigma_yield_compression = 148

# Calculating total force applied to the hand
F_total = weight_kg * gravity * force_multiplier

# Calculating force on the radius
F_radius = F_total * force_distribution_radius

# Convert angle to radians for calculations
angle_radians = np.radians(angle_degrees)

# Axial force component (F_axial) = F_radius * cos(angle)
# Bending force component (F_bending) = F_radius * sin(angle)
F_axial = F_radius * np.cos(angle_radians)
F_bending = F_radius * np.sin(angle_radians)

# Cross-sectional area (A) of the bone
R_mm = outer_radius_mm
r_mm = outer_radius_mm - cortical_thickness_mm
A_mm2 = np.pi * (R_mm**2 - r_mm**2)

# Second moment of area (I) for the cylindrical section of the bone
I_mm4 = (np.pi / 4) * (R_mm**4 - r_mm**4)

# Axial stress (σA) = P/A
sigma_A = F_axial / A_mm2  # Axial stress in MPa

# Bending stress (σB) = My/I
# M = F_bending * distance to critical point, y = distance from neutral axis to outer edge
M_Nmm = F_bending * distance_to_critical_point_mm  # Moment in N*mm due to bending component
y_mm = outer_radius_mm  # distance from neutral axis to outer edge for max stress
sigma_B = (M_Nmm * y_mm) / I_mm4  # Bending stress in MPa

# Principal stresses at the anterior and posterior sides of the bone
# Anterior side (tension due to bending)
sigma_principal_anterior = sigma_A + sigma_B

# Posterior side (compression due to bending)
sigma_principal_posterior = sigma_B - sigma_A if sigma_B > sigma_A else 0  # Compression can't be negative

# Factor of Safety for the anterior side (tension side)
FS_anterior = sigma_yield_tension / sigma_principal_anterior

# Factor of Safety for the posterior side (compression side)
FS_posterior = sigma_yield_compression / sigma_principal_posterior if sigma_principal_posterior > 0 else float('inf')

# Output the results
print(f"Total force applied to the hand: {F_total:.2f} N")
print(f"Force on the radius: {F_radius:.2f} N")
print(f"Axial force component: {F_axial:.2f} N")
print(f"Bending force component: {F_bending:.2f} N")
print(f"Axial stress (σA): {sigma_A:.2f} MPa")
print(f"Bending stress (σB): {sigma_B:.2f} MPa")
print(f"Principal stress on the anterior side: {sigma_principal_anterior:.2f} MPa")
print(f"Principal stress on the posterior side: {sigma_principal_posterior:.2f} MPa")
print(f"Factor of Safety on the anterior side: {FS_anterior:.2f}")
print(f"Factor of Safety on the posterior side: {FS_posterior:.2f}")
