import numpy as np

weight_kg = 22  
force_multiplier = 3.2  
gravity = 9.81  
force_distribution_radius = 0.83  
outer_radius_mm = 12  
cortical_thickness_mm = 2.4  
distance_to_critical_point_mm = 33  
angle_degrees = 21  

# yield stresses in mpa
sigma_yield_tension = 93
sigma_yield_compression = 148

# calculating total force applied to the hand
F_total = weight_kg * gravity * force_multiplier

# Calculating force on the radius
F_radius = F_total * force_distribution_radius

# Convert angle to radians for calculations
angle_radians = np.radians(angle_degrees)

# Axial force component 
F_axial = F_radius * np.cos(angle_radians)
# Bending force component 
F_bending = F_radius * np.sin(angle_radians)

# Cross-sectional area (A) of the bone
R_mm = outer_radius_mm
r_mm = outer_radius_mm - cortical_thickness_mm
A_mm2 = np.pi * (R_mm**2 - r_mm**2)

#second moment of area (I) for the cylindrical section of the bone
I_mm4 = (np.pi / 4) * (R_mm**4 - r_mm**4)

# Axial stress 
sigma_A = F_axial / A_mm2  # Axial stress in mpa

# bending stress 
# M =F_bending*distance to critical point, y=distance from neutral axis to outer edge
M_Nmm = F_bending * distance_to_critical_point_mm  # moment in N*mm due to bending component
y_mm = outer_radius_mm  # distance from neutral axis to outer edge for max stress
sigma_B = (M_Nmm * y_mm) / I_mm4  # bending stress in MPa

# principal stresses at the anterior and posterior sides of the bone
# Anterior side, tension 
sigma_principal_anterior = sigma_A + sigma_B

# Posterior side, compression 
sigma_principal_posterior = sigma_B - sigma_A if sigma_B > sigma_A else 0  # Compression can't be negative

#calculating peak forces N*mm^2
peak_force_anterior = sigma_principal_anterior * A_mm2  
peak_force_posterior = sigma_principal_posterior * A_mm2 

# factor of safety for the anterior/ tension 
FS_anterior = sigma_yield_tension / sigma_principal_anterior

# factor of safety for the posterior side, compression
FS_posterior = sigma_yield_compression / sigma_principal_posterior if sigma_principal_posterior > 0 else float('inf')



print(f"Total force applied to the hand: {F_total:.2f} N")
print(f"Force on the radius: {F_radius:.2f} N")
print(f"Axial force component: {F_axial:.2f} N")
print(f"Bending force component: {F_bending:.2f} N")
print(f"Axial stress: {sigma_A:.2f} MPa")
print(f"Bending stress : {sigma_B:.2f} MPa")
print(f"Principal stress on the anterior side: {sigma_principal_anterior:.2f} MPa")
print(f"Principal stress on the posterior side: {sigma_principal_posterior:.2f} MPa")
print(f"Peak stress on the anterior side: {peak_force_anterior:.2f} N*mm^2")
print(f"Peak stress on the posterior side: {peak_force_posterior:.2f} N*mm^2")
print(f"Factor of Safety on the anterior side: {FS_anterior:.2f}")
print(f"Factor of Safety on the posterior side: {FS_posterior:.2f}")
