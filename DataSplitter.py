import pandas as pd

# Load the dataset
df = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/test/AD.csv')

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Now, define your column groups without adding spaces around the names
common_columns = ['Identifier', 'Experiment', 'Well_ID', 'Row', 'Column', 'Field', 'Object Number']
morphological_columns = common_columns + ['CellArea', 'cellbody_area', 'Cell_Elongation', 'cell_full_length', 'cell_half_width', 'Cell_length_by_area', 'Cell_width_by_area', 'NucleusArea', 'Nuc_Elongation', 'Nuc_full_length', 'Nuc_half_width', 'Nuc_Roundness', 'roundness', 'body_roundness', 'percentProtrusion', 'protrusion_extent', 'total_protrusionarea', 'mean_protrusionarea', 'number_protrusions', 'skeleton_area', 'skeleton_node_count', 'skeletonareapercent']
intensity_measurements_columns = common_columns + ['cytointensityActin', 'cytointensityTubulin', 'CytoIntensityH', 'CytoNonMembraneIntensityActin', 'CytoNonMembraneIntensityTubulin', 'MembraneIntensityActin', 'MembraneIntensityTubulin', 'NucIntensityActin', 'NucIntensityTubulin', 'NucIntensityH', 'ProtrusionIntensityActin', 'ProtrusionIntensityTubulin', 'ringIntensityActin', 'ringIntensityTubulin', 'RingIntensityH', 'WholeCellIntensityActin', 'WholeCellIntensityTubulin', 'WholeCellIntensityH']
spatial_texture_columns = common_columns + ['centers_area', 'centers_distance', 'cytoplasm_area', 'GaborMax1_Actin', 'GaborMin1_Actin', 'HarConCellAct', 'HarConCytoTub', 'HarConMembAct', 'HarCorrCellAct', 'HarCorrCytoTub', 'HarCorrMembAct', 'HarHomCellAct', 'HarHomCytoTub', 'HarHomMembAct', 'HarSVCellAct', 'HarSVCytoTub', 'HarSVMembAct', 'logNucbyRingActin', 'logNucbyRingTubulin', 'mean_prlength', 'MembranebyCytoOnlyActin', 'MembranebyCytoOnlyTubulin', 'NucbyCytoArea', 'NucbyRingActin', 'NucbyRingTubulin', 'NucPlusRingActin', 'NucPlusRingTubulin', 'RingbyCytoActin', 'RingbyCytoTubulin', 'ringregion_area']
ser_analysis_columns = common_columns + ['SERBrightCellAct', 'SERBrightCytoTub', 'SERBrightMembAct', 'SERBrightNuc', 'SERDarkCellAct', 'SERDarkCytoTub', 'SERDarkMembAct', 'SERDarkNuc', 'SEREdgeCellAct', 'SEREdgeCytoTub', 'SEREdgeMembAct', 'SEREdgeNuc', 'SERHoleCellAct', 'SERHoleCytoTub', 'SERHoleMembAct', 'SERHoleNuc', 'SERRidgeCellAct', 'SERRidgeCytoTub', 'SERRidgeMembAct', 'SERRidgeNuc', 'SERSaddleCellAct', 'SERSaddleCytoTub', 'SERSaddleMembAct', 'SERSaddleNuc', 'SERSpotCellAct', 'SERSpotCytoTub', 'SERSpotMembAct', 'SERSpotNuc', 'SERValleyCellAct', 'SERValleyCytoTub', 'SERValleyMembAct', 'SERValleyNuc']

# Create dataframes for each group
morphological_df = df[morphological_columns]
intensity_measurements_df = df[intensity_measurements_columns]
spatial_texture_df = df[spatial_texture_columns]
ser_analysis_df = df[ser_analysis_columns]

# Define a function to save dataframes to CSV
def save_group_to_csv(dataframe, filename):
    path = f'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/{filename}.csv'
    dataframe.to_csv(path, index=False)
    print(f'File saved: {path}')

# Save each group to a separate CSV file
save_group_to_csv(morphological_df, 'morphological_data')
save_group_to_csv(intensity_measurements_df, 'intensity_measurements_data')
save_group_to_csv(spatial_texture_df, 'spatial_texture_data')
save_group_to_csv(ser_analysis_df, 'ser_analysis_data')
