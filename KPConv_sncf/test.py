import pathlib
from collections import Counter
from utils.ply import read_ply, write_ply

data_dir = pathlib.Path(r"C:\Users\Milo\OneDrive - Universiteit Utrecht\Scriptie\Data\KPConv\train")

for ply in data_dir.iterdir():
    if ply.is_file() and ply.suffix == ".ply":
        data = read_ply(str(ply))

        # print(type(data[0]))
        # exit()
        print(Counter([row[3] for row in data]))

        continue

        # # Filter the rows where scalar_Classification is <= 9
        mask = (data["scalar_Classification"] <= 9) & (data["scalar_Classification"] != 0)

        # Create filtered data
        filtered_data = data[mask]

        # Define the field names for the .ply file
        field_names = [
            'x', 'y', 'z', 'scalar_Return_Number', 'scalar_Number_Of_Returns',
            'scalar_Scan_Angle_Rank', 'scalar_Point_Source_ID', 'scalar_Gps_Time',
            'scalar_EdgeOfFlightLine', 'scalar_Classification', 'scalar_Intensity'
        ]

        # Write the filtered data back to the same file
        write_ply(str(ply), 
                 [
                     filtered_data['x'], filtered_data['y'], filtered_data['z'],
                     filtered_data['scalar_Return_Number'],
                     filtered_data['scalar_Number_Of_Returns'],
                     filtered_data['scalar_Scan_Angle_Rank'],
                     filtered_data['scalar_Point_Source_ID'],
                     filtered_data['scalar_Gps_Time'],
                     filtered_data['scalar_EdgeOfFlightLine'],
                     filtered_data['scalar_Classification'],
                     filtered_data['scalar_Intensity']
                 ],
                 field_names)
        
        print(f"Saved: {ply}")
