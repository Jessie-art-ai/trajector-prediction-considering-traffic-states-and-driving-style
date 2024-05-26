import os


def merge_files(input_directory, output_directory):
    files = os.listdir(input_directory)
    grouped_files = {}

    # Group files by the pattern rounD_<any>_<clump/unclump/neutral>_<number>
    for file in files:
        if file.endswith(".txt"):
            parts = file.split('_')
            if len(parts) == 4:
                clump_status = parts[2]
                number = parts[3].split('.')[0]
                key = f"{clump_status}_{number}"
                if key not in grouped_files:
                    grouped_files[key] = []
                grouped_files[key].append(file)

    # Merge files in each group and save to the output directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for key, file_list in grouped_files.items():
        merged_file_name = f"rounD_{key}.txt"
        with open(os.path.join(output_directory, merged_file_name), 'w') as outfile:
            for fname in file_list:
                with open(os.path.join(input_directory, fname)) as infile:
                    outfile.write(infile.read())
                outfile.write("\n")  # Add newline between files


def main():
    base_dir = r'E:\yjy\code\Improving-Multi-agent-Trajectory-Prediction-using-Traffic-States-on-Interactive-Driving' \
               r'-Scenarios-master\TS-TRANSFORMER\rounD\process_5fps_2'
    output_base_dir = r'E:\yjy\code\Improving-Multi-agent-Trajectory-Prediction-using-Traffic-States-on-Interactive' \
                      r'-Driving-Scenarios-master\TS-TRANSFORMER\rounD\merge_rounD'
    sub_dirs = ['train', 'test', 'val']

    for sub_dir in sub_dirs:
        input_directory = os.path.join(base_dir, sub_dir)
        output_directory = os.path.join(output_base_dir, sub_dir)
        if os.path.exists(input_directory):
            merge_files(input_directory, output_directory)
        else:
            print(f"Directory {input_directory} does not exist")


if __name__ == "__main__":
    main()
