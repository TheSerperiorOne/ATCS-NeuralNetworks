def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        try:
            number = float(line)
            number /= 255
            processed_lines.append(str(number) + '\n')
        except ValueError:
            processed_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(processed_lines)

if __name__ == "__main__":
    input_file = "Data/TrainingData2.txt"  # Change this to your input file name
    output_file = "TrainingData2.txt"  # Change this to your output file name
    process_file(input_file, output_file)