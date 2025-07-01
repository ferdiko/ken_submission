import json
import csv

def main(xxx):
    txt_file = 'mt_bench_single_model_samples.txt'
    json_file = f'mt_bench_single_model_samples/certs_{xxx}_norm_prod_h1.json'

    # Initialize variables to store the lists
    xxx_list = None
    model_70b_list = None

    # Read the txt file and extract the lists
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Check for the xxx line
            if line.startswith(f'mt_bench_single_model_samples/2x1600_{xxx}.txt:'):
                # Extract the list after ': '
                parts = line.split(': ', 1)
                if len(parts) > 1:
                    xxx_list_str = parts[1]
                    xxx_list = eval(xxx_list_str.split(', ', 1)[1])  # Use eval to convert string to list
            # Check for the 70b line
            elif line.startswith('mt_bench_single_model_samples/2x1600_70b.txt:'):
                # Extract the list after ': '
                parts = line.split(': ', 1)
                if len(parts) > 1:
                    model_70b_list_str = parts[1]
                    model_70b_list = eval(model_70b_list_str.split(', ', 1)[1])  # Use eval to convert string to list

    # Ensure both lists were found
    if xxx_list is None:
        print(f"Could not find data for xxx='{xxx}' in '{txt_file}'.")
        return
    if model_70b_list is None:
        print(f"Could not find data for '70b' model in '{txt_file}'.")
        return


    csv_file = "mt_bench_single_model_samples/prefill_certs.txt"


    # # Read the CSV file
    # csv_data = {}
    # with open(csv_file, 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         if len(row) >= 2:
    #             index = int(row[0])
    #             value = float(row[1])
    #             csv_data[index] = value

    # # Map CSV keys to indices in the lists
    # mapped_data = {}
    # for index, csv_value in csv_data.items():
    #     try:
    #         # Get corresponding values from the lists
    #         xxx_value = xxx_list[index]
    #         model_70b_value = model_70b_list[index]
    #         mapped_data[index] = {
    #             'cert': csv_value,
    #             f'{xxx}_value': xxx_value,
    #             '70b_value': model_70b_value
    #         }
    #     except IndexError:
    #         print(f"Index '{index}' is out of range in the lists.")
    #         continue


    # Read the JSON file
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Map JSON keys to indices in the lists
    # Assuming the JSON keys correspond to indices
    mapped_data = {}
    for key in json_data.keys():
        try:
            index = int(key)  # Convert key to integer index
            # Get corresponding values from the lists
            xxx_value = xxx_list[index]
            model_70b_value = model_70b_list[index]
            mapped_data[key] = {
                'cert': json_data[key],
                f'{xxx}_value': xxx_value,
                '70b_value': model_70b_value
            }
        except (ValueError, IndexError):
            print(f"Key '{key}' is not a valid index or index out of range.")
            continue

    
    total_sum = 0
    total_small = 0
    total_large = 0
    thresh = 0.43
    for _, values in mapped_data.items():
        if float(values['cert']) < thresh:
            total_large += 1
            total_sum += float(values['70b_value'])
        else:
            total_small += 1
            total_sum += float(values[f'{xxx}_value'])


    print("thresh:", thresh)
    print("avg score:", total_sum/len(mapped_data))
    print(f"cost: large({total_large}) small({total_small})")

    # Output the mapped data
    # for key, values in mapped_data.items():
    #     print(f"Index {key}:")
    #     print(f"  JSON Data: {values['json_data']}")
    #     print(f"  {xxx} Value: {values[f'{xxx}_value']}")
    #     print(f"  70b Value: {values['70b_value']}\n")

if __name__ == "__main__":
    main("3b")
