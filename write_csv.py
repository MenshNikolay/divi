import csv




def save_to_csv(results, output_file):
    if not results:
        print("Данные для записи не были получены.")
        return
    
    fieldnames = list(results[0].keys())
    
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames,delimiter=',')
        if file.tell() == 0:
            writer.writeheader()
        for result in results:
            writer.writerow(result)