import pickle

# ... (Previous code logic)
def process():
    data1 = []
    try:
        with open("Broad_line_AGNs.txt", "r") as f:
            for line in f:
                parts = line.split()
                if parts:
                    data1.append(parts)
    except Exception as e:
        print(f"Error reading data1: {e}")
        return

    data2 = []
    try:
        with open("AGNs.txt", "r") as f:
            for line in f:
                parts = line.split()
                if parts:
                    data2.append(parts)
    except Exception as e:
        print(f"Error reading data2: {e}")
        return

    broadLINEAGNdata = []
    if len(data1) > 9:
        try:
            broadLINEAGNdata.append([[float(data1[9][7]), "".join(list(data1[9][8])[:4])], [0.3, 0.5]])
        except Exception as e:
            print(f"Error processing row 9: {e}")

    for j in range(11, 254):
        if j-1 < len(data1):
            try:
                row = data1[j-1]
                broadLINEAGNdata.append([[float(row[7]), "".join(list(row[8])[:4])], [0.3, 0.5]])
            except Exception as e:
                print(f"Error processing row {j-1}: {e}")

    otherAGNdata = []
    for j in range(10, 37):
        if j-1 < len(data2):
            try:
                row = data2[j-1]
                val = float(row[-1])
                otherAGNdata.append([[row[-3], row[-2]], [0.3, max([val, 0.01])], 0])
            except Exception as e:
                print(f"Error processing data2 row {j-1}: {e}")

    dynamicallymeasured = []
    for j in range(37, 116):
        if j-1 < len(data2):
            try:
                row = data2[j-1]
                val = float(row[-1])
                dynamicallymeasured.append([[row[-3], row[-2]], [0.3, max([val, 0.01])], 0])
            except Exception as e:
                print(f"Error processing data2 row {j-1}: {e}")

    with open("agn_data.pkl", "wb") as f:
        pickle.dump({
            "broadLINEAGNdata": broadLINEAGNdata,
            "otherAGNdata": otherAGNdata,
            "dynamicallymeasured": dynamicallymeasured
        }, f)
    
    print(f"Processed {len(broadLINEAGNdata)} broad AGN, {len(otherAGNdata)} other AGN, {len(dynamicallymeasured)} dynamic.")

if __name__ == "__main__":
    process()
