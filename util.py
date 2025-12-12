throb_count = 0
def throb():
    global throb_count
    if throb_count % 2 == 0:
        print(".", end="", flush=True)
    elif throb_count % 2 == 1:
        print("\r \r", end="", flush=True)
    throb_count+=1
