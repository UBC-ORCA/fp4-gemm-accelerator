import sys

infile = ""
outfile = ""
outsize = 0

if len(sys.argv) != 4:
    print("Please use {} infile outfile outsize".format(sys.argv[0]));
    exit(-1)

infile = sys.argv[1]
outfile = sys.argv[2]
outsize = int(sys.argv[3], 16)

print(f"Params {infile} {outfile} {outsize}")

hexarr = ["00" for i in range(outsize)]
fileaddr = -1
with open(infile, 'r') as hexdump:
    for line in hexdump:
        linedata = line.strip().split(" ")
        if(linedata[0][0] == "@"):
            fileaddr = int(linedata[0][1:], 16)
            continue
        if(fileaddr == -1):
            print("Invalid format")
        for hexbyte in linedata:
            hexarr[fileaddr] = hexbyte
            fileaddr += 1

with open(outfile, 'w') as hexout:
    for hexbyte in hexarr:
        hexout.write(hexbyte + " ")
        

