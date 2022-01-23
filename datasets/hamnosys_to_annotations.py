#Col 1 ID
#Col 2 dunno
#Col 3 dunno
#Col 4 dunno
#Col 5 Word
#Col 6 Hamnosys
#Col 7 Hamnosys copy
#Col 8 Symmetry operator 
#Col 9 Dom hand - Handshape - base form
#Col 10 Dom hand - Handhape - Thumb position
#Col 11 Dom hand - Handshape - Bending
#Col 12 Dom hand - Handposition - Ext finger direction
#Col 13 Dom hand - Handposition - Palm orientation
#Col 14 Dom hand - Handlocation - Frontal plane l/r
#Col 15 Dom hand - Handlocation - Frontal plane t/b
#Col 16 Dom hand - Handlocation - Distance from the body

from argparse import ArgumentParser
import sys

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename", metavar='file', 
                    type=str, help="File to read from")
parser.add_argument("-l", "--log", dest="logging", metavar='log', 
                    type=bool, help="Enable logging")
 
args = parser.parse_args()

listofentries = []

with open(args.filename) as file:
    for line in file:
        sublist = line.split()
        sublist.append(sublist[5])
        listofentries.append(sublist)

nondominanhandbaseform = []
nondominanhandthumbposition = []
nondominanhandbending = []
nondominanhandextfingdirect = []
nondominanhandpalmorient = []
nondominanhandlocationTB = []

for i in listofentries:
    nondominanhandbaseform.append("99")
    nondominanhandthumbposition.append("99")
    nondominanhandbending.append("99")
    nondominanhandextfingdirect.append("99")
    nondominanhandpalmorient.append("99")
    nondominanhandlocationTB.append("99")

SymmetryOperatorsDict = {
#   "0" : None,
    "1" : "",
    "2" : "",
    "3" : "",
    "4" : "",
    "5" : "",
    "6" : "",
    "7" : "",
    "8" : "",
    "9" : "",
    "10" : "",
    "11" : "",
    "12" : ""
}

if (args.logging):
    print("1. Symetry operators analysis:")

for entry in listofentries:
    #Search for symmetry operators that consists of 3 symbols
    char = entry[6][0:3]
    if (char == SymmetryOperatorsDict["1"]):
        entry.append("1")
        entry[6] = entry[6][3:]
        continue
    elif (char == SymmetryOperatorsDict["2"]):
        entry.append("2")
        entry[6] = entry[6][3:]
        continue
    #todo fix it
    elif (char == SymmetryOperatorsDict["10"]):
        entry.append("10")
        entry[6] = entry[6][3:]
        continue
    #todo fix it
    elif (char == SymmetryOperatorsDict["11"]):
        entry.append("11")
        entry[6] = entry[6][3:]
        continue

    #Search for symmetry operators that consists of 2 symbols
    char = entry[6][0:2]
    if (char == SymmetryOperatorsDict["3"]):
        entry.append("3")
        entry[6] = entry[6][2:]
        continue
    elif (char == SymmetryOperatorsDict["4"]):
        entry.append("4")
        entry[6] = entry[6][2:]
        continue
    elif (char == SymmetryOperatorsDict["5"]):
        entry.append("5")
        entry[6] = entry[6][2:]
        continue
    elif (char == SymmetryOperatorsDict["6"]):
        entry.append("6")
        entry[6] = entry[6][2:]
        continue

    #Search for symmetry operators that consists of 1 symbol
    char = entry[6][0:1]
    if (char == SymmetryOperatorsDict["7"]):
        entry.append("7")
        entry[6] = entry[6][1:]
        continue
    elif (char == SymmetryOperatorsDict["8"]):
        entry.append("8")
        entry[6] = entry[6][1:]
        continue
    elif (char == SymmetryOperatorsDict["9"]):
        entry.append("9")
        entry[6] = entry[6][1:]
        continue
    elif (char == SymmetryOperatorsDict["12"]):
        entry.append("12")
        entry[6] = entry[6][1:]
        continue

    #If no symmetry operators were found mark class as 0
    else:
        entry.append("0")

if (args.logging):
    for entry in listofentries:
        print(entry[0] + ": " + entry[5] + " -> " + entry[6] + "(" + entry[7] + ")")
        if (len(entry) > 8):
            print("Critical error")
            sys.exit(0)

UnknownSymbols1Dict = {
    "0" : "",
    "1" : "",
}

#Remove signs that may appear in beetwen symmetry operator and handshape - base form
for entry in listofentries:
    char = entry[6][0:1]
    for key, value in UnknownSymbols1Dict.items():
        if (char == value):
            entry[6] = entry[6][1:]

#Do it twice since sometimes there are two in a row
for entry in listofentries:
    char = entry[6][0:1]
    for key, value in UnknownSymbols1Dict.items():
        if (char == value):
            entry[6] = entry[6][1:]

HandshapeBaseformsDict = {
    "0" : "",
    "1" : "",
    "2" : "",
    "3" : "",
    "4" : "",
    "5" : "",
    "6" : "",
    "7" : "",
    "8" : "",
    "9" : "",
    "10" : "",
    "11" : "",
    "12" : ""
}

if (args.logging):
    print("2. Handshape - base form analisys:")

for entry in listofentries:
    char = entry[6][0:1]
    for key, value in HandshapeBaseformsDict.items():
        if (char == value):
            entry.append(key)
            entry[6] = entry[6][1:]
        continue
 
if (args.logging):
    for entry in listofentries:
        print(entry[0] + ": " + entry[5] + " -> " + entry[6] + "(" + entry[8] + ")")
        if (len(entry) > 9):
            print("Critical error")
            print(entry)
            sys.exit(0)

HandshapeThumbPositionDict = {
#   "0" : None
    "1" : "",
    "2" : "",
    "3" :  "",
}

if (args.logging):
    print("3. Handshape - Thumb position analisys:")

for entry in listofentries:
    char = entry[6][0:1]
    for key, value in HandshapeThumbPositionDict.items():
        if (char == value):
            entry.append(key)
            entry[6] = entry[6][1:]
    if (len(entry) == 9):
        entry.append("0")

if (args.logging):
    for entry in listofentries:
        print(entry[0] + ": " + entry[5] + " -> " + entry[6] + "(" + entry[9] + ")")
        if (len(entry) > 10):
            print("Critical error")
            sys.exit(0)

HandshapeBendingDict = {
#   "0" : None,
    "1" : "",
    "2" : "",
    "3" : "",
    "4" : "",
    "5" : "",
}

if (args.logging):
    print("4. Handshape - bending analisys:")

UnknownSymbols2Dict = {
    "0" : "",
    "1" : "",
    "2" : "",
    "4" : "",
    "5" : "",
    "6" : "",
    "7" : "",
    "8" : "",
    "9" : "",
    "10" : "",
    "11" : "",
    "12" : "",
}

#Remove signs that may appear in beetwen 
for entry in listofentries:
    for i in range(4):
        char = entry[6][0:1]
        for key, value in UnknownSymbols2Dict.items():
            if (char == value):
                entry[6] = entry[6][1:]

for entry in listofentries:
    char = entry[6][0:1]
    for key, value in HandshapeBendingDict.items():
        if (char == value):
            entry.append(key)
            entry[6] = entry[6][1:]
    if (len(entry) == 10):
        entry.append("0")

#for some entries thumb is placed after the bending sign, so let's go back to thumb
for entry in listofentries:
    char = entry[6][0:1]
    for key, value in HandshapeThumbPositionDict.items():
        if (char == value):
            entry[9] = (key)
            entry[6] = entry[6][1:]

#Entry 178 in pjm has 2 bendings in a row, remove if it is duplicated
for entry in listofentries:
    char = entry[6][0:1]
    for key, value in HandshapeBendingDict.items():
        if (char == value):
            entry[6] = entry[6][1:]

if (args.logging):
    for entry in listofentries:
        print(entry[0] + ": " + entry[5] + " -> " + entry[6] + "(" + entry[10] + ")")
        if (len(entry) > 11):
            print("Critical error")
            sys.exit(0)

for entry,i in zip(listofentries,range(len(listofentries))):
    char = entry[6][0:1]
    if (char == "" or char==""):
        entry[6] = entry[6][1:]
        char = entry[6][0:1]
        for key, value in HandshapeBaseformsDict.items():
            if (char == value):
                nondominanhandbaseform[i] = key
                entry[6] = entry[6][1:]
                char = entry[6][0:1]
        char = entry[6][0:1]
        for key, value in HandshapeThumbPositionDict.items():
            if (char == value):
                nondominanhandthumbposition[i] = key
                entry[6] = entry[6][1:]
        char = entry[6][0:1]
        for key, value in HandshapeBendingDict.items():
            if (char == value):
                nondominanhandbending[i] = key
                entry[6] = entry[6][1:]
        #for some entries thumb is placed after the bending sign, so let's go back to thumb
        char = entry[6][0:1]
        for key, value in HandshapeThumbPositionDict.items():
            if (char == value):
                nondominanhandthumbposition[i] = key
                entry[6] = entry[6][1:]
        char = entry[6][0:1]
        if (char == ""):
            entry[6] = entry[6][1:]

#Remove signs that may appear in beetwen 
for entry in listofentries:
    for i in range(3):
        char = entry[6][0:1]
        for key, value in UnknownSymbols2Dict.items():
            if (char == value):
                entry[6] = entry[6][1:]

HandpositionFingerDirectionDict = {
    "0" : "",
    "1" : "",
    "2" : "",
    "3" : "",
    "4" : "",
    "5" : "",
    "6" : "",
    "7" : "",
    "8" : "",
    "9" : "",
    "10" : "",
    "11" : "",
    "12" : "",
    "13" : "",
    "14" : "",
    "15" : "",
    "16" : "",
    "17" : "",
}

if (args.logging):
    print("5. Handposition - extended finger direction:")

#Remove signs that may appear in beetwen symmetry operator and handshape - base form
for entry in listofentries:
    char = entry[6][0:1]
    for key, value in UnknownSymbols1Dict.items():
        if (char == value):
            entry[6] = entry[6][1:]

for entry in listofentries:
    char = entry[6][0:1]
    for key, value in HandpositionFingerDirectionDict.items():
        if (char == value):
            entry.append(key)
            entry[6] = entry[6][1:]
    if (len(entry) == 11):
        entry.append("ERROR")

if (args.logging):
    for entry in listofentries:
        print(entry[0] + ": " + entry[5] + " -> " + entry[6] + "(" + entry[11] + ")")
        if (len(entry) > 12):
            print("Critical error")
            sys.exit(0)

#Remove if dupicated
for entry in listofentries:
    char = entry[6][0:1]
    for key, value in HandpositionFingerDirectionDict.items():
        if (char == value):
            entry[6] = entry[6][1:]

if (args.logging):
    print("5. Handposition - Thumb:")

HandpositionPalmOrientationDict = {
    "0" : "",
    "1" : "",
    "2" : "",
    "3" : "",
    "4" : "",
    "5" : "",
    "6" : "",
    "7" : "",
}

for entry in listofentries:
    char = entry[6][0:1]
    for key, value in HandpositionPalmOrientationDict.items():
        if (char == value):
            entry.append(key)
            entry[6] = entry[6][1:]
    if (len(entry) == 12):
        entry.append("ERROR")

if (args.logging):
    for entry in listofentries:
        print(entry[0] + ": " + entry[5] + " -> " + entry[6] + "(" + entry[11] + ")")
        if (len(entry) > 13):
            print("Critical error")
            sys.exit(0)

if (args.logging):
    print("non dominant hand analysis")

for entry,i in zip(listofentries,range(len(listofentries))):
    char = entry[6][0:1]
    if (char == "" or char == ""):
        entry[6] = entry[6][1:]
        char = entry[6][0:1]
        for key, value in HandshapeBaseformsDict.items():
            if (char == value):
                nondominanhandbaseform[i] = key
                entry[6] = entry[6][1:]
                char = entry[6][0:1]
        char = entry[6][0:1]
        for key, value in HandshapeThumbPositionDict.items():
            if (char == value):
                nondominanhandthumbposition[i] = key
                entry[6] = entry[6][1:]
        char = entry[6][0:1]
        for key, value in HandshapeBendingDict.items():
            if (char == value):
                nondominanhandbending[i] = key
                entry[6] = entry[6][1:]
        char = entry[6][0:1]
        for key, value in HandpositionFingerDirectionDict.items():
            if (char == value):
                nondominanhandextfingdirect[i] = key
                entry[6] = entry[6][1:]
        char = entry[6][0:1]
        for key, value in HandpositionPalmOrientationDict.items():
            if (char == value):
                nondominanhandpalmorient[i] = key
                entry[6] = entry[6][1:]
        char = entry[6][0:1]
        if (char == ""):
            entry[6] = entry[6][1:]

if (args.logging):
    for entry in listofentries:
        if (len(entry) > 13):
            print("Critical error")
            sys.exit(0)

#remove some signs
UnknownSymbols3Dict = {
    "0" : "",
    "1" : "",
    "2" : "",
    "4" : "",
    "5" : "",
}

for entry in listofentries:
    char = entry[6][0:1]
    for key, value in UnknownSymbols3Dict.items():
        if (char == value):
            entry[6] = entry[6][1:]

if (args.logging):
    print("hand location - fronal plane r/l")

HandlocationFronalPlaneRL = {
    "0" : "", #left to the left
    "1" : "", #left
    #"2" : "", #center
    "3" : "", #right
    "4" : "", #right to the right
}

for entry in listofentries:
    char = entry[6][0:1]
    if (char == ""):
        entry.append("0")
        entry[6] = entry[6][1:]
    elif (char == ""):
        entry.append("1")
        entry[6] = entry[6][1:]
    else:
        entry.append("2")

if (args.logging):
    for entry in listofentries:
        print(entry[0] + ": " + entry[5] + " -> " + entry[6] + "(" + entry[13] + ")")
        if (len(entry) > 14):
            print("Critical error")
            sys.exit(0)

if (args.logging):
    print("hand location - fronal plane t/b")

HandLocationFronalPlaneTB = {
#   "0" : None,   
    "1" : "",
    "2" : "",
    "3" : "",
    "4" : "",
    "5" : "",
    "6" : "",
    "7" : "",
    "8" : "",
    "9" : "",
    "10" : "",
    "11" : "",
    "12" : "",
    "13" : "",
    "14" : "",
    "15" : "",
    "16" : "",
    "17" : "",
    "18" : "",
    "19" : "",
    "20" : "",
    "21" : "",
    "22" : "",
    "23" : "",
    "24" : "",
    "25" : "",
    "26" : "",
    "27" : "",
    "28" : "",
    "29" : "",
    "30" : "",
    "31" : "",
    "32" : "",
    "33" : "",
    "34" : "",
    "35" : "",
}

MovementSigns = {
    "0" : "",
    "1" : "",
    "2" : "",
    "3" : "",
    "4" : "",
    "5" : "",
    "6" : "",
    "7" : "",
    "8" : "",
    "9" : "",
    "10" : "",
    "11" : "",
    "12" : "",
    "13" : "",
    "14" : "",
    "15" : "",
    "16" : "",
    "17" : "",
    "18" : "",
    "19" : "",
    "20" : "",
    "21" : "",
    "22" : "",
    "23" : "",
    "24" : "",
    "25" : "",
    "26" : "",
    "27" : "",
    "28" : "",
    "29" : "",
    "30" : "",
    "31" : "",
    "32" : "",
    "33" : "",
    "34" : "",
    "35" : "",
    "36" : "",
    "37" : "",
    "38" : "",
    "39" : "",
    "40" : "",
    "41" : "",
    "42" : "",
    "43" : "",
    "44" : "",
    "45" : "",
    #
}

HandLocationDistanceDict = {
    "0" : "",
    "1" : "",
    "2" : "",
#   "3" : None
    "4" : "",
    "5" : "",
    "6" : "",
}

for entry in listofentries:
    char = entry[6][0:1]
    for key, value in HandLocationFronalPlaneTB.items():
        if (char == value):
            entry.append(key)
            entry[6] = entry[6][1:]
    for key, value in MovementSigns.items():
        if (char == value):
            entry.append("0")
    for key, value in HandLocationDistanceDict.items():
        if (char == value):
            entry.append("0")
    if (char == ""):
            entry.append("0")
    #if the sign was not recognized try to check the next symbol
    if (len(entry) == 14):
        char = entry[6][1:2]
        for key, value in MovementSigns.items():
            if (char == value):
                entry.append("0")
    if (len(entry) == 14):
        char = entry[6][1:2]
        for key, value in HandLocationDistanceDict.items():
            if (char == value):
                entry.append("0")
    if (len(entry) == 14):
        entry.append("ERROR")

if (args.logging):
    for entry in listofentries:
        print(entry[0] + ": " + entry[5] + " -> " + entry[6] + "(" + entry[14] + ")")
        if (len(entry) > 15):
            print("Critical error")
            sys.exit(0)

DescriptionDict = {
    "1" : "ID",
    "2" : "dunno",
    "3" : "dunno",
    "4" : "dunno",
    "5" : "word",
    "6" : "Hamnosys",
    "7" : "Hamnosys copy co work on",
    "8" : "Symmetry operator",
    "9" : "Dom hand - Handshape - base form",
    "10" : "Dom hand - Handhape - Thumb position",
    "11" : "Dom hand - Handshape - Bending",
    "12" : "Dom hand - Handposition - Ext finger direction",
    "13" : "Dom hand - Handposition - Palm orientation",
    "14" : "Dom hand - Handlocation - Frontal plane l/r",
    "15" : "Dom hand - Handlocation - Frontal plane t/b",
    "16" : "Dom hand - Handlocation - Distance from the body",
} 

f = open("/dih4/dih4_2/hearai/data/frames/pjm/annotations_hamnosys.txt", "w")
for key,value in DescriptionDict.items():
    f.write(value + ";")
f.write("\n")

for entry in listofentries:
    for i in range(len(entry)):
        f.write(entry[i] + ";")
    f.write("\n")
f.close()

