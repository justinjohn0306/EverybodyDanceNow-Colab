import glob
import os
import re
import sys

def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None
  
# Function to rename multiple files 
def main(mypath, name_stub, len_num): 
	count = 0 
	files = glob.glob(mypath+"*.png")
	for current in files:
		
		number_trailing = re.findall('\d+',current )
		number_trailing = [s.lstrip("0") for s in number_trailing]
		print(current, mypath+name_stub+str(number_trailing[0]).zfill(len_num)+"_keypoints")
		#print(number_trailing)
		os.rename(current, mypath+name_stub+"_"+str(count).zfill(len_num)+".png") 
		count = count + 1
  
# Driver Code 
# Calling main() function, run command ffmpeg -i output_%4d.png  output.avi
print(sys.argv)
main(sys.argv[1], "output",4 ) 
