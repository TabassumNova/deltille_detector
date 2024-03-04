import subprocess
output = subprocess.check_output(["./build/apps/deltille_detector","-t", "./test/ico_deltille.dsc", "-f", "./test/p2.png", "-o", "./test", "-s"])
# print(output)
pass