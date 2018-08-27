#Testing program for all benchmarks
#To run input the program with the args
#	benchmark name in lowercase + tripcount
import subprocess
import csv
import sys
import os
import math

#Number of iterations to run all tests on
iterations = 10

#Additional arguments for specific benchmarks that require them
sradCompileArgs = "-lm"
sradRunArgs = "0.5 512 512 4"		#Change "512 512" to dimensions of image used
ludCompileArgs = "LUD/common.c -lm"
ludRunArgs = "-s"

#The function that initiates each benchmark run and reads the run time results for printing
def run_command(command):
    os.system(command)

    oname = 'p'
    output = open(oname,'r')
    time = 0
    
    #Reads the given run time output, account for different levels (s, ms, ns, etc)
    for line in output:
        if "GPU Runtime" in line:
            lineParts = line.split()
            if "ms" in lineParts[2]:
                time += float(lineParts[2][:-2])*0.001
            elif "us" in lineParts[2]:
                time += float(lineParts[2][:-2])*0.000001
            elif "ns" in lineParts[2]:
                time += float(lineParts[2][:-2])*0.000000001
            else:
                time += float(lineParts[2][:-1])
            break
    output.close()
    return time

#Main function for the program    
def main():
    n = str(sys.argv[1])
    size = str(sys.argv[2])

    #Adds all the tested files to a list for testing, with a related boolean value to
    #ensure that any cuda code is also handled for asynchronous tests
    versions = list()
    usesLibrary = list()
    versions.append(n+".c")
    usesLibrary.append(False)
    versions.append(n+"K.c")
    usesLibrary.append(False)
    versions.append(n+"KE.c")
    usesLibrary.append(False)
    versions.append(n+"KG.c")
    usesLibrary.append(False)
    versions.append(n+"KEG.c")
    usesLibrary.append(False)
    versions.append(n+"KA.c")
    usesLibrary.append(True)
    versions.append(n+"KEGA.c")
    usesLibrary.append(True)

    #Creates any compile and run time arguments based on the benchmark
    compileArgs = ""
    runArgs = ""
    if (n == "srad"):
        compileArgs += sradCompileArgs
        runArgs += size+" "+sradRunArgs
    elif (n == "lud"):
        compileArgs += ludCompileArgs
        runArgs += ludRunArgs+" "+size

    #The run command used
    runCommand = "./a.out "+runArgs+" >p 2>&1"

    #Variables used for the run results
    means = list()
    variance = list()
    version = 0

    #Loop runs through each benchmark version to perform testing on each
    for name in versions:
        #Chooses the correct compilation command in case of cuda code being present
        if (usesLibrary[version]):
            os.system("clang -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -lcudart -I/usr/local/cuda/include/ "+compileArgs+" -DSIZE="+size+" "+n.upper()+"/"+name)
        else:
            os.system("clang -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda "+compileArgs+" -DSIZE="+size+" "+n.upper()+"/"+name)

        #Establish all variables to run the iterations
        times = list()
        r = 0
        i = 1
        print("Testing: "+name)

        #Runs the benchmark version for the given number of iterations
        #Records the run time and adds the value to the overall time to calculate the mean
        while i <= iterations:
            time = run_command(runCommand)
            times.append(time)
            r += time
            os.system("rm p")
            print("Run %2d Time: %12.6f" % (i,time))
            i += 1
    
        #Calculates the mean and the variance percentage for the versions iterations
        mean = r/float(iterations)
        vrcPer = 0
        for time in times:
            diff = mean - time
            vrcPer += diff * diff
        vrcPer = math.sqrt(vrcPer/float(iterations))/mean
        means.append(mean)
        variance.append(vrcPer*100)
        version += 1

    #Prints out all benchmark version results in a dashboard
    i = 0
    print("\n----------------{:^19}----------------".format(n+" Results"))
    print("|   Version   |     Mean     | Variance | Speedup |")
    print("---------------------------------------------------")
    for name in versions:
        print("|%13s|%14.6f|%10.4f|%9.4f|" % (name,means[i],variance[i],means[0]/means[i]))
        i += 1
    print("----------------------{:^7}----------------------\n".format(size))

    #Removes any leftover files from benchmark execution
    os.system("rm a.out")
    if (n == "srad"):
        os.system("rm image_out.pgm")

main()
