import extractFeaturesBulk
import matchFeatures
import FGCT
import time

dthr = 280
alpha = 1.2
sigma = 0.25
num_tri = 30



print("Extracting logo image SIFT features... \n")
logopath = 'logos'
logoImages = extractFeaturesBulk.extractFeaturesBulk(logopath)
print("Done \n")
print("Extracting test image SIFT features... \n")
testpath = 'test'
testImages = extractFeaturesBulk.extractFeaturesBulk(testpath)
print("Done \n")
for i in range(len(testImages)):
    for j in range(len(logoImages)):
        print('\n')
        print("********************************************************")
        t = time.process_time()
        pairs = matchFeatures.matchFeatures(logoImages[j], testImages[i], dthr)
        elapsed_time =  time.process_time()-t
        print("Pairs created in ",elapsed_time)
        t = time.process_time()
        Dab,a,b = FGCT.extract_triangles (testImages[i], logoImages[j], pairs, num_tri)
        elapsed_time =  time.process_time()-t
        print("Triangles  created in ",elapsed_time)
        t = time.process_time()
        correspodence1,correspodence2,consistensy,Dtemp = FGCT.FGCT(Dab, alpha, sigma)
        elapsed_time =  time.process_time()-t
        print("FGCT  elapsed time is ",elapsed_time)
        print("Correspodence 1 is", correspodence1)
        print("Correspodence 2 is", correspodence2)
        print("********************************************************")
        print('\n')
        FGCT.plot_triangles(logoImages[j],testImages[i],consistensy,Dtemp,a,b, pairs)

