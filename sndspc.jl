# Load in a recording
# Split it into 10s frames with 5s step
# Get MFCCs and SFs
# Optional: Reduce with ICA -- 10 dof? ... and t-SNE?
# Cluster with k-means and EM
# Classify with SVM -- what's the target dimension?

using LibSndFile
using MFCC
