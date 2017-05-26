#=
sndspc.jl: Pattern Spaces for Sound

Josh Berson, josh@joshberson.net
May 2017

What:
- Extract spectral envelope features from short segments of an audio recording
- Embed feature space in a lower-dim space suitable for visualization and learning

Why:
- Explore ways of heuristically representing phenomenologically significant
  features of sound and observing how they change over time

Inspiration:
- https://lust.nl/#projects-7158
- journals.plos.org/plosone/article?id=10.1371/journal.pone.0164783
- http://www.seaandsailor.com/audiosp_julia.html
- Dupont and Ravet, Improved audio classification using a novel non-linear
  dimensionality reduction ensemble approach (2013)
=#

# Load in a recording
# Split it into samples
# Get MFCCs and SFs
# Optional: Reduce with ICA -- 10 dof? ... and t-SNE?
# Graph
# Cluster with k-means and EM
# Classify with SVM -- what's the target dimension?


using StatsBase
using MultivariateStats
using SampledSignals
using LibSndFile
using DSP
using MFCC

sourcePath = "/Users/josh/Sync/Recordings/LaosFebMar2016/lp.wav"


function extractSamples( path, step = 1.5, n = 2 )
  #=
  Stream an audio file, create an array of samples, step * n seconds/sample
  Returns samples, exception if any
  =#

  samples = []
  exc = 0
  try
    source = loadstream(path)
    sr = samplerate(source)

    step = floor(step * sr) # Switch from seconds to frames
    n = max(1, floor(n)) # Whole # steps/sample

    # Extract overlapping samples from the source
    # Overlap is (n - 1)/n steps, i.e., consume one new step/sample
    samp = []
    append!(samp, read(source, (n - 1) * step))
    while !eof(source)
      append!(samp, read(source, step))
      push!(samples, samp)
      samp = samp[step + 1 : end]
        # Non-destructively left-shift samp by one step of frames
    end

    close(source)

  catch (e)
    exc = e
    samples = []
  end

  samples, exc
end

function extractFeatures( sample )
  #=
  Generates a feature vector for the spectral envelope of an audio sample
  Features:
  * MFCC coefs 1–13: mean, variance, skew, kurtosis
  * As above for MFCC gradient and Hessian
  * Spectral flatness in log bands: mean, variance, skew, kurtosis

  First MFC coefficient is log power, less useful for speech recognition
  but possibly useful for pattern space visualization

  Returns features as (Vector, Dict)
  =#

  sig = Float64.(sample.data) # Convert FixedPoint to floating-point
  sr = samplerate(sample)

  # See https://github.com/davidavdav/MFCC.jl
  # Default is 63 filterbanks -- adjust with kwarg nbands
  mfc = feacalc(sig; sr=sr, augtype=:ddelta, sadtype=:none, normtype=:none, defaults=:wbspeaker, numcep=13)
  mfc = mfc[1] # Discard metadata

  #
  # Now get MFCC mean, variance, skew, and kurtosis

  # Extract columns. Each column = framewise values for one coefficient
  # size(cols) = 39
  cols = [ view(mfc,:,c) for c in 1:size(mfc,2) ]

  mfmean = mean.(cols) # sum.(cols) ./ size(mfc,1)
    # Without the ., mean() would pull elements from across the columns,
    # yielding rowwise means

  mfvar = var.(cols)
  mfskew = skew.(cols)
  mfkur = kurtosis.(cols)

  #
  # Spectral flatness

  mono = (sig[:,1] .+ sig[:,2]) ./ 2 # Average channels to get mono signal
  spec = spectrogram(mono, 1024, 512; fs=sr, window=hanning)
    # 1024 frames per STFT, 512-frame overlap
  pow = power(spec)

  # Log-bin the power spectrum down to 20Hz
  fbands = size(pow,1)
  bins = []
  nbins = [ i for i in 1:12 if sr >> i >= 20 ]
  for i in nbins
    push!(bins, pow[ fbands >> i + 1 : fbands >> (i - 1), : ])
      # fbands >> i + 1 bc we have one additional band, for 0.0Hz
      # This way the first (highest-frequency) bin gets one extra band
  end

  sfmean, sfvar, sfskew, sfkur = [], [], [], []
  # For subarray b, rows = powers at different frequencies (F domain),
  # cols = powers at a frequency at different points in time (T domain)
  for b in bins
    cols = [ view(b,:,c) for c in 1:size(b,2) ]
    means = mean.(cols)
    geomeans = geomean.(cols)
    flatnesses = geomeans ./ means # size(flatnesses) == (1,nbins)
    append!(sfmean, mean(flatnesses))
    append!(sfvar, variance(flatnesses))
    append!(sfskew, skew(flatnesses))
    append!(sfkur, kurtosis(flatnesses))
  end
  # size(sfmean) == nbins, same for higher moments

  # Return a tuple with feature vector and Dict
  # With 13 MFCCs and 11 log bins, that's 200 features
  (
    [ mfmean; mfvar; mfskew; mfkur; sfmean; sfvar; sfskew; sfkur ],
    Dict(
      "MFCCmean" => mfmean,
      "MFCCvariance" => mfvar,
      "MFCCskew" => mfskew,
      "MFCCkurtosis" => mfkur,
      "SFmean" => sfmean,
      "SFvariance" => sfvar,
      "SFskew" => sfskew,
      "SFkurtosis" => sfkur
    )
  )
end

S = extractSamples(sourcePath)
X = extractFeatures.(S)

# Check that size(S) is correct given length of source and step
# Check that size(X) = (size(S), 200)
