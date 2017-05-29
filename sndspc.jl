#=
sndspc.jl: Pattern Spaces for Sound

Josh Berson, josh@joshberson.net
May 2017

What:
- Extract spectral envelope features from short segments of an audio recording
- Embed feature space in a lower-dim space suitable for visualization and learning
- Use TensorFlow to classify sounds? Need labeling

Why:
- Explore ways of heuristically representing phenomenologically significant
  features of sound and observing how they change over time

Inspiration:
- http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0164783
- https://lust.nl/#projects-7158
- http://www.seaandsailor.com/audiosp_julia.html
- Dupont and Ravet, Improved audio classification using a novel non-linear
  dimensionality reduction ensemble approach (2013)
=#

using StatsBase # variance, skew, kurtosis, etc
using MultivariateStats # ICA
#using Learn # ObjectiveFunction req Julia ≥ 0.6
#using TensorFlow # as above

#using GLVisualize # as above
#using GR # as above
using Plots

using SampledSignals
using LibSndFile
using DSP
using MFCC


sourcePath = "/Users/josh/Sync/Recordings/LaosFebMar2016/lp.wav"


function extractSamples( path, step = 1.5, n = 2 )
  #=
  Stream an audio file, create an array of samples, step * n seconds/sample
  Returns (samples, exception or 0)
  =#

  samples = []
  exc = 0
  try
    source = loadstream(path)
    sr = samplerate(source)

    step = Int64((step * sr)) # Switch from seconds to frames
    n = max(1, floor(n)) # Whole # steps/sample

    # Extract overlapping samples from the source
    # Overlap is (n - 1)/n steps, i.e., consume one new step/sample
    samp = []
    append!(samp, read(source, (n - 1) * step))
    for i in 2 : nframes(source) / step
      append!(samp, read(source, step))
      #@show samp
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

  mfmean = mean.(cols)
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

S, exc = extractSamples(sourcePath)
@show size(S)
@show exc

#X, Xdict = extractFeatures.(S)
#@show size(X)

# https://multivariatestatsjl.readthedocs.io/en/latest/ica.html
# “X – The data matrix, of size (m, n). Each row corresponds to a mixed signal,
#  while each column corresponds to an observation (e.g all signal value at a
#  particular time step).”

#ica = fit(ICA, X', 10) # FIXME: X' or X?

#@show size(ica.W)

# NOTES
# Huber penalty to regress with outliers/noisy samples?
