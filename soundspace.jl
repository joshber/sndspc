#=
soundpattern.jl: Pattern Spaces for Sound

Josh Berson, josh@joshberson.net
May 2017

What:
- Extract spectral envelope features from short segments of an audio recording
- Embed feature space in a lower-dim space suitable for visualization and learning

Why:
- Explore ways of heuristically representing phenomenologically significant
  features of sound and observing how they change over time

Inspirations:
- Dupont and Ravet, Improved audio classification using a novel non-linear
  dimensionality reduction ensemble approach (2013)
=#

using StatsBase, MultivariateStats
using WAV, DSP, MFCC
#using GR # ObjectiveFunction needs v0.6
using Plots


#sourcePath = "/Users/josh/Sync/Recordings/LaosFebMar2016/lp.wav"
sourcePath = "/Users/josh/Sync/Recordings/Inshriach/Inshriach.WAV"

function extractSamples( path, len = 3, step = len/2 )
  #=
  Read an audio file, construct an array of samples
  Returns samples, sample rate
  =#

  source, fs = wavread(path)

  # Switch from seconds to frames
  len = Int64(floor(len * fs))
  step = Int64(floor(step * fs))

  # Extract samples of length len at intervals step
  p = 1
  n = Int64(floor((size(source, 1) - (len - step)) / step))
  samples = Array{Array{Float64, 2}}(n)
  for i in 1:n
    samples[i] = source[p : p + len - 1, 1:2]
    p += step
  end

  samples, fs
end

function extractFeatures( sample, fs )
  #=
  Generates a feature vector for the spectral envelope of an audio signal
  Features:
  * MFCC coefs 1–13: mean, variance, skewness, kurtosis
  * As above for MFCC gradient and Hessian
  * Spectral flatness in log bands: mean, variance, skewness, kurtosis

  First MFC coefficient is log power, less useful for speech recognition
  but possibly useful for pattern space visualization

  Returns a feature vector
  =#

  # See https://github.com/davidavdav/MFCC.jl
  # Default is 63 filterbanks -- adjust with kwarg nbands
  mfc = feacalc(sample; sr=fs, augtype=:ddelta, sadtype=:none, normtype=:none, defaults=:wbspeaker, numcep=13)
  mfc = mfc[1] # Discard metadata

  #
  # Now get MFCC mean, variance, skewness, and kurtosis

  # Extract columns. Each column = framewise values for one coefficient
  # size(cols) = 39
  cols = [ view(mfc,:,c) for c in 1:size(mfc,2) ]

  mfµ = mean.(cols)
    # Without the ., mean() would pull elements from across the columns,
    # yielding rowwise means

  mfσ2 = var.(cols)
  mfγ = skewness.(cols)
  mfkur = kurtosis.(cols)

  #
  # Spectral flatness

  mono = (sample[:,1] .+ sample[:,2]) ./ 2 # Average channels to get mono signal
  spec = spectrogram(mono, 4096; fs=fs, window=hanning)
    # 4096 frames per STFT (default is length(signal)/8); default overlap is .5
  pow = power(spec)

  # Log-bin the power spectrum down to 20Hz
  fbands = size(pow,1)
  binshifts = [ i for i in 1:12 if Int64(fs) >> i >= 20 && fbands >> i >= 1 ]
  nbins = length(binshifts)
  @show fbands
  @show nbins
  bins = Array{Array{Float64, 2}}(nbins)
  for i in 1:nbins
    bins[i] = pow[ fbands >> binshifts[i] + 1 : fbands >> (binshifts[i] - 1), 1:end ]
      # fbands >> i + 1 bc we have one additional band, for 0.0Hz
      # This way the first (highest-frequency) bin gets one extra band
  end

  sfµ = Array{Float64}(nbins)
  sfσ2 = Array{Float64}(nbins)
  sfγ = Array{Float64}(nbins)
  sfkur = Array{Float64}(nbins)
  # For subarray b, rows = powers at different frequencies (F domain),
  # cols = powers at a frequency at different points in time (T domain)
  for i in 1:nbins
    cols = [ view(bins[i],:,c) for c in 1:size(bins[i],2) ]
    means = mean.(cols)
    geomeans = geomean.(cols)
    sf = geomeans ./ means # size(sf) == (1,nbins)
    sfµ[i] = mean(sf)
    sfσ2[i] = var(sf)
    sfγ[i] = skewness(sf)
    sfkur[i] = kurtosis(sf)
  end
  # length(sfmean) == nbins, same for higher moments

  # Return a feature vector
  # With 13 MFCCs and 11 log bins, that's 200 features
  [ mfµ; mfσ2; mfγ; mfkur; sfµ; sfσ2; sfγ; sfkur ]#=,
  Dict(
    "MFCCmean" => mfµ,
    "MFCCvariance" => mfσ2,
    "MFCCskewness" => mfγ,
    "MFCCkurtosis" => mfkur,
    "SFmean" => sfµ,
    "SFvariance" => sfσ2,
    "SFskewness" => sfγ,
    "SFkurtosis" => sfkur
  )=#
end

samples, fs = extractSamples(sourcePath)

X = extractFeatures.(samples, fs)
@show size(X)

# https://multivariatestatsjl.readthedocs.io/en/latest/ica.html
# “X – The data matrix, of size (m, n). Each row corresponds to a mixed signal,
#  while each column corresponds to an observation (e.g all signal value at a
#  particular time step).”

#ica = fit(ICA, X, 10) # FIXME: X or X'?

#@show size(ica.W)

# NOTES
# Huber penalty to regress with outliers/noisy samples?
