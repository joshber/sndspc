#=
FIXME--
- Fewer SF features -- ≤12KHz, ≥200?, more fband resolution, use spectrogram defaults
- Cut feature vector to suit fs
=#
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
- Nadav Hochman's work on style spaces for image social media
- Dupont and Ravet, Improved audio classification using a novel non-linear
  dimensionality reduction ensemble approach (2013)
=#

using StatsBase, MultivariateStats
using WAV, DSP, MFCC
#using GR, GLVisualize # ObjectiveFunction needs v0.6
using Plots


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
  # Array{Float64, 2}: stereo sampling
  p = 1
  n = Int64(floor((size(source, 1) - (len - step)) / step))
  samples = Array{Array{Float64, 2}}(n)
  for i in 1:n
    samples[i] = source[p : p + len - 1, 1:2]
    p += step
  end

  samples, fs
end

function extractFeatures( sample, fs, sfmin, sfmax )
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
  # Default is 63 filterbanks — adjust with kwarg nbands
  # augtype=::delta means get first and second derivatives too
  mfc = feacalc(sample; sr=fs, augtype=:ddelta, sadtype=:none, normtype=:none, defaults=:wbspeaker, numcep=13)
  mfc = mfc[1] # Discard metadata

  #
  # Now get MFCC mean, variance, skewness, and kurtosis

  # Extract columns. Each column = framewise values for one coefficient
  # size(cols) == 39
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
  spec = spectrogram(mono; fs=fs, window=hanning)
    # Default STFT frame span is length(signal)/8). Default overlap is .5
  pow = power(spec)

  # Log-bin the power spectrum down to sfmin Hz
  fbands = size(pow,1)
  binshifts = logbins(fs, sfmin, sfmax)
    # N.b., assert(fbands >> binshifts[end] ≥ 2) — no problem for reasonable sfmax
  nbins = length(binshifts)
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
  # For subarray bins[i], rows = powers at different frequencies (F domain),
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

  [ mfµ; mfσ2; mfγ; mfkur; sfµ; sfσ2; sfγ; sfkur ]
end

function logbins( fs, fmin, fmax )
  b = map( i -> fmin ≤ Int64(fs / 2) >> i ≤ fmax, 1:11 )
  [ i for i in 1:11 if b[i] ]
end

function constructFeatureSpace( samples, fs )
  #=
  Dot vectorization would yield a vector of vectors
  What we need is a two-dimensional array
  =#
  sfmin, sfmax = 200, 12000
  # 144 MFCC features + 4n for spectral flatness
  # where n is # log bins with lower bound in [sfmin,sfmax]Hz
  nfeatures = 144 + 4 * length(logbins(fs, sfmin, sfmax))
  X = Array{Float64, 2}(length(samples), nfeatures)
  for i in 1:length(samples)
    X[i, 1:end] = extractFeatures(samples[i], fs, sfmin, sfmax)[1:end]
  end
  X
end

function main()
  sourcePath = "/Users/josh/Sync/Recordings/LaosFebMar2016/lp.wav"

  # TODO: Need higher resolution in the time domain, say step=.1s
  # But that froze my MBA
  X = constructFeatureSpace(extractSamples(sourcePath, 3, 1)...)
  pca = fit(PCA, X, 10)
  @show principalratio(pca)
    # TODO: Select PCA dim to capture 95 percent of covariance

  # TODO: KPCA with Gaussian kernel to capture nonlinearities in the feature space
  # TODO: Plot — 3D animation of PCs against time, with PC value on vertical
end

main()

# FIXME: Is ICA appropriate here? Kernel PCA?
# ICA does not seem really appropriate, since the different dimensions of the feature space
# here do not correspond to independent series of observations along the lines of
# sources in a cocktail party problem or regions of an image
# Maybe FIRST apply PCA/svd to reduce dimensionality, THEN apply ICA?
# * PCA to reduce dimensionality
# * ICA to maximize statistical independence / joint entropy among basis vectors
# https://stats.stackexchange.com/questions/97704/does-ica-require-to-run-pca-first
# * How do you decide optimal number of ICs?
# https://stats.stackexchange.com/questions/94463/what-are-the-advantages-of-kernel-pca-over-standard-pca

# Kernel PCA -- nonlinear dim reduction

# NOTES
# Huber penalty to regress with outliers/noisy samples?
