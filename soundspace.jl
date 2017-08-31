#=
soundspace.jl: Pattern Spaces for Sound

TODO Next steps:
* Exclude SF from features
* Filters: Lowpass, Bandpass
* Try t-SNE: TSne.jl req ≥ v0.6 https://distill.pub/2016/misread-tsne/

Josh Berson, josh@joshberson.net
May, August–September 2017

What:
1. Extract spectral envelope features from short segments (shingles) of an audio recording
2. Project feature space into a lower-dim space suitable for visualization and learning
3. Aug 2017: Sort segments in projected feature space and concatenate —
   a form of procedural composition using (ambient) recorded sound as input
4. TODO: Render an animated PHENOMENOGRAM of projected feature space for the original source

Why:
* Explore ways of heuristically representing phenomenologically significant
  features of sound and observing (and auditioning) how they change over time

Inspirations:
* Nadav Hochman's work on style spaces for image social media
  http://nadavhochman.net/
* Dupont and Ravet, Improved audio classification using a novel non-linear
  dimensionality reduction ensemble approach (2013)
  Idea of combining MFCCs and fbanded spectral flatness then projecting
  comes from Dupont and Ravet
* Hiroki Sasajima, Colony (Impulsive Habitat, IHab040, 2012)
  http://www.impulsivehabitat.com/releases/ihab040.htm
  (An album of ambient insect noise)
* See also Berson, “Sound and Pain” https://goo.gl/Qn2HTI
=#

using StatsBase, Distances, ManifoldLearning
#using TSne # Requires Julia ≥ v0.6

using WAV, DSP, MFCC
  # N.b.: MFCC does not work with Julia v0.6 — error in rasta.jl

#using GLVisualize, GLAbstraction, Reactive


function extractShingles( path; len=1, fps=10 )
  #=
  Read an audio file, construct an array of shingles
  Returns shingles, sample rate
  =#

  source, fs = wavread(path)
  f = digitalfilter(Bandpass(300, 8000; fs=fs), Butterworth(2))
  #f = digitalfilter(Lowpass(8000; fs=fs), Butterworth(2))
  filt!(source, f, source)

  # Switch from seconds to frames
  len = Int64(floor(len * fs))
  step = Int64(floor(1/fps * fs))

  # Extract shingles of length len at intervals step
  # Array{Float64,2}: stereo sampling
  p = 1
  n = Int64(floor((size(source, 1) - (len - step)) / step))
  shingles = Array{Array{Float64,2}}(n)
  for i in 1:n
    shingles[i] = source[p : p + len - 1, 1:2]
    p += step
  end

  shingles, fs, len/fs, fps
end

function extractFeatures( shingle, fs, sfmin, sfmax )
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
  mfc = feacalc(shingle; sr=fs, augtype=:ddelta, sadtype=:none, normtype=:none, defaults=:wbspeaker, numcep=13)
  mfc = mfc[1] # Discard metadata

  #
  # Now get MFCC mean, variance, skewness, and kurtosis

  # Extract columns. Each column = framewise values for one coefficient
  # size(cols) == 39
  cols = [ view(mfc,:,c) for c in 1:size(mfc,2) ]

  # Without the ., ~() would pull elements from across the columns, yielding rowwise ~
  # Use .(cols) for mean and var even though there's an optional dimension arg,
  # otherwise the vector has the wrong orientation for concatenation at the end
  mfµ = mean.(cols)
  mfσ2 = var.(cols)
  mfγ = skewness.(cols)
  mfkur = kurtosis.(cols)

  #
  # Spectral flatness

  # TODO: More sophisticated approach to channel blending:
  # Sum/take higher-energy channel if ≥ .95 energy (say) is in one channel

  mono = (shingle[:,1] .+ shingle[:,2]) ./ 2 # Average channels to get mono signal
  spec = spectrogram(mono; fs=fs, window=hanning)
    # Default STFT frame span is length(signal)/8). Default overlap is .5
  pow = power(spec)

  # Log-bin the power spectrum down to sfmin Hz
  fbands = size(pow,1)
  binshifts = logbins(fs, sfmin, sfmax)
    # !! assert(fbands >> binshifts[end] ≥ 2) — no problem for reasonable sfmax
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
    means = mean(bins[i], 1) # mean.(cols)
    geomeans = geomean.(cols)
    sf = geomeans ./ means # size(sf) == (1,nbins)
    sfµ[i] = mean(sf)
    sfσ2[i] = var(sf)
    sfγ[i] = skewness(sf)
    sfkur[i] = kurtosis(sf)
  end
  # length(sfµ) == nbins, same for higher moments

  [ mfµ; mfσ2; mfγ; mfkur; sfµ; sfσ2; sfγ; sfkur ]
end

function logbins( fs, fmin, fmax )
  #=
  Returns a vector of integers giving binary left shifts from a folding frequency
  to frequencies in some range [fmin, fmax]

  The frequencies generated by these shifts serve as lower bounds on
  frequency log bins

  It's easier to use than to describe
  =#
  b = map( i -> fmin ≤ Int64(fs / 2) >> i ≤ fmax, 1:11 )
  [ i for i in 1:11 if b[i] ]
end

function constructFeatureSpace( shingles, fs )
  #=
  Dot vectorization would yield a vector of vectors
  What we need is a two-dimensional array, hence the for loop
  =#

  sfmin, sfmax = 200, 12000

  # 156 MFCC features (13 MFCCs + ∇ and Hessian · 4 moments)
  # + 4n for spectral flatness where n is # log bins with lower bound in [sfmin,sfmax]Hz
  nfeatures = 156 + 4 * length(logbins(fs, sfmin, sfmax))

  X = Array{Float64, 2}(length(shingles), nfeatures)
  for i in 1:length(shingles)
    X[i, 1:end] = extractFeatures(shingles[i], fs, sfmin, sfmax)[1:end]
  end
  X
end

function standardize( X )
  #=
  X is (n, d): n samples, d dimensions
  Returns:
  * X translated and scaled to N(0, 1)
  * n
  =#
  n = size(X,1)
  ( X - repmat(mean(X, 1), n) ) ./ repmat(std(X, 1), n), n
end

function projectPCA( X; minvar=.99 )
  #=
  X is (n, d): n samples, d dimensions

  Performs PCA on X

  Returns:
  * A projection of standardized X onto the first k principal directions,
    where those k directions account for at least minvar fraction of the total variance
  * The first k principal directions as a matrix, with PDs in columns
    (projection basis)

  Principal component analysis is one of those things that makes less sense
  the more you think about it. Confusion abounds: Samples in rows or in columns,
  decompose the sample matrix or the covariance matrix, choice of decomposition
  (eig vs svd), efficiency and numerical stability of different strategies.
  I've found these helpful:

  https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
  https://arxiv.org/abs/1404.1100
  https://stats.stackexchange.com/questions/79043/why-pca-of-data-by-means-of-svd-of-the-data
  =#

  Xs, n = standardize(X)

  # Construct correlation (standardized covariance) matrix
  corr = Xs'Xs / (n - 1)

  F = svdfact(corr)
  S, V = F[:S], F[:V]

  # Determine how many principal directions we need
  # in order to preserve the desired proportion of variance
  # S comes sorted in descending order
  minvar = min(minvar, 1.0)
  varsums = [ sum(S[1:i]) for i in 1:length(S) ]
  k = find( pvar -> pvar / varsums[end] ≥ minvar, varsums)[1]
  #=
  # Less idiomatic but eliminates redundant summing and consing
  totalvar = sum(S)
  pvar = 0.0
  k = 0
  while pvar / totalvar < minvar
    k += 1
    pvar += S[k]
  end
  =#

  Xs * V[:,1:k], V[:,1:k]
end

function projectRBF( X; minvar=.99, gamma=10.0 )
  #=
  Radial Basis Function (Gaussian) kernel

  http://sebastianraschka.com/Articles/2014_kernel_pca.html
  http://www.eric-kim.net/eric-kim-net/posts/1/kernel_trick.html
  https://stats.stackexchange.com/questions/131138/what-makes-the-gaussian-kernel-so-magical-for-pca-and-also-in-general
  https://stats.stackexchange.com/questions/168051/kernel-svm-i-want-an-intuitive-understanding-of-mapping-to-a-higher-dimensional/168082

  Caveats:
  * Kernel is n^2, whereas correlation is d^2, and n >> d
  * Kernel-based method is nonparametric — Not good for imputing new samples

  Strategy:
  * Standardize X
  * Get similarity matrix (kernel-based)
  * Decompose with SVD
  * Eigenvalues = S — Sum to find # directions to meet variance threshold
  * Projection = XV (standardized X)

  TODO: Not yet working, or at least, I don't yet know how to interpret the result
  I.e., I seem to get zero dimensionality reduction — conserved covariance requires
  the entire transformed array, which is of Dim > Dim(X)
  =#

  L22 = pairwise(SqEuclidean(), X') # Squared L2 norms. pairwise() operates on columns
  K = exp.(-gamma * L22)

  # 0-center the kernel matrix
  cntr = ones(n, n) / n
  K = K - cntr * K - K * cntr + cntr * K * cntr
    # TODO: Check output against other centering formulas

  F = svdfact(K)
  S, V = F[:S], F[:V]

  # Determine how many principal directions we need
  # in order to preserve the desired proportion of variance
  # S comes sorted in descending order
  varsums = [ sum(S[1:i]) for i in 1:length(S) ]
  k = find( pvar -> pvar / varsums[end] ≥ minvar, varsums)[1]

  V[:,1:k]
end

function main()
  #=
  Pattern space pipeline

  1. Shingle the source
  2. Generate a feature space on a per-shingle basis
  3. Project the feature space via PCA or some nonlinear technique
     (RBF PCA, Locally Linear Embedding, etc)
  4. Sort the shingles according to values in the projected feature space
  5. Concatenate the sorted shingles to produce a new composition highlighting variation
     (we hope) in perceptually significant features of the spectral envelope
  6. TODO: Plot the projected features as a “phenomenogram” along the lines of
     http://gr-framework.org/examples/audio_ex3.html

     With Inshriach.wav as training data, shingles of 1s/5 fps, and linear PCA embedding,
     .99 variance gets us from 180 to 118 features,
     .95 to 84
     .90 to 64
     .50 to  9

     So even for low-structure ambient-type sound, the feature space is highly informative

     What to do with/about the fact that the data is highly autocorrelated?
     Does not seem to bother the fMRI people.

     TODO: See if nonlinear methods afford better compression

  TODO:
  Prepend a step to the pipeline to break the source up into manageable segments
  (i.e., two-level segmenting to prevent the pipeline from choking on, say, my 2013 MBA)
  1. Stream a long audio file
  2. Split it into (60s + mod(60s, sampleLen)) segments with (sampleLen - step) overlaps
  3. Write the segments to a folder
  4. Apply constructFeatureSpace(...) to the whole folder
  5. Concatenate the feature spaces
  6. Project the entire space at once
  =#

  path = "/Users/josh/Dropbox/Shirooni/Recordings/"
  #path = "/Users/josh/Dropbox/Recordings/LaosFebMar2016/"
  source = "小学校/170830_01"

  # Shingle the source and construct a feature space
  shingles, fs, len, fps = extractShingles("$(path)$(source).wav"; len=.5, fps=10)
  X = constructFeatureSpace(shingles, fs)

  #=
  Projections
  =#

  # Project into principal component space and take a minimum-variance subspace
  Xpca, pcaBasis = projectPCA(X; minvar=.5)
  @show size(X)
  @show size(Xpca) # How many principal directions needed to preserve 50 percent covariance?

  #=
  Nonlinear embedding. Achtung:
  https://github.com/wildart/ManifoldLearning.jl/issues/6
  https://stackoverflow.com/questions/33741396/non-empty-collection-error-in-manifoldlearning-jl

  Diffusion Maps and LLE seem not to be working

  t-SNE is in a separate package, versionwise mutually incompatible with MFCC
  Plus t-SNE is nonconvex and highly sensitive to perplexity and terrible at eliciting
  global structure — thought it would be nice for visualizing sound TYPES …
  =#

  # Project into an LTSA subspace of dim 5
  ltsa = transform(LTSA, X'; d=5)
  Xltsa = projection(ltsa)'

  #=
  Sort the shingles by projected feature-space values

  Doing this efficiently, i.e., O(n) time and O(1) space, is tricky:
  https://stackoverflow.com/questions/7365814/in-place-array-reordering

  For simplicity, we'll just cons up new arrays

  1. Get the permutation by sorting the projected-feature array
  2. Traverse the permutation, building a new array
  3. Concatenate and write

  Of course the elegant way to do this would be to store projections in an array
  But with just a handful, cut and paste is hardly a great blow to the ego
  =#

  # Permute the projection
  Xproj = Xpca
  projtype = "pca"
  sumsq = [ sum(i.^2) for i in [ Xproj[j,:] for j in 1:size(Xproj,1) ] ]
    # Sum of squares. TODO: This does not feel idiomatic, but I'm at a loss
  perm = sortperm(sumsq; rev=true)
  shinglesPermuted = [ shingles[perm[i]] for i in 1:length(shingles) ]

  # Write the new composition
  conc = vcat(shinglesPermuted...)
  wavwrite(conc, "$(path)out/$(source) out $(len)s $(fps)fps proj=$(projtype).wav"; Fs=fs, nbits=24, compression=WAVE_FORMAT_PCM)

  # Permute the projection
  Xproj = Xltsa
  projtype = "ltsa"
  sumsq = [ sum(i.^2) for i in [ Xproj[j,:] for j in 1:size(Xproj,1) ] ]
    # Sum of squares. TODO: This does not feel idiomatic, but I'm at a loss
  perm = sortperm(sumsq; rev=true)
  shinglesPermuted = [ shingles[perm[i]] for i in 1:length(shingles) ]

  # Write the new composition
  conc = vcat(shinglesPermuted...)
  wavwrite(conc, "$(path)out/$(source) out $(len)s $(fps)fps proj=$(projtype).wav"; Fs=fs, nbits=24, compression=WAVE_FORMAT_PCM)

  # TODO: Animated phenomenogram of projected features against time, with features on vertical
  # Cf. http://gr-framework.org/examples/audio_ex3.html
  #=anim = @animate for i in 1 : size(pca,1)
    # plot( ... ) for one frame
  end
  gif(anim, "phenomenogram.gif"; fps=featureFramerate)
  =#
end

main()
