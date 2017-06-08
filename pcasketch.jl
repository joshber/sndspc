function embed( X; varratio=.99 )
  #=
  X is (n, d): n samples, d dimensions

  Returns:
  * A projection of normalized X onto the first k principal directions,
    where those k directions account for varratio of the total variance
  * The first k principal directions as a matrix, with PDs in columns

  Principal component analysis is one of those things that makes less sense
  the more you think about it. Samples in rows or in columns, decompose the
  sample matrix or the covariance matrix, choice of decomposition (eig vs svd),
  efficiency and numerical stability of different strategies. I've found these
  helpful:

  https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
  https://arxiv.org/abs/1404.1100
  https://stats.stackexchange.com/questions/79043/why-pca-of-data-by-means-of-svd-of-the-data

  Strategy:
  * Normalize X
  * Get correlation matrix (or kernel-based similarity matrix)
  * Decompose with SVD
  * Eigenvalues = diagm(Σ) — Sum to find # directions to meet variance threshold
  * Projection = XV
  =#

  # Translate and scale to Xν ~ N(0, 1)
  n = size(X, 1)
  Xµ = X - repmat(mean(X, 1), n) # Set µ = 0
  Xν = Xν ./ repmat(std(X, 1), n) # Set σ = 1

  sim = X'X ./ (n - 1) # TODO: This is where we'll swap in the Gaussian kernel

  U, Σ, V = svd(sim) # TODO: svdfact, then extract Σ and V — we don't need U

  # Determine how many components we need in order to preserve
  # the desired proportion of variance
  # Σ comes sorted in descending order
  totalvar = sum(Σ)
  pvar = 0.0
  k = 0
  while pvar / totalvar < varratio
    k += 1
    pvar += var[k]
  end

  # FIXME: Equivalent to Xν * V? CHECK IN CONSOLE
  # Equivalent when truncated before rather than after multiplication?
  U[:,1:k] * Σd[1:k,1:k], V[:,1:k]
end
